"""Enzyme's forward sweeps on the CPU target, linked into the kernels' own code.

On CUDA every derivative of a user's function is ``numba_enzyme.jvp`` of its
device function. numba-enzyme has a CPU backend too, but it is a host-side
API (a shared object called through ctypes), with no second order, no stack
arrays, and a cache keyed on a function's source text, which two closures of
the same source share. So this module takes two pieces of numba-enzyme, the
lowering of a function to LLVM IR and the forward-mode driver that wraps it
for Enzyme, and does the rest itself:

1. ``numba_enzyme.lowering.lower`` compiles ``f(y, t, p)`` with ``numba.cfunc``
   for 1-D float64 arrays ``y`` and ``p``, and checks its ABI.
2. The forward driver ``jvp_<entry>`` is synthesised for it
   (:func:`_forward_driver`), without the reverse-mode entry numba-enzyme's
   own build adds.
3. ``llvm-extract`` keeps the entry and what it calls, ``llvm-link`` adds the
   driver, and ``opt -passes=enzyme`` writes the derivative.
4. The result is checked for numba's allocator (:func:`_check_no_allocation`),
   every definition but the entry made internal and renamed apart
   (:func:`_privatise`), and the module added to a numba code library that
   the calling kernel links, so LLVM can inline the derivative where it is
   called.

The second order is Enzyme again, forward over the first-order entry
(:func:`_second_order_ir`), as ``jvp(jvp(f))`` is on CUDA.

The calling conventions are modax's, the same as the finite differences of
:mod:`modax._differences` and Enzyme's sweeps on CUDA:
``tangent(out, y, t, p, dy, dt, dp)`` and
``second(out, y, t, p, u_y, u_t, u_p, v_y, v_t, v_p, w_y, w_t, w_p)``, one
direction per call. ``y`` and ``p`` may be longer than what ``f`` reads (a
caller passes its augmented state, and the launch's parameter row carries
:data:`N_DRIVER_COLUMNS <modax._target.N_DRIVER_COLUMNS>` more columns); the
directions need cover only what it reads.

A derivative is built when numba first types a call to it, that is when the
kernel that uses it compiles, as on CUDA. It is built once per Python
function object and number of outputs, so two closures of the same source
get their own derivatives. Builds run under numba's compiler lock, which every
build in the process already takes, so two threads compiling kernels never
build the same derivative twice or link half of one.

Enzyme cannot differentiate through numba's allocator. ``cuda.local.array``
is therefore a zeroed stack array on this target (:func:`stack_array`), as it
is on the device; a local array whose size numba cannot see as a constant
falls back to ``np.zeros``, and a derivative that reaches one fails to build
with an error saying so rather than differentiating it.

Requirements: numba 0.61 and llvmlite 0.44 (numba-enzyme lowers through numba
0.61's typed-pointer IR, which the LLVM 15 tools read), ``numba-enzyme-cuda``,
and LLVM 15's ``opt``, ``llvm-link`` and ``llvm-extract`` with the Enzyme
plugin (:func:`find_tools`). :func:`unavailable` says what is missing.

Not done: an on-disk cache of Enzyme's output. Lowering, which a cache keyed
on the lowered IR could not skip, is most of a build (about 5.6 s of 6.1 s
for DISCO-EB's derivatives, against 0.4 s of Enzyme); and several directions
per call (the CUDA Jacobian's ``SEED_BATCH``), which needs a vector-mode
driver of its own.
"""

from __future__ import annotations

import itertools
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

TOOLS_ENV_VAR = "MODAX_ENZYME_TOOLS"
KEEP_ENV_VAR = "MODAX_ENZYME_KEEP"
_PLUGIN_ENV_VAR = "NUMBA_ENZYME_PLUGIN_PATH"
_TOOL_NAMES = ("opt", "llvm-link", "llvm-extract")
_PLUGIN_NAMES = ("LLVMEnzyme-15.dylib", "LLVMEnzyme-15.so")


class EnzymeUnavailable(ImportError):
    """Enzyme cannot run here: a package, a version or a tool is missing."""


class EnzymeBuildError(RuntimeError):
    """A derivative could not be built."""


# --- the tools ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Tools:
    """LLVM 15's ``opt``, ``llvm-link`` and ``llvm-extract`` and the Enzyme plugin."""

    opt: Path
    llvm_link: Path
    llvm_extract: Path
    plugin: Path
    origin: str
    """Where they were found, for messages."""


def _tools_under(root: Path, origin: str) -> tuple[Tools | None, list[str]]:
    """The tools in ``root/bin`` or ``root/llvm15/bin``, the plugin in ``root``, ``root/enzyme`` or ``root/lib``."""
    bins = [root / "bin", root / "llvm15" / "bin"]
    found: dict[str, Path] = {}
    for name in _TOOL_NAMES:
        for directory in bins:
            if (directory / name).is_file():
                found[name] = directory / name
                break
    plugin = next(
        (
            directory / name
            for directory in (root, root / "enzyme", root / "lib")
            for name in _PLUGIN_NAMES
            if (directory / name).is_file()
        ),
        None,
    )
    missing = [
        f"{name} (not in {' or '.join(map(str, bins))})"
        for name in _TOOL_NAMES
        if name not in found
    ]
    if plugin is None:
        missing.append(
            f"the Enzyme plugin LLVMEnzyme-15 (not in {root}, {root}/enzyme or {root}/lib)"
        )
    if missing:
        return None, missing
    return (
        Tools(found["opt"], found["llvm-link"], found["llvm-extract"], plugin, origin),
        [],
    )


def find_tools() -> Tools:
    """Find the tools, in this order:

    1. ``$MODAX_ENZYME_TOOLS``, a directory holding them (on macOS, what
       ``cpu_harness/build_enzyme_toolchain.sh`` builds). When it is set,
       nothing else is looked at.
    2. The ``_vendor`` directory of the ``numba-enzyme-cuda`` wheel (Linux).
    3. ``opt-15``, ``llvm-link-15`` and ``llvm-extract-15`` on ``PATH``, with
       the plugin at ``$NUMBA_ENZYME_PLUGIN_PATH``.

    Raises :class:`EnzymeUnavailable` listing what is missing where.
    """
    configured = os.environ.get(TOOLS_ENV_VAR, "").strip()
    if configured:
        tools, missing = _tools_under(
            Path(configured).expanduser(), f"${TOOLS_ENV_VAR}"
        )
        if tools is None:
            raise EnzymeUnavailable(
                f"{TOOLS_ENV_VAR}={configured} lacks " + "; ".join(missing)
            )
        return tools
    tried: list[str] = []
    try:
        import numba_enzyme

        vendor = Path(numba_enzyme.__file__).resolve().parent / "_vendor"
        tools, missing = _tools_under(vendor, "the numba-enzyme-cuda wheel")
        if tools is not None:
            return tools
        tried.append(
            f"no tools in {vendor} (a source build of numba-enzyme-cuda has none)"
        )
    except ImportError:
        pass
    on_path = {name: shutil.which(f"{name}-15") for name in _TOOL_NAMES}
    plugin = os.environ.get(_PLUGIN_ENV_VAR, "")
    if all(on_path.values()) and plugin and Path(plugin).is_file():
        return Tools(
            Path(on_path["opt"]),  # ty: ignore[invalid-argument-type]
            Path(on_path["llvm-link"]),  # ty: ignore[invalid-argument-type]
            Path(on_path["llvm-extract"]),  # ty: ignore[invalid-argument-type]
            Path(plugin),
            "PATH",
        )
    tried.append(
        "no opt-15, llvm-link-15 and llvm-extract-15 on PATH with "
        f"${_PLUGIN_ENV_VAR}; set {TOOLS_ENV_VAR} to a directory holding them"
    )
    raise EnzymeUnavailable("; ".join(tried))


_TOOLS: Tools | None = None


def tools() -> Tools:
    """:func:`find_tools`, once."""
    global _TOOLS
    if _TOOLS is None:
        _TOOLS = find_tools()
    return _TOOLS


def _run(command: list[str], step: str, label: str, record: dict) -> None:
    started = time.perf_counter()
    try:
        done = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        raise EnzymeBuildError(f"{label}: {step} could not run ({exc})") from exc
    record[f"{step}_s"] = record.get(f"{step}_s", 0.0) + time.perf_counter() - started
    if done.returncode != 0:
        raise EnzymeBuildError(
            f"{label}: {step} failed with status {done.returncode}\n"
            f"  {' '.join(command)}\n{done.stderr[-4000:]}"
        )


def _check_tools(found: Tools) -> None:
    """Run ``opt`` with the plugin on an empty module, so a tool that cannot run here says so now."""
    with tempfile.TemporaryDirectory(prefix="modax-enzyme-") as work:
        empty = Path(work) / "empty.ll"
        empty.write_text("")
        try:
            done = subprocess.run(
                [str(found.opt), f"-load-pass-plugin={found.plugin}", "-passes=enzyme",
                 "-S", str(empty), "-o", os.devnull],
                capture_output=True, text=True, timeout=120,
            )  # fmt: skip
            version = subprocess.run(
                [str(found.opt), "--version"],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise EnzymeUnavailable(
                f"{found.opt} (from {found.origin}) does not run: {exc}"
            ) from exc
    if done.returncode != 0:
        raise EnzymeUnavailable(
            f"{found.opt} with {found.plugin} (from {found.origin}) fails: {done.stderr.strip()[-500:]}"
        )
    if "LLVM version 15." not in version.stdout:
        raise EnzymeUnavailable(f"{found.opt} (from {found.origin}) is not LLVM 15")


# --- numba-enzyme's private API -----------------------------------------------------------
#
# Everything this module uses of numba-enzyme beyond its public names is here.
# ``lowering.lower`` and ``lowering.LoweringError`` are documented API; the
# driver pieces below are not (numba-enzyme-cuda 0.3.1, ``numba_enzyme/driver.py``):
#
#   _SCALAR_IR_TYPE, _EXCINFO_PTR_TYPE, _ir_type_from_str, _target_lines,
#   _Primal, _synthesise_jvp
#
# ``driver.synthesise`` would do, but it also emits the reverse-mode entry,
# which costs Enzyme a reverse sweep nothing here calls. A numba-enzyme that
# renames these fails :func:`unavailable` with the names it lacks, and the
# CPU target falls back to finite differences.

_DRIVER_NAMES = (
    "_SCALAR_IR_TYPE",
    "_EXCINFO_PTR_TYPE",
    "_ir_type_from_str",
    "_target_lines",
    "_Primal",
    "_synthesise_jvp",
)


def _check_private_api() -> None:
    from numba_enzyme import driver, lowering

    lacking = [name for name in _DRIVER_NAMES if not hasattr(driver, name)]
    lacking += [
        name for name in ("lower", "LoweringError") if not hasattr(lowering, name)
    ]
    if lacking:
        raise EnzymeUnavailable(
            "this numba-enzyme lacks "
            + ", ".join(lacking)
            + " (modax/_enzyme_cpu.py was written against 0.3.1)"
        )


def _lower(py: Callable[..., Any], arg_types: tuple, label: str) -> Any:
    """``numba_enzyme.lowering.lower``; ``None`` for a function returning a one-tuple, which it does not lower."""
    from numba_enzyme.lowering import LoweringError, lower

    try:
        return lower(py, arg_types)
    except LoweringError as exc:
        message = str(exc)
        if "at least two values" in message:
            return None
        if "unsupported kernel body" in message:
            called = re.findall(r"'(NRT_\w+)'", message) or ["numba's allocator"]
            raise _allocation_error(label, called) from exc
        raise EnzymeBuildError(
            f"{label}: numba-enzyme cannot lower it: {message}"
        ) from exc


def _forward_driver(kernel: Any) -> tuple[str, str]:
    """The IR of a module defining ``jvp_<entry>``, and that symbol: ``driver.synthesise`` without the reverse entry."""
    from llvmlite import ir
    from numba_enzyme import driver

    _, _, *flat = kernel.arg_types
    scalar = driver._SCALAR_IR_TYPE[kernel.return_type]
    storage = (
        scalar if kernel.n_outputs == 1 else ir.ArrayType(scalar, kernel.n_outputs)
    )
    module = ir.Module(name="modax_enzyme_driver")
    module.triple, module.data_layout = driver._target_lines(kernel.ir)
    function = ir.Function(
        module,
        ir.FunctionType(
            ir.IntType(32),
            [
                storage.as_pointer(),
                driver._EXCINFO_PTR_TYPE.as_pointer(),
                *(driver._ir_type_from_str(s) for s in flat),
            ],
        ),
        name=kernel.entry_symbol,
    )
    dup = ir.GlobalVariable(module, ir.IntType(32), name="enzyme_dup")
    dup.linkage = "external"
    const = ir.GlobalVariable(module, ir.IntType(32), name="enzyme_const")
    const.linkage = "external"
    primal = driver._Primal(
        kernel=kernel, module=module, function=function, dup=dup, const=const,
        scalar_type=scalar, storage_type=storage,
    )  # fmt: skip
    symbol = driver._synthesise_jvp(primal)
    return str(module), symbol


# --- whether Enzyme can run here ----------------------------------------------------------


def unavailable() -> str | None:
    """Why Enzyme cannot run on the CPU here, or ``None`` when it can."""
    import importlib.util

    import llvmlite
    import numba

    if not numba.__version__.startswith("0.61.") or not llvmlite.__version__.startswith(
        "0.44."
    ):
        return (
            f"numba {numba.__version__} / llvmlite {llvmlite.__version__} is installed, and "
            "numba-enzyme needs numba 0.61 / llvmlite 0.44"
        )
    try:
        found = importlib.util.find_spec("numba_enzyme") is not None
    except (ImportError, ValueError):
        found = False
    if not found:
        return "numba-enzyme-cuda is not installed"
    try:
        _check_private_api()
        _check_tools(tools())
    except EnzymeUnavailable as exc:
        return str(exc)
    except ImportError as exc:
        return f"numba_enzyme does not import ({exc})"
    return None


# --- stack arrays -------------------------------------------------------------------------


def _literal_dims(shape: Any) -> tuple[int, ...] | None:
    from numba.core import types

    if isinstance(shape, types.IntegerLiteral):
        return (int(shape.literal_value),)
    if isinstance(shape, types.BaseTuple) and all(
        isinstance(s, types.IntegerLiteral) for s in shape
    ):
        return tuple(int(s.literal_value) for s in shape)
    return None


def _dtype_of(dtype: Any) -> Any:
    from numba.core import types

    if isinstance(dtype, types.NumberClass):
        return dtype.instance_type
    if isinstance(dtype, types.DType):
        return dtype.dtype
    if isinstance(dtype, types.Number):
        return dtype
    return None


def make_stack_array(fill: float = 0.0) -> Callable[..., Any]:
    """``cuda.local.array(shape, dtype)`` for the CPU: an array on the caller's stack.

    Every element starts as ``fill`` (zero, or NaN under ``MODAX_POISON``),
    the array owns no memory (a null meminfo), and nothing is allocated from
    numba's heap, so Enzyme can differentiate a function that uses one. The
    shape must be an integer or a tuple of integers numba sees as constants
    (a literal, or a closed-over int); anything else is ``np.zeros`` (or a
    NaN-filled ``np.empty``), which a function Enzyme differentiates cannot
    contain (:func:`_check_no_allocation`). ``cuda.shared.array`` is the same:
    on the CPU a block is one thread.
    """
    from llvmlite import ir
    from numba.core import cgutils, types
    from numba.extending import intrinsic

    poison = fill != 0.0

    @intrinsic(prefer_literal=True)
    def stack_array(typingctx, shape, dtype):
        dims, element = _literal_dims(shape), _dtype_of(dtype)
        if element is None:
            return None
        if dims is None:
            if isinstance(shape, types.Integer):
                ndim = 1
            elif isinstance(shape, types.BaseTuple) and all(
                isinstance(s, types.Integer) for s in shape
            ):
                ndim = len(shape)
            else:
                return None
            heap_type = types.Array(element, ndim, "C")

            def heap(context, builder, signature, args):
                if poison:

                    def make(s, d):
                        array = np.empty(s, d)
                        array.reshape(-1)[:] = np.nan
                        return array

                else:

                    def make(s, d):
                        return np.zeros(s, d)

                return context.compile_internal(builder, make, signature, args)

            return heap_type(shape, dtype), heap
        array_type = types.Array(element, len(dims), "C")

        def codegen(context, builder, signature, args):
            from numba.np.arrayobj import populate_array

            count = int(np.prod(dims)) if dims else 1
            item_type = context.get_data_type(element)
            itemsize = context.get_abi_sizeof(item_type)
            data = cgutils.alloca_once(builder, item_type, size=max(count, 1))
            if poison:
                nan = context.get_constant(element, float("nan"))
                with cgutils.for_range(
                    builder, context.get_constant(types.intp, count)
                ) as loop:
                    builder.store(nan, builder.gep(data, [loop.index]))
            else:
                cgutils.memset(
                    builder, data, ir.Constant(ir.IntType(64), count * itemsize), 0
                )
            array = context.make_array(array_type)(context, builder)
            strides, step = [], itemsize
            for d in reversed(dims):
                strides.append(step)
                step *= d
            intp = types.intp
            populate_array(
                array,
                data=data,
                shape=cgutils.pack_array(
                    builder, [context.get_constant(intp, d) for d in dims]
                ),
                strides=cgutils.pack_array(
                    builder, [context.get_constant(intp, s) for s in reversed(strides)]
                ),
                itemsize=context.get_constant(intp, itemsize),
                meminfo=None,
            )
            return array._getvalue()

        return array_type(shape, dtype), codegen

    return stack_array


# --- one build ----------------------------------------------------------------------------

# Calls of numba's runtime other than reference counting (which the lowering
# leaves to Enzyme as inactive).
_NRT_CALL = re.compile(r"\bcall\b[^\n]*@(NRT_\w+)")
_NRT_ALLOWED = frozenset({"NRT_incref", "NRT_decref", "NRT_MemInfo_call_dtor"})

BUILDS: list[dict] = []
"""One record per build in this process: what was built and how long each step took."""

_counter = itertools.count()


def _label(py: Callable[..., Any]) -> str:
    return (
        f"{getattr(py, '__module__', None) or '?'}.{getattr(py, '__qualname__', '?')}"
    )


def _check_no_allocation(ir_text: str, label: str) -> None:
    """Refuse a function that, anywhere in what it calls, allocates from numba's heap."""
    called = sorted({m.group(1) for m in _NRT_CALL.finditer(ir_text)} - _NRT_ALLOWED)
    if called:
        raise _allocation_error(label, called)


def _allocation_error(label: str, called: list[str]) -> EnzymeBuildError:
    return EnzymeBuildError(
        f"{label}: Enzyme cannot differentiate a function that allocates from "
        f"numba's heap (it calls {', '.join(called)}). On the CPU target "
        "cuda.local.array is a stack array only when its size is a constant "
        "numba can see (a literal or a closed-over int); np.zeros and np.empty "
        "always allocate. Give the size as such a constant, or set "
        "MODAX_CPU_DERIVATIVES=differences."
    )


class _Work:
    """A scratch directory for one build's IR files: removed afterwards, or kept under ``$MODAX_ENZYME_KEEP``."""

    def __init__(self, label: str):
        keep = os.environ.get(KEEP_ENV_VAR, "").strip()
        if keep:
            name = re.sub(r"[^A-Za-z0-9_.]", "_", label)[-60:]
            self.path = Path(keep) / f"{next(_counter):04d}_{name}"
            self.path.mkdir(parents=True, exist_ok=True)
            self._temporary = None
        else:
            self._temporary = tempfile.TemporaryDirectory(prefix="modax-enzyme-")
            self.path = Path(self._temporary.name)

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, *exc: object) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()


@dataclass(frozen=True)
class FirstOrder:
    """Enzyme's output for one function: IR text defining ``symbol``, the forward sweep."""

    text: str
    symbol: str
    n_out: int
    scalar: bool
    """The sweep returns its one value rather than writing an ``out`` array."""
    label: str


def _array_type() -> Any:
    from numba.core import types

    return types.Array(types.float64, 1, "C")


def build_first_order(py: Callable[..., Any], n_out: int) -> FirstOrder:
    """Lower ``py(y, t, p)`` and run Enzyme over it (steps 1 to 3 of the module docstring)."""
    import numba
    from numba.core import types

    label = _label(py)
    record: dict[str, Any] = {"what": "first order", "function": label, "n_out": n_out}
    started = time.perf_counter()
    arg_types = (_array_type(), types.float64, _array_type())
    kernel = _lower(py, arg_types, label)
    if kernel is None:
        # numba-enzyme lowers no one-tuple return: differentiate its element
        inner = numba.njit(error_model="numpy")(py)

        def element(y, t, p):
            return inner(y, t, p)[0]

        kernel = _lower(element, arg_types, label)
    record["lower_s"] = time.perf_counter() - started
    if kernel.n_outputs != n_out:
        raise EnzymeBuildError(
            f"{label}: returns {kernel.n_outputs} values where {n_out} were declared"
        )
    driver_ir, symbol = _forward_driver(kernel)
    found = tools()
    with _Work(label) as work:
        (work / "lowered.ll").write_text(kernel.ir)
        _run(
            [str(found.llvm_extract), f"--func={kernel.entry_symbol}", "--recursive",
             "--keep-const-init", str(work / "lowered.ll"), "-S", "-o", str(work / "primal.ll")],
            "llvm-extract", label, record,
        )  # fmt: skip
        _check_no_allocation((work / "primal.ll").read_text(), label)
        (work / "driver.ll").write_text(driver_ir)
        _run(
            [str(found.llvm_link), str(work / "primal.ll"), str(work / "driver.ll"),
             "-S", "-o", str(work / "combined.ll")],
            "llvm-link", label, record,
        )  # fmt: skip
        _run(
            [str(found.opt), f"-load-pass-plugin={found.plugin}", "-passes=enzyme",
             "-S", str(work / "combined.ll"), "-o", str(work / "enzyme.ll")],
            "enzyme", label, record,
        )  # fmt: skip
        text = (work / "enzyme.ll").read_text()
    record["total_s"] = time.perf_counter() - started
    BUILDS.append(record)
    return FirstOrder(text, symbol, n_out, kernel.n_outputs == 1, label)


def _second_order_ir(first: FirstOrder, entry: str) -> str:
    """``first``'s IR plus ``entry``, Enzyme's forward sweep of the forward sweep: ``D2f[u, v] + Df[w]``.

    ``entry(y, v_y, u_y, w_y, t, v_t, u_t, w_t, p, v_p, u_p, w_p, n_y, n_p, out)``.
    The first-order sweep takes ``(y, dy=u, t, dt, p, dp)``; differentiating
    it along ``v`` in its primal arguments and ``w`` in its directions gives
    the second derivative along ``(u, v)`` plus the first along ``w``.
    """
    out_type = "double" if first.scalar else f"[{first.n_out} x double]"
    sweep_type = (
        ("double" if first.scalar else "void")
        + " (double*, double*, i64, i64, i64, i64, double, double, "
        + "double*, double*, i64, i64, i64, i64"
        + ("" if first.scalar else f", {out_type}*")
        + ")*"
    )
    arguments = f"""i8* bitcast ({sweep_type} @{first.symbol} to i8*),
      i32 %dup, double* %y, double* %vy, i32 %dup, double* %uy, double* %wy,
      i32 %con, i64 %ny, i32 %con, i64 8, i32 %con, i64 %ny, i32 %con, i64 8,
      i32 %dup, double %t, double %vt, i32 %dup, double %ut, double %wt,
      i32 %dup, double* %p, double* %vp, i32 %dup, double* %up, double* %wp,
      i32 %con, i64 %np, i32 %con, i64 8, i32 %con, i64 %np, i32 %con, i64 8"""
    if first.scalar:
        body = f"""
  %r = call double (i8*, ...) @__enzyme_fwddiff_second({arguments})
  store double %r, double* %out, align 8
  ret void"""
        declaration = "declare double @__enzyme_fwddiff_second(i8*, ...)"
    else:
        body = f"""
  %tmp = alloca {out_type}, align 8
  %dtmp = alloca {out_type}, align 8
  store {out_type} zeroinitializer, {out_type}* %tmp, align 8
  store {out_type} zeroinitializer, {out_type}* %dtmp, align 8
  call void (i8*, ...) @__enzyme_fwddiff_second({arguments},
      i32 %dup, {out_type}* %tmp, {out_type}* %dtmp)
  %r = load {out_type}, {out_type}* %dtmp, align 8
  store {out_type} %r, {out_type}* %out, align 8
  ret void"""
        declaration = "declare void @__enzyme_fwddiff_second(i8*, ...)"
    return f"""{first.text}
define void @{entry}(double* %y, double* %vy, double* %uy, double* %wy,
                     double %t, double %vt, double %ut, double %wt,
                     double* %p, double* %vp, double* %up, double* %wp,
                     i64 %ny, i64 %np, {out_type}* %out) {{
entry:
  %dup = load i32, i32* @enzyme_dup, align 4
  %con = load i32, i32* @enzyme_const, align 4{body}
}}
{declaration}
"""


def build_second_order(first: FirstOrder) -> tuple[str, str]:
    """Enzyme over :func:`_second_order_ir`: the IR text and the entry's symbol."""
    entry = "modax_second_order"
    record: dict[str, Any] = {
        "what": "second order",
        "function": first.label,
        "n_out": first.n_out,
    }
    started = time.perf_counter()
    found = tools()
    with _Work(first.label + ".second") as work:
        (work / "second.ll").write_text(_second_order_ir(first, entry))
        _run(
            [str(found.opt), f"-load-pass-plugin={found.plugin}", "-passes=enzyme",
             "-S", str(work / "second.ll"), "-o", str(work / "enzyme.ll")],
            "enzyme", first.label, record,
        )  # fmt: skip
        text = (work / "enzyme.ll").read_text()
    record["total_s"] = time.perf_counter() - started
    BUILDS.append(record)
    return text, entry


# --- linking into numba -------------------------------------------------------------------


def _privatise(ir_text: str, keep: str) -> tuple[Any, str]:
    """Parse ``ir_text`` with every definition renamed apart and all but ``keep`` internal.

    Two derivatives linked into one kernel would otherwise collide on the
    helpers and constants numba emits for every function (and a derivative on
    the primal it was taken of). Declarations, Enzyme's activity markers
    among them, keep their names. Returns the module and ``keep``'s new name.
    """
    import llvmlite.binding as llvm

    suffix = f".modax_ez{next(_counter)}"
    module = llvm.parse_assembly(ir_text)
    for function in module.functions:
        if function.is_declaration:
            continue
        exported = function.name == keep
        function.name = function.name + suffix
        if not exported:
            function.linkage = "internal"
    for variable in module.global_variables:
        if variable.is_declaration:
            continue
        variable.name = variable.name + suffix
        variable.linkage = "internal"
    module.verify()
    return module, keep + suffix


def _library(ir_text: str, keep: str) -> tuple[Any, str]:
    """A finalised numba code library holding ``ir_text``, and the symbol it exports."""
    from numba.core.registry import cpu_target

    module, symbol = _privatise(ir_text, keep)
    library = cpu_target.target_context.codegen().create_library(symbol)
    library.add_llvm_module(module)
    library.finalize()
    return library, symbol


class _Lazy:
    """A value built on first use, under numba's compiler lock."""

    def __init__(self, build: Callable[[], Any]):
        self._build = build
        self._value: Any = None

    def get(self) -> Any:
        from numba.core.compiler_lock import global_compiler_lock

        if self._value is None:
            with global_compiler_lock:
                if self._value is None:
                    self._value = self._build()
        return self._value


_FIRST: dict[tuple[Any, int], _Lazy] = {}


def _python_function(fn: Any) -> Callable[..., Any]:
    return getattr(fn, "py_func", fn)


def first_order(fn: Any, n_out: int) -> _Lazy:
    """The shared, lazily built :class:`FirstOrder` of ``fn``, keyed on the function object."""
    from numba.core.compiler_lock import global_compiler_lock

    py = _python_function(fn)
    with global_compiler_lock:
        key = (py, n_out)
        if key not in _FIRST:
            _FIRST[key] = _Lazy(lambda: build_first_order(py, n_out))
        return _FIRST[key]


def clear_caches() -> None:
    """Forget the first-order builds (derivatives already linked into kernels stay there)."""
    _FIRST.clear()


def _check_array(typ: Any, what: str, label: str) -> None:
    from numba.core import errors, types

    if not (
        isinstance(typ, types.Array)
        and typ.ndim == 1
        and typ.dtype == types.float64
        and typ.layout == "C"
    ):
        raise errors.TypingError(
            f"{label}: the CPU Enzyme sweep takes {what} as a contiguous 1-D "
            f"float64 array, not {typ}"
        )


def _check_scalar(typ: Any, what: str, label: str) -> None:
    from numba.core import errors, types

    if not isinstance(typ, (types.Float, types.Integer)):
        raise errors.TypingError(f"{label}: {what} must be a number, not {typ}")


def make_tangent(fn: Any, n_out: int) -> Callable[..., None]:
    """``tangent(out, y, t, p, dy, dt, dp)``: ``out = Df(y, t, p)[(dy, dt, dp)]``, Enzyme, built on first use."""
    import numba
    from llvmlite import ir
    from numba.core import cgutils, types
    from numba.extending import intrinsic

    first = first_order(fn, n_out)
    label = _label(_python_function(fn))

    def link():
        built = first.get()
        started = time.perf_counter()
        library, symbol = _library(built.text, built.symbol)
        BUILDS.append(
            {
                "what": "link first order",
                "function": label,
                "total_s": time.perf_counter() - started,
            }
        )
        return built, library, symbol

    linked = _Lazy(link)
    f64, i64 = ir.DoubleType(), ir.IntType(64)
    p64 = f64.as_pointer()

    @intrinsic
    def sweep(typingctx, out, y, t, p, dy, dt, dp):
        for typ, what in ((out, "out"), (y, "y"), (p, "p"), (dy, "dy"), (dp, "dp")):
            _check_array(typ, what, label)
        _check_scalar(t, "t", label)
        _check_scalar(dt, "dt", label)
        built, library, symbol = linked.get()
        out_ptr = p64 if built.scalar else ir.ArrayType(f64, built.n_out).as_pointer()
        group = [p64, p64, i64, i64, i64, i64]
        function_type = ir.FunctionType(
            f64 if built.scalar else ir.VoidType(),
            group + [f64, f64] + group + ([] if built.scalar else [out_ptr]),
        )

        def codegen(context, builder, signature, args):
            kinds = signature.args
            arrays = {
                i: context.make_array(kinds[i])(context, builder, args[i])
                for i in (0, 1, 3, 4, 6)
            }

            def fields(primal, shadow):
                a, da = arrays[primal], arrays[shadow]
                return [
                    a.data, da.data, a.nitems, a.itemsize,
                    cgutils.unpack_tuple(builder, a.shape)[0],
                    cgutils.unpack_tuple(builder, a.strides)[0],
                ]  # fmt: skip

            time_value = context.cast(builder, args[2], kinds[2], types.float64)
            time_direction = context.cast(builder, args[5], kinds[5], types.float64)
            call = fields(1, 4) + [time_value, time_direction] + fields(3, 6)
            callee = cgutils.get_or_insert_function(
                builder.module, function_type, symbol
            )
            if built.scalar:
                builder.store(builder.call(callee, call), arrays[0].data)
            else:
                builder.call(callee, call + [builder.bitcast(arrays[0].data, out_ptr)])
            context.active_code_library.add_linking_library(library)
            return context.get_dummy_value()

        return types.void(out, y, t, p, dy, dt, dp), codegen

    @numba.njit(inline="always")
    def tangent(out, y, t, p, dy, dt, dp):
        sweep(out, y, t, p, dy, dt, dp)

    return tangent


def make_second_tangent(fn: Any, n_out: int) -> Callable[..., None]:
    """``second(out, y, t, p, u_y, u_t, u_p, v_y, v_t, v_p, w_y, w_t, w_p)``: ``D2f[u, v] + Df[w]``, built on first use."""
    import numba
    from llvmlite import ir
    from numba.core import cgutils, types
    from numba.extending import intrinsic

    first = first_order(fn, n_out)
    label = _label(_python_function(fn))

    def link():
        built = first.get()
        text, entry = build_second_order(built)
        started = time.perf_counter()
        library, symbol = _library(text, entry)
        BUILDS.append(
            {
                "what": "link second order",
                "function": label,
                "total_s": time.perf_counter() - started,
            }
        )
        return built, library, symbol

    linked = _Lazy(link)
    f64, i64 = ir.DoubleType(), ir.IntType(64)
    p64 = f64.as_pointer()
    names = (
        "out",
        "y",
        "t",
        "p",
        "u_y",
        "u_t",
        "u_p",
        "v_y",
        "v_t",
        "v_p",
        "w_y",
        "w_t",
        "w_p",
    )

    @intrinsic
    def sweep(typingctx, out, y, t, p, uy, ut, up, vy, vt, vp, wy, wt, wp):
        arguments = (out, y, t, p, uy, ut, up, vy, vt, vp, wy, wt, wp)
        for typ, what in zip(arguments, names):
            if what in ("t", "u_t", "v_t", "w_t"):
                _check_scalar(typ, what, label)
            else:
                _check_array(typ, what, label)
        built, library, symbol = linked.get()
        out_ptr = p64 if built.scalar else ir.ArrayType(f64, built.n_out).as_pointer()
        function_type = ir.FunctionType(
            ir.VoidType(), [p64] * 4 + [f64] * 4 + [p64] * 4 + [i64, i64, out_ptr]
        )

        def codegen(context, builder, signature, args):
            kinds = signature.args

            def array(i):
                return context.make_array(kinds[i])(context, builder, args[i])

            def value(i):
                return context.cast(builder, args[i], kinds[i], types.float64)

            callee = cgutils.get_or_insert_function(
                builder.module, function_type, symbol
            )
            builder.call(
                callee,
                [array(1).data, array(7).data, array(4).data, array(10).data,
                 value(2), value(8), value(5), value(11),
                 array(3).data, array(9).data, array(6).data, array(12).data,
                 array(1).nitems, array(3).nitems,
                 builder.bitcast(array(0).data, out_ptr)],
            )  # fmt: skip
            context.active_code_library.add_linking_library(library)
            return context.get_dummy_value()

        return types.void(*arguments), codegen

    @numba.njit(inline="always")
    def second(out, y, t, p, u_y, u_t, u_p, v_y, v_t, v_p, w_y, w_t, w_p):
        sweep(out, y, t, p, u_y, u_t, u_p, v_y, v_t, v_p, w_y, w_t, w_p)

    return second
