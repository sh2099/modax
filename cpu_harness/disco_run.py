"""What the ``disco`` launcher runs: the stand-in first, then the script.

Says on stderr which modax, DISCO-EB and DISCO-CMB a run used, so that a log
records it, then behaves as ``python`` does for a script, ``-m module`` or
``-c code``.
"""

from __future__ import annotations

import importlib.util
import os
import runpy
import subprocess
import sys

import discoeb_cpu

discoeb_cpu.install()


def _commit(path: str) -> str:
    """The short commit of the checkout holding ``path``, with ``+`` if it has changes."""
    try:
        head = subprocess.run(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        return head + ("+" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "?"


def _describe() -> str:
    from importlib.metadata import PackageNotFoundError, version

    import modax
    from modax import _target

    parts = []
    modax_root = os.path.dirname(os.path.dirname(modax.__file__))
    try:
        import tomllib

        with open(os.path.join(modax_root, "pyproject.toml"), "rb") as handle:
            modax_version = tomllib.load(handle)["project"]["version"]
    except (OSError, KeyError):
        modax_version = "?"
    parts.append(f"modax {modax_version} ({_commit(modax_root)}, {_target.TARGET})")
    for name in ("discoeb", "discocmb"):
        try:
            module = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            module = None
        if module is None or module.origin is None:
            continue
        try:
            release = version({"discoeb": "disco-eb", "discocmb": "disco-cmb"}[name])
        except PackageNotFoundError:
            release = "source"
        parts.append(f"{name} {release} ({_commit(os.path.dirname(module.origin))})")
    return "disco: " + ", ".join(parts)


def main() -> None:
    arguments = sys.argv[1:]
    if not os.environ.get("DISCO_QUIET"):
        print(_describe(), file=sys.stderr, flush=True)
    if not arguments:
        raise SystemExit("disco: give a script, -m module or -c code")
    if arguments[0] == "-m":
        sys.argv = arguments[1:]
        sys.path.insert(0, os.getcwd())
        runpy.run_module(arguments[1], run_name="__main__", alter_sys=True)
    elif arguments[0] == "-c":
        sys.argv = ["-c", *arguments[2:]]
        sys.path.insert(0, os.getcwd())
        exec(compile(arguments[1], "<disco -c>", "exec"), {"__name__": "__main__"})
    else:
        sys.argv = arguments
        sys.path.insert(0, os.path.dirname(os.path.abspath(arguments[0])))
        runpy.run_path(arguments[0], run_name="__main__")


if __name__ == "__main__":
    main()
