#!/usr/bin/env bash
# Build what modax's CPU target needs for Enzyme derivatives on an Apple
# Silicon Mac, in one directory outside any checkout.
#
#   build_enzyme_toolchain.sh [PREFIX]
#
# PREFIX defaults to $ENZYME_TOOLCHAIN, else ../../enzyme-toolchain beside the
# modax checkouts (~/Documents/PhD/code/enzyme-toolchain). It ends up holding:
#
#   PREFIX/llvm15/bin/{opt,llvm-link,llvm-extract}   LLVM 15.0.7's own release binaries
#   PREFIX/enzyme/LLVMEnzyme-15.dylib                Enzyme v0.0.289 built against them
#   PREFIX/venv/                                     Python 3.12, numba 0.61.2, llvmlite 0.44.0,
#                                                    numba-enzyme-cuda 0.3.1 and what DISCO-EB
#                                                    and DISCO-CMB need
#   PREFIX/env.sh                                    MODAX_ENZYME_TOOLS for a shell
#
# MODAX_ENZYME_TOOLS=PREFIX is what modax reads (modax/_enzyme_cpu.py); the
# disco launcher sets it when it picks PREFIX/venv.
#
# Why each piece:
# - numba-enzyme-cuda has Linux wheels only, which vendor Linux LLVM 15 tools
#   and the Enzyme plugin. On macOS it is built from its sdist with
#   NUMBA_ENZYME_VENDOR_FROM_PYPI=0, which makes a pure-Python wheel (its
#   default build hook would download the Linux binaries), and the tools come
#   from here instead.
# - numba-enzyme lowers through numba 0.61's typed-pointer IR, which the LLVM
#   15 tools read; numba 0.62 and later emit opaque pointers and fail. So the
#   environment pins numba 0.61.2 / llvmlite 0.44.0 (and numpy 2.2, the newest
#   numba 0.61 supports). The environment beside it on numba 0.68 keeps
#   working with finite differences.
# - Enzyme v0.0.289 is the tag numba-enzyme-cuda 0.3.1 pins for its own plugin.
#   It builds against LLVM 15.0.7's release tree (headers and CMake files) in
#   well under a minute; only the three tools and the plugin are kept.
#
# Steps are skipped when their output is already there, so a rerun finishes
# an interrupted build. Set ENZYME_TOOLCHAIN_FROM=DIR to copy already-built
# tools and plugin from DIR/llvm15/bin and DIR/enzyme (or DIR/llvm15-min/bin
# and DIR/enzyme-plugin) instead of downloading and building them.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
code="$(cd "$here/../.." && pwd -P)"
prefix="${1:-${ENZYME_TOOLCHAIN:-$code/enzyme-toolchain}}"
mkdir -p "$prefix"
prefix="$(cd "$prefix" && pwd -P)"
build="$prefix/build"

LLVM_VERSION=15.0.7
LLVM_TARBALL="clang+llvm-$LLVM_VERSION-arm64-apple-darwin22.0.tar.xz"
LLVM_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/$LLVM_TARBALL"
ENZYME_VERSION=0.0.289
ENZYME_URL="https://github.com/EnzymeAD/Enzyme/archive/refs/tags/v$ENZYME_VERSION.tar.gz"
PYTHON=3.12

[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
    echo "build_enzyme_toolchain: this is for Apple Silicon macOS; on Linux the numba-enzyme-cuda wheel brings its own tools" >&2
    exit 2
}
for tool in curl tar cmake make uv xcrun; do
    command -v "$tool" >/dev/null || { echo "build_enzyme_toolchain: needs $tool" >&2; exit 2; }
done
export SDKROOT="${SDKROOT:-$(xcrun --show-sdk-path)}"
say() { echo "[toolchain] $*" >&2; }

# --- LLVM 15 tools and the Enzyme plugin ----------------------------------------------
tools=(opt llvm-link llvm-extract)
plugin="$prefix/enzyme/LLVMEnzyme-15.dylib"
mkdir -p "$prefix/llvm15/bin" "$prefix/enzyme"
have_tools=1
for t in "${tools[@]}"; do [[ -x "$prefix/llvm15/bin/$t" ]] || have_tools=0; done

if [[ -n "${ENZYME_TOOLCHAIN_FROM:-}" && ( $have_tools == 0 || ! -f "$plugin" ) ]]; then
    from="$ENZYME_TOOLCHAIN_FROM"
    bin="$from/llvm15/bin"; [[ -d "$bin" ]] || bin="$from/llvm15-min/bin"
    lib="$from/enzyme/LLVMEnzyme-15.dylib"; [[ -f "$lib" ]] || lib="$from/enzyme-plugin/LLVMEnzyme-15.dylib"
    say "copying tools from $bin and the plugin from $lib"
    for t in "${tools[@]}"; do cp -p "$bin/$t" "$prefix/llvm15/bin/$t"; done
    cp -p "$lib" "$plugin"
    have_tools=1
fi

if [[ $have_tools == 0 || ! -f "$plugin" ]]; then
    mkdir -p "$build"
    llvm="$build/${LLVM_TARBALL%.tar.xz}"
    if [[ ! -d "$llvm/lib/cmake/llvm" ]]; then
        say "downloading $LLVM_URL (about 650 MB)"
        curl -fsSL --retry 3 -o "$build/$LLVM_TARBALL" "$LLVM_URL"
        say "sha256 $(shasum -a 256 "$build/$LLVM_TARBALL" | cut -d' ' -f1)"
        tar -xJf "$build/$LLVM_TARBALL" -C "$build"
        rm -f "$build/$LLVM_TARBALL"
    fi
    for t in "${tools[@]}"; do cp -p "$llvm/bin/$t" "$prefix/llvm15/bin/$t"; done
    if [[ ! -f "$plugin" ]]; then
        src="$build/Enzyme-$ENZYME_VERSION"
        if [[ ! -d "$src/enzyme" ]]; then
            say "downloading $ENZYME_URL"
            curl -fsSL --retry 3 -o "$build/enzyme.tar.gz" "$ENZYME_URL"
            tar -xzf "$build/enzyme.tar.gz" -C "$build"
            rm -f "$build/enzyme.tar.gz"
        fi
        say "building Enzyme v$ENZYME_VERSION against LLVM $LLVM_VERSION"
        cmake -S "$src/enzyme" -B "$build/enzyme-build" \
            -DLLVM_DIR="$llvm/lib/cmake/llvm" -DCMAKE_BUILD_TYPE=Release >"$build/enzyme-cmake.log"
        cmake --build "$build/enzyme-build" --target LLVMEnzyme-15 -j "$(sysctl -n hw.ncpu)" \
            >"$build/enzyme-build.log"
        cp -p "$build/enzyme-build/Enzyme/LLVMEnzyme-15.dylib" "$plugin"
    fi
    # the release tree is 4.5 GB and nothing reads it once the plugin is built
    [[ -n "${KEEP_BUILD:-}" ]] || rm -rf "$build"
fi
for t in "${tools[@]}"; do "$prefix/llvm15/bin/$t" --version >/dev/null; done
say "$("$prefix/llvm15/bin/opt" --version | grep -i 'LLVM version')"

# --- the Python environment ---------------------------------------------------------------
venv="$prefix/venv"
[[ -x "$venv/bin/python" ]] || uv venv -q --python "$PYTHON" "$venv"
# The numba 0.61 pin and what it holds back; the rest mirrors the numba 0.68
# environment (cpu_harness/.venv-discoeb-lock in the modax 0.4 checkout).
uv pip install -q --python "$venv/bin/python" \
    "numba==0.61.2" "llvmlite==0.44.0" "numpy==2.2.6" \
    "jax==0.10.2" "jaxlib==0.10.2" "ml-dtypes==0.5.4" "opt-einsum==3.4.0" \
    "scipy==1.18.0" "networkx==3.6.1" "cvxopt==1.3.3" \
    "classy==3.3.4.0" "matplotlib==3.11.0" \
    "pytest==9.1.1" "pytest-benchmark==5.2.3" "pytest-cov==7.1.0" \
    "pytest-regressions==2.11.0" "pytest-datadir==1.8.0" "coverage==7.17.2" \
    "mpmath" "ipython"
NUMBA_ENZYME_VENDOR_FROM_PYPI=0 uv pip install -q --python "$venv/bin/python" \
    --no-binary numba-enzyme-cuda "numba-enzyme-cuda==0.3.1"

cat >"$prefix/env.sh" <<EOF
# source this: Enzyme on modax's CPU target from this toolchain
export MODAX_ENZYME_TOOLS="$prefix"
EOF

MODAX_ENZYME_TOOLS="$prefix" "$venv/bin/python" - <<'EOF'
import numba, llvmlite, numba_enzyme
print(f"[toolchain] numba {numba.__version__}, llvmlite {llvmlite.__version__}, numba-enzyme-cuda {numba_enzyme.__version__}")
EOF
say "done: $prefix (MODAX_ENZYME_TOOLS=$prefix, python $venv/bin/python)"
