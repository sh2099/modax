"""Apply the kit's Enzyme cc 6.x patch (ROOT/scripts/cc6x_patch.py, unmodified), then run
SCRIPT [args] as __main__, or pytest when SCRIPT is the word 'pytest'.
The kit's own pascal_run.py is not used because it imports discoeb.
At exit it prints which modax was imported (file and the commit of its checkout), refuses a
modax from anywhere but MC_SRC, and lists any modax CPU-target module that got imported."""
import atexit, os, runpy, sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import cc6x_patch  # noqa: F401,E402


def _report():
    m = sys.modules.get("modax")
    src = os.environ.get("MC_SRC", "")
    if m is None:
        print("[runner] modax was not imported", flush=True)
        return
    head = open(os.path.join(src, ".git", "HEAD")).read().strip() if src else "?"
    ok = os.path.realpath(m.__file__).startswith(os.path.realpath(src) + os.sep)
    print(f"[runner] modax {m.__file__} commit {head} from MC_SRC: {ok}", flush=True)
    cpu = sorted(k for k in sys.modules if k in ("modax._enzyme_cpu", "modax._differences"))
    t = sys.modules.get("modax._target")
    print(f"[runner] modax._target: {'TARGET=' + t.TARGET + ' CPU_DERIVATIVES=' + t.CPU_DERIVATIVES if t else 'absent'}"
          f" | CPU-target modules imported: {cpu or 'none'}", flush=True)
    if not ok:
        print("[runner] ERROR: modax did not come from MC_SRC", flush=True)
        os._exit(3)


atexit.register(_report)
sys.argv = sys.argv[1:]
if not sys.argv:
    sys.exit(__doc__)
if sys.argv[0] == "pytest":
    import pytest

    sys.path.insert(0, os.getcwd())
    sys.exit(pytest.main(sys.argv[1:]))
sys.path.insert(0, os.getcwd())
runpy.run_path(sys.argv[0], run_name="__main__")
