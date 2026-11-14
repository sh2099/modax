"""Apply the kit's Enzyme cc 6.x patch (ROOT/scripts/cc6x_patch.py, unmodified), then run
SCRIPT [args] as __main__, or pytest when SCRIPT is the word 'pytest'.
The kit's own pascal_run.py is not used because it imports discoeb."""
import os, runpy, sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import cc6x_patch  # noqa: F401,E402

sys.argv = sys.argv[1:]
if not sys.argv:
    sys.exit(__doc__)
if sys.argv[0] == "pytest":
    import pytest

    sys.path.insert(0, os.getcwd())
    sys.exit(pytest.main(sys.argv[1:]))
sys.path.insert(0, os.getcwd())
runpy.run_path(sys.argv[0], run_name="__main__")
