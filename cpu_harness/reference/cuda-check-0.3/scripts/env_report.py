"""The distributions this interpreter sees (name, version, where), then the modax it imports.
Usage: env_report.py"""
import importlib.metadata as md, os, sys

print("python", sys.version.split()[0], sys.executable)
print("sys.path[:4]", sys.path[:4])
rows = sorted({(d.metadata["Name"].lower(), d.version) for d in md.distributions()})
for name, ver in rows:
    print(f"{name:40s} {ver}")
import modax  # noqa: E402

print("imported modax:", modax.__file__)
print("modax checkout HEAD:", open(os.path.join(os.environ["MC_SRC"], ".git", "HEAD")).read().strip())
