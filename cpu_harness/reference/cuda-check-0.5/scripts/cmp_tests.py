"""Per-test comparison of two pytest -rA logs. Usage: cmp_tests.py LOG_A LOG_B"""

import re, sys
from collections import Counter


def outcomes(path):
    res = {}
    pat = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)")
    verbose = re.compile(
        r"^(tests/\S+::\S.*?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b"
    )
    for line in open(path, errors="replace"):
        m = verbose.match(line)
        if m:
            res[m.group(1)] = m.group(2)
    return res


a, b = outcomes(sys.argv[1]), outcomes(sys.argv[2])
print("counts a:", dict(Counter(a.values())), "total", len(a))
print("counts b:", dict(Counter(b.values())), "total", len(b))
keys = sorted(set(a) | set(b))
diff = [
    (k, a.get(k, "ABSENT"), b.get(k, "ABSENT")) for k in keys if a.get(k) != b.get(k)
]
print(f"\ntests with different outcome: {len(diff)}")
for k, x, y in diff:
    print(f"  {x:8s} -> {y:8s}  {k}")
print("\nnot passing, same outcome in both:")
for k in keys:
    if a.get(k) == b.get(k) and a[k] != "PASSED":
        print(f"  {a[k]:8s} {k}")
