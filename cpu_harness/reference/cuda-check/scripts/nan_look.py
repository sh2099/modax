import numpy as np, sys
d = np.load(sys.argv[1])
for k in ("hier3_jvp_p0/tangent","hier3_jvp_p1/tangent","hier3_jvp_p2/tangent","hier3_jvp_y0/1"):
    a = d[k]; print(k, "nan count", int(np.isnan(a).sum()), "inf", int(np.isinf(a).sum()), "of", a.size, "| nan per state column", np.isnan(a).sum(axis=(0,1)), "| nan per save", np.isnan(a).sum(axis=(0,2)))
print(d["hier3_jvp_p0/tangent"][0])
print("primal finite:", np.isfinite(d["hier3_jvp_p0/primal"]).all(), {k: d[k].tolist() for k in d.files if k.startswith("hier3_jvp_p0/stats")})
