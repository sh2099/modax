import numpy as np, sys
d=np.load(sys.argv[1])
for k in d.files:
    if k.endswith("accepted_steps"): print(k, int(d[k].min()), int(d[k].max()))
print(d["rob_rodas5p_fp64/0"][0,-1], d["rob_jvp_params_fp64/tangent"][0,-1], d["lorenz_tsit5_jvp_params/tangent"][0,-1])
print("local==shared", d["lorenz_tsit5_local/0"].tobytes()==d["lorenz_tsit5_shared/0"].tobytes())
