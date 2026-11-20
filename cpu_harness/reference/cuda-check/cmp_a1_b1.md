## a run1 (v0.4.0) vs b run1 (cpu-target-0.4): 126 vs 126 arrays
| array | dtype | shape | bitwise | max abs diff | max rel diff | n differing | finite |
|---|---|---|---|---|---|---|---|
| chain_dense_fp64/0 | float64 | (64, 3, 4) | identical | 0 | 0 | 0 | True |
| chain_dense_fp64/1/accepted_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| chain_dense_fp64/1/loop_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| chain_dense_fp64/1/rejected_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| chain_sparse_fp32/0 | float64 | (64, 3, 4) | identical | 0 | 0 | 0 | True |
| chain_sparse_fp32/1/accepted_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| chain_sparse_fp32/1/loop_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| chain_sparse_fp32/1/rejected_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| chain_sparse_fp64/0 | float64 | (64, 3, 4) | identical | 0 | 0 | 0 | True |
| chain_sparse_fp64/1/accepted_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| chain_sparse_fp64/1/loop_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| chain_sparse_fp64/1/rejected_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| chain_sparse_grad | float64 | (3,) | identical | 0 | 0 | 0 | True |
| hier2_fp64/0 | float64 | (2, 7, 2) | identical | 0 | 0 | 0 | True |
| hier2_fp64/1/accepted_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier2_fp64/1/loop_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier2_fp64/1/rejected_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_auto/0 | float64 | (2, 7, 3) | identical | 0 | 0 | 0 | True |
| hier3_auto/1/accepted_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_auto/1/loop_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_auto/1/rejected_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_cache/0 | float64 | (2, 7, 3) | identical | 0 | 0 | 0 | True |
| hier3_cache/1/accepted_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_cache/1/loop_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_cache/1/rejected_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_default_method/0 | float64 | (2, 7, 3) | identical | 0 | 0 | 0 | True |
| hier3_default_method/1/accepted_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_default_method/1/loop_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_default_method/1/rejected_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_fp64/0 | float64 | (2, 7, 3) | identical | 0 | 0 | 0 | True |
| hier3_fp64/1/accepted_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_fp64/1/loop_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_fp64/1/rejected_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_hook/0 | float64 | (2, 7, 3) | identical | 0 | 0 | 0 | True |
| hier3_hook/1 | float64 | (2, 7, 2) | identical | 0 | 0 | 0 | True |
| hier3_hook/2/accepted_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_hook/2/loop_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_hook/2/rejected_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_jvp_p0/primal | float64 | (2, 7, 3) | identical | 0 | 0 | 0 | True |
| hier3_jvp_p0/stats/accepted_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_jvp_p0/stats/loop_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_jvp_p0/stats/rejected_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_jvp_p0/tangent | float64 | (2, 7, 3) | identical | 0 | 0 | 0 | False |
| hier3_jvp_p1/primal | float64 | (2, 7, 3) | identical | 0 | 0 | 0 | True |
| hier3_jvp_p1/stats/accepted_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_jvp_p1/stats/loop_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_jvp_p1/stats/rejected_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_jvp_p1/tangent | float64 | (2, 7, 3) | identical | 0 | 0 | 0 | False |
| hier3_jvp_p2/primal | float64 | (2, 7, 3) | identical | 0 | 0 | 0 | True |
| hier3_jvp_p2/stats/accepted_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_jvp_p2/stats/loop_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_jvp_p2/stats/rejected_steps | int32 | (2,) | identical | 0 | 0 | 0 | True |
| hier3_jvp_p2/tangent | float64 | (2, 7, 3) | identical | 0 | 0 | 0 | False |
| hier3_jvp_y0/0 | float64 | (2, 7, 3) | identical | 0 | 0 | 0 | True |
| hier3_jvp_y0/1 | float64 | (2, 7, 3) | identical | 0 | 0 | 0 | False |
| hook_jvp_rodas5p/0/0 | float64 | (3, 33, 2) | identical | 0 | 0 | 0 | True |
| hook_jvp_rodas5p/0/1 | float64 | (3, 33, 2) | identical | 0 | 0 | 0 | True |
| hook_jvp_rodas5p/1/0 | float64 | (3, 33, 2) | identical | 0 | 0 | 0 | True |
| hook_jvp_rodas5p/1/1 | float64 | (3, 33, 2) | identical | 0 | 0 | 0 | True |
| hook_jvp_tsit5/0/0 | float64 | (3, 33, 2) | identical | 0 | 0 | 0 | True |
| hook_jvp_tsit5/0/1 | float64 | (3, 33, 2) | identical | 0 | 0 | 0 | True |
| hook_jvp_tsit5/1/0 | float64 | (3, 33, 2) | identical | 0 | 0 | 0 | True |
| hook_jvp_tsit5/1/1 | float64 | (3, 33, 2) | identical | 0 | 0 | 0 | True |
| hook_nohistory_rodas5p/0 | float64 | (3, 1, 2) | identical | 0 | 0 | 0 | True |
| hook_nohistory_rodas5p/1 | float64 | (3, 33, 2) | identical | 0 | 0 | 0 | True |
| hook_nohistory_tsit5/0 | float64 | (3, 1, 2) | identical | 0 | 0 | 0 | True |
| hook_nohistory_tsit5/1 | float64 | (3, 33, 2) | identical | 0 | 0 | 0 | True |
| hook_rodas5p/0 | float64 | (3, 33, 2) | identical | 0 | 0 | 0 | True |
| hook_rodas5p/1 | float64 | (3, 33, 2) | identical | 0 | 0 | 0 | True |
| hook_rodas5p/2/accepted_steps | int32 | (3,) | identical | 0 | 0 | 0 | True |
| hook_rodas5p/2/loop_steps | int32 | (3,) | identical | 0 | 0 | 0 | True |
| hook_rodas5p/2/rejected_steps | int32 | (3,) | identical | 0 | 0 | 0 | True |
| hook_tsit5/0 | float64 | (3, 33, 2) | identical | 0 | 0 | 0 | True |
| hook_tsit5/1 | float64 | (3, 33, 2) | identical | 0 | 0 | 0 | True |
| hook_tsit5/2/accepted_steps | int32 | (3,) | identical | 0 | 0 | 0 | True |
| hook_tsit5/2/loop_steps | int32 | (3,) | identical | 0 | 0 | 0 | True |
| hook_tsit5/2/rejected_steps | int32 | (3,) | identical | 0 | 0 | 0 | True |
| lorenz_auto/0 | float64 | (64, 11, 3) | identical | 0 | 0 | 0 | True |
| lorenz_auto/1/accepted_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| lorenz_auto/1/loop_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| lorenz_auto/1/rejected_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_auto/0 | float64 | (64, 11, 3) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_auto/1/accepted_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_auto/1/loop_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_auto/1/rejected_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_jvp_params/primal | float64 | (8, 5, 3) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_jvp_params/stats/accepted_steps | int32 | (8,) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_jvp_params/stats/loop_steps | int32 | (8,) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_jvp_params/stats/rejected_steps | int32 | (8,) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_jvp_params/tangent | float64 | (8, 5, 3) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_local/0 | float64 | (64, 11, 3) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_local/1/accepted_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_local/1/loop_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_local/1/rejected_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_local_cache/0 | float64 | (64, 11, 3) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_local_cache/1/accepted_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_local_cache/1/loop_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_local_cache/1/rejected_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_shared/0 | float64 | (64, 11, 3) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_shared/1/accepted_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_shared/1/loop_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| lorenz_tsit5_shared/1/rejected_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| rob_auto/0 | float64 | (64, 5, 3) | identical | 0 | 0 | 0 | True |
| rob_auto/1/accepted_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| rob_auto/1/loop_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| rob_auto/1/rejected_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| rob_jit_vmap_fp64 | float64 | (16, 1, 5, 3) | identical | 0 | 0 | 0 | True |
| rob_jvp_params_fp64/primal | float64 | (8, 5, 3) | identical | 0 | 0 | 0 | True |
| rob_jvp_params_fp64/stats/accepted_steps | int32 | (8,) | identical | 0 | 0 | 0 | True |
| rob_jvp_params_fp64/stats/loop_steps | int32 | (8,) | identical | 0 | 0 | 0 | True |
| rob_jvp_params_fp64/stats/rejected_steps | int32 | (8,) | identical | 0 | 0 | 0 | True |
| rob_jvp_params_fp64/tangent | float64 | (8, 5, 3) | identical | 0 | 0 | 0 | True |
| rob_jvp_y0_fp32/0 | float64 | (8, 5, 3) | identical | 0 | 0 | 0 | True |
| rob_jvp_y0_fp32/1 | float64 | (8, 5, 3) | identical | 0 | 0 | 0 | True |
| rob_rodas5p_default_fp32/0 | float64 | (64, 5, 3) | identical | 0 | 0 | 0 | True |
| rob_rodas5p_default_fp32/1/accepted_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| rob_rodas5p_default_fp32/1/loop_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| rob_rodas5p_default_fp32/1/rejected_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| rob_rodas5p_fp64/0 | float64 | (64, 5, 3) | identical | 0 | 0 | 0 | True |
| rob_rodas5p_fp64/1/accepted_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| rob_rodas5p_fp64/1/loop_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| rob_rodas5p_fp64/1/rejected_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| rob_rodas5p_fp64_cache/0 | float64 | (64, 5, 3) | identical | 0 | 0 | 0 | True |
| rob_rodas5p_fp64_cache/1/accepted_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| rob_rodas5p_fp64_cache/1/loop_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |
| rob_rodas5p_fp64_cache/1/rejected_steps | int32 | (64,) | identical | 0 | 0 | 0 | True |

126/126 arrays bitwise identical

