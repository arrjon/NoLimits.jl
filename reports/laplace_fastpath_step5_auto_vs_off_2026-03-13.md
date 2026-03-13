# Step 5 Fast-Path Comparison (`auto` vs `off`)

- Date: 2026-03-13
- Scope: Step 5 activates fastpath for eligible ODE offset models with a generic polish pass.
- Speedup is reported as `off_time / auto_time` (values > 1 mean `auto` is faster).

| case | n_rows | laplace_auto_time_s | laplace_off_time_s | laplace_speedup | laplace_abs_obj_diff | laplace_map_auto_time_s | laplace_map_off_time_s | laplace_map_speedup | laplace_map_abs_obj_diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gaussian | 120 | 0.048419 | 0.061692 | 1.2741 | 0.0 | 0.04704 | 0.060116 | 1.278 | 0.0 |
| ode_offset | 120 | 0.130226 | 0.131285 | 1.0081 | 0.0 | 0.164467 | 0.177911 | 1.0817 | 0.0 |
| ode_eta | 120 | 0.294301 | 0.274449 | 0.9325 | 0.0 | 0.297665 | 0.299693 | 1.0068 | 0.0 |
