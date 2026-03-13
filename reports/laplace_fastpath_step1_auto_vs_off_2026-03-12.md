# Step 1 Fast-Path Comparison (`auto` vs `off`)

- Date: 2026-03-12
- Scope: Step 1 adds configuration + logging only (no numerical fast-path backend).

| case | n_rows | laplace_abs_obj_diff | laplace_map_abs_obj_diff |
|---|---:|---:|---:|
| gaussian | 120 | 0.0 | 0.0 |
| lognormal | 120 | 0.0 | 0.0 |
| bernoulli | 120 | 0.0 | 0.0 |
| poisson | 120 | 0.0 | 0.0 |
| ode_offset | 120 | 0.0 | 0.0 |
| ode_eta | 120 | 0.0 | 0.0 |
