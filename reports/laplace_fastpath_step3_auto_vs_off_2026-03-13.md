# Step 3 Fast-Path Comparison (`auto` vs `off`)

- Date: 2026-03-13
- Scope: Step 3 enables scalar-inner fast backend for fully eligible models.
- Speedup is reported as `off_time / auto_time` (values > 1 mean `auto` is faster).

| case | n_rows | laplace_auto_time_s | laplace_off_time_s | laplace_speedup | laplace_abs_obj_diff | laplace_map_auto_time_s | laplace_map_off_time_s | laplace_map_speedup | laplace_map_abs_obj_diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gaussian | 120 | 0.049978 | 0.062173 | 1.244 | 0.0 | 0.047697 | 0.067817 | 1.4218 | 0.0 |
| lognormal | 120 | 0.054567 | 0.072125 | 1.3218 | 4.06447e-7 | 0.046586 | 0.058621 | 1.2583 | 0.0 |
| bernoulli | 120 | 0.049338 | 0.05751 | 1.1656 | 3.0414e-8 | 0.046835 | 0.077087 | 1.6459 | 1.18684e-7 |
| poisson | 120 | 0.047417 | 0.065535 | 1.3821 | 5.6458e-8 | 0.058951 | 0.071521 | 1.2132 | 2.9153e-8 |
| ode_offset | 120 | 0.164668 | 0.135519 | 0.823 | 0.0 | 0.177885 | 0.176127 | 0.9901 | 0.0 |
| ode_eta | 120 | 0.34502 | 0.339565 | 0.9842 | 0.0 | 0.576453 | 0.579267 | 1.0049 | 0.0 |
