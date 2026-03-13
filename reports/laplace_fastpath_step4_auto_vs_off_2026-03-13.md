# Step 4 Fast-Path Comparison (`auto` vs `off`)

- Date: 2026-03-13
- Scope: Step 4 enables Newton-inner fast backend for fully eligible models, including dense (`n_b > 1`) batches.
- Speedup is reported as `off_time / auto_time` (values > 1 mean `auto` is faster).

| case | n_rows | laplace_auto_time_s | laplace_off_time_s | laplace_speedup | laplace_abs_obj_diff | laplace_map_auto_time_s | laplace_map_off_time_s | laplace_map_speedup | laplace_map_abs_obj_diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gaussian | 120 | 0.041558 | 0.062245 | 1.4978 | 0.0 | 0.043742 | 0.058722 | 1.3425 | 0.0 |
| lognormal | 120 | 0.051728 | 0.061522 | 1.1893 | 4.06447e-7 | 0.040675 | 0.054298 | 1.3349 | 0.0 |
| bernoulli | 120 | 0.043938 | 0.049222 | 1.1203 | 3.0414e-8 | 0.041303 | 0.068695 | 1.6632 | 1.18684e-7 |
| poisson | 120 | 0.044874 | 0.055132 | 1.2286 | 5.6458e-8 | 0.055085 | 0.066978 | 1.2159 | 2.9153e-8 |
| gaussian_dense | 120 | 0.086432 | 0.086516 | 1.001 | 2.0e-12 | 0.085323 | 0.090072 | 1.0557 | 3.0e-12 |
| lognormal_dense | 120 | 0.090171 | 0.089998 | 0.9981 | 1.8e-11 | 0.089945 | 0.087668 | 0.9747 | 9.0e-12 |
| bernoulli_dense | 120 | 0.072413 | 0.087145 | 1.2034 | 2.994e-9 | 0.067025 | 0.09757 | 1.4557 | 1.8484e-8 |
| poisson_dense | 120 | 0.073684 | 0.078918 | 1.071 | 2.168e-9 | 0.063045 | 0.070494 | 1.1182 | 3.13e-10 |
| ode_offset | 120 | 0.129716 | 0.123105 | 0.949 | 0.0 | 0.165595 | 0.158488 | 0.9571 | 0.0 |
| ode_eta | 120 | 0.260627 | 0.253311 | 0.9719 | 0.0 | 0.277661 | 0.275032 | 0.9905 | 0.0 |
