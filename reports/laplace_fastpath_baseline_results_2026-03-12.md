# Laplace Fast-Path Baseline (Fallback Reference)

- Date: 2026-03-12
- Notes: This is the pre-fast-path fallback baseline. Use for `auto` vs `off` comparisons in subsequent steps.

| case | n_rows | laplace_time_s | laplace_obj | laplace_conv | laplace_map_time_s | laplace_map_obj | laplace_map_conv |
|---|---:|---:|---:|:---:|---:|---:|:---:|
| gaussian | 120 | 0.1166 | 50.593 | false | 0.099 | 53.9106 | false |
| lognormal | 120 | 0.0886 | 16.1127 | true | 0.113 | 19.6249 | false |
| bernoulli | 120 | 0.1976 | 67.8908 | false | 0.0668 | 71.1717 | false |
| poisson | 120 | 0.085 | 148.4639 | true | 0.0658 | 150.824 | false |
| ode_offset | 120 | 0.4658 | 10.674 | false | 0.3101 | 13.2691 | false |
| ode_eta | 120 | 1.9933 | -12.963 | false | 6.4223 | -8.4934 | false |
