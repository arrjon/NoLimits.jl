# Laplace / LaplaceMAP Speed Analysis for Models Linear in Random Effects

Date: 2026-03-12
Repo: `NoLimits`

## 1. Executive Verdict

Yes, there is substantial room to speed up estimation when the conditional mean (or linear predictor) is linear in random effects.

- For Gaussian mixed models (Normal outcome + Gaussian RE), the current Laplace machinery is doing much more work than necessary and can be replaced by an exact Gaussian-integral path.
- For GLMMs (e.g., Bernoulli-logit, Poisson-log), Laplace is still approximate, but large speedups are possible by exploiting linear-predictor structure in the inner solve and curvature calculations.

## 2. What the Current Laplace Path Is Spending Time On

Code path analyzed: `src/estimation/laplace.jl` (objective, EBE solve, Hessian/logdet, gradient assembly).

Main costs in current implementation:

1. Inner EBE optimization (`_laplace_get_bstar!` / `_laplace_solve_batch!`) via generic `Optimization.jl` LBFGS.
2. Repeated Hessian computations of `logf` wrt random effects (`_laplace_hessian_b`) via `ForwardDiff.hessian!`.
3. Logdet-gradient computation (`trace` or `exact`) that differentiates Hessian-related quantities.
4. Per-evaluation reconstruction/allocation overhead in `_build_eta_ind`, `ComponentArray`/`NamedTuple` assembly, and repeated formula evaluation.

Profile evidence (cached-EBE Gaussian random intercept): hotspots are dominated by `ForwardDiff` Hessian/Jacobian/gradient calls and `NoLimits._laplace_logf_batch` + `_build_eta_ind` + `calculate_formulas_obs`.

## 3. Benchmarks (Synthetic, Non-ODE, Linear-in-RE)

All times are for one objective+gradient evaluation from `_laplace_objective_and_grad`.

- `first`: fresh cache (includes computing `b*`)
- `cached`: same `θ` immediately after (reuses `b*`)

### 3.1 Random intercept (`n_b = 1`, 80 batches)

| Model | trace-grad | hutchinson | exact-nested |
|---|---:|---:|---:|
| Gaussian | first 0.0106s, cached 0.0045s | first 0.0394s, cached 0.0152s | first 2.5361s, cached 0.0189s |
| Bernoulli-logit | first 0.0132s, cached 0.0043s | first 0.0244s, cached 0.0152s | first 2.5090s, cached 0.0191s |
| Poisson-log | first 0.0158s, cached 0.0048s | first 0.0448s, cached 0.0152s | first 2.5346s, cached 0.0185s |

Notes:
- `trace` is clearly best for this low-dimensional case.
- `hutchinson` is consistently slower here.
- `exact-nested` has large first-call cost and is slower than `trace` in cached mode for `n_b=1`.

### 3.2 Crossed random effects example (`η_id + η_site`, mean `n_b ≈ 11`)

Cached per-eval after warmup:

- `trace`: 0.03004s
- `hutchinson`: 0.19826s
- `exact`: 0.03212s

Notes:
- For larger `n_b`, `trace` and `exact` become comparable.
- `hutchinson` is still much slower at this scale.

### 3.3 EBE-only timing (Gaussian random intercept)

`_laplace_get_bstar!`:
- first: 0.005727s
- cached: ~3e-6s

Interpretation: about half of first-eval cost is the EBE solve in this scenario.

### 3.4 Allocation signal (Gaussian random intercept)

Median bytes per evaluation (cached or first-like evaluation) remained high (~7.54 MB/eval), indicating persistent allocation pressure independent of EBE caching.

### 3.5 Closed-form Gaussian reference

For the same Gaussian random-intercept setup, a direct closed-form marginal Gaussian objective:

- objective: 0.000138s/eval
- gradient (via ForwardDiff over closed form): 0.000151s/eval

Compared to current cached Laplace objective+gradient:

- Laplace cached: 0.00512s/eval
- Ratio: ~33.8x slower than closed-form gradient path

This is the strongest quantitative signal that a Gaussian-linear fast path is worth implementing.

### 3.6 Laplace vs LaplaceMAP runtime (Gaussian fit, maxiters=25)

- Laplace: 0.7944s
- LaplaceMAP: 0.7716s
- Ratio (MAP/Laplace): 0.97

Conclusion: for this class, LaplaceMAP prior term is not the runtime bottleneck.

## 4. What This Means for Linear / GLMM Cases

## 4.1 Gaussian LMM case (Normal outcome + Gaussian RE)

When the observation model is Gaussian with random effects entering linearly, the marginalization over random effects is Gaussian and can be done exactly. In that regime, using generic Laplace inner optimization + AD Hessians is unnecessary overhead.

Expected gain from a dedicated path: large (order-of-magnitude; benchmark suggests ~10x to 30x+ depending shape).

## 4.2 GLMM case (Bernoulli/Poisson with linear predictor)

Integral is not closed form, so Laplace remains approximate. But linear-predictor structure gives exploitable algebra:

- score/Hessian wrt `b` has structured form (`Z'(...)`, `Z'WZ + Q`)
- inner solve can use Newton/IRLS (or scalar Newton for `n_b=1`) instead of generic LBFGS
- can reduce repeated AD overhead significantly

Expected gain: moderate-to-large (often 2x-6x for typical small-`n_b` GLMM batches, larger where inner solve dominates).

## 5. Prioritized Optimization Plan

## P0. Add Exact Gaussian-Linear Fast Path (highest impact)

Scope:
- Applies when all outcomes are Normal with affine dependence on random effects and RE priors are Gaussian.
- Works for both Laplace and LaplaceMAP (MAP only adds fixed-effect prior term).

Design:
- Detect eligible model at compile/build stage (or via explicit user opt-in flag initially).
- Build per-batch linear algebra form once (`Z`, offsets, grouping maps).
- Evaluate exact marginal objective/gradient by Cholesky solves on compact systems.

Risk:
- Detection complexity if fully automatic from arbitrary formula AST.
- Mitigation: start with explicit opt-in (`structure=:linear_re_gaussian`) plus strict runtime validation.

## P1. Add Linear-Predictor GLMM Fast Inner Solve

Scope:
- Bernoulli-logit / Poisson-log / Binomial-logit first (canonical links).
- Random effects enter affine in predictor.

Design:
- Replace inner LBFGS with safeguarded Newton/IRLS per batch.
- Provide scalar-specialized path for `n_b=1`.
- Reuse factorization from Newton step where possible in logdet correction.

Risk:
- Numerical stability near separation/extreme counts.
- Mitigation: line search, damping, fallback to current generic path.

## P2. Reduce Per-Eval Allocation and Formula Overhead

Targets:
- `_build_eta_ind` allocation churn (`NamedTuple`/`ComponentArray` creation).
- Repeated distribution construction in `_laplace_logf_batch`.

Design:
- Preallocate reusable RE containers per batch/thread.
- Cache distribution objects keyed by `(θ signature, level/const_cov rep)` where safe.
- Avoid reconstruction in loops when only scalar values change.

Expected gain:
- Useful across all Laplace/LaplaceMAP models, including non-linear ones.

## P3. Make logdet-gradient mode adaptive

Empirical guidance from benchmarks:
- `n_b=1`: `trace` clearly better than `exact`.
- `n_b~11`: `trace` and `exact` similar.
- `hutchinson`: slower in tested dense/small-medium settings.

Recommendation:
- Keep current default (`trace=true`, `hutchinson=false`).
- Add heuristic/option to auto-switch between `trace` and `exact` by `n_b` and maybe `nθ`.

## 6. LaplaceMAP-Specific Notes

- For linear mixed-type models with modest fixed-effect dimension, LaplaceMAP overhead from fixed-effect priors is small relative to Laplace core costs.
- The same structural optimizations (P0/P1/P2) benefit LaplaceMAP almost directly.

## 7. Suggested Implementation Sequence

1. Implement P0 with explicit opt-in + tests vs current Laplace on Gaussian-linear cases.
2. Implement P2 low-risk allocation/cache improvements (shared with all models).
3. Implement P1 GLMM linear-predictor inner solver with fallback.
4. Add P3 adaptive mode tuning once enough benchmark coverage exists.

## 8. Validation Requirements

- Numerical equivalence tests:
  - P0 vs current Laplace (should match objective/gradient and fitted params within tolerance; in Gaussian case should improve stability and speed).
- Robustness tests:
  - extreme variances, near-singular random-effect covariance, sparse groups, crossed effects.
- Performance regression tests:
  - random-intercept (`n_b=1`) and crossed (`n_b>1`) benchmarks with thresholds.

---

## Bottom Line

The current implementation is robust and general, but for models linear in random effects it leaves significant performance on the table.

Most valuable next step: implement an exact Gaussian-linear branch for Laplace/LaplaceMAP, then a structured GLMM linear-predictor inner solver. The measured speed headroom is large enough to justify dedicated implementation.
