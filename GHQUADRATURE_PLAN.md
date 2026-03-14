# Plan: Sparse Deterministic Integration Backend for NLME (GHQuadrature)

> **Progress legend:** ✅ Done | 🔲 Not started
> **Steps completed:** 3 / 3

## Context

NoLimits.jl currently offers Laplace, FOCEI, MCEM, and SAEM for marginal-likelihood-based RE estimation. All of these either use a point approximation (Laplace/FOCEI) or Monte Carlo sampling (MCEM/SAEM). A deterministic alternative is sparse-grid (Smolyak) quadrature, which yields exact integration in the limit as the grid level increases, is ForwardDiff-compatible by construction (nodes are fixed Float64 constants), and converges faster than MC for smooth integrands. This plan implements `GHQuadrature` as a new `FittingMethod` plugging into the existing `fit_model` API.

**User decisions from planning:**
- Phase 1: Gauss-Hermite only
- Always use sparse grid (no automatic fallback)
- Signed logsumexp (Smolyak has negative weights)
- New `GHQuadrature <: FittingMethod`

---

## Mathematical Summary

For a batch of individuals sharing RE levels, the marginal likelihood is:

```
log L(θ) = Σ_batches log ∫ p(y_batch | b, θ) p(b | θ) db
```

For a Gaussian prior `p(b) = N(μ, LL^T)`, the change of variables `b = μ + Lz`, `z ~ N(0,I)` transforms the integral into:

```
log ∫ p(y | μ + Lz, θ) N(z; 0, I) dz
```

which is approximated by Smolyak sparse-grid quadrature with GH nodes `{(z_r, W_r)}`:

```
log L_batch ≈ signed_logsumexp_r [ log|W_r| + Σ_i ℓ_i(μ + Lz_r, θ) ]
```

---

## Architecture (4 new files + 2 modifications)

### New Files

```
src/estimation/nodes.jl       # GH rule, Smolyak construction, global cache
src/estimation/remeasure.jl   # AbstractREMeasure, GaussianRE, build_gaussian_re_from_batch
src/estimation/kernel.jl      # signed_logsumexp, batch_loglik_ghq
src/estimation/ghquadrature.jl  # GHQuadrature, GHQuadratureResult, _fit_model dispatch
```

### Modified Files

```
src/NoLimits.jl               # Add 4 include statements after laplace.jl
src/estimation/common.jl      # Add GHQuadratureResult dispatch to get_random_effects, get_loglikelihood
```

No new package dependencies — uses `LinearAlgebra` (already present) for Golub-Welsch eigenvalue decomposition.

---

## Phase 1 Implementation (this plan)

### 1. `nodes.jl` — Gauss-Hermite + Smolyak

**Types:**

```julia
struct GHQuadratureNodes{T<:AbstractFloat}
    dim::Int
    level::Int
    nodes::Matrix{T}        # (d, R): each column is one quadrature point
    logweights::Vector{T}   # (R,): log|Smolyak weight|
    signs::Vector{Int8}     # (R,): net sign (+1 or -1) per node
end
```

**1D GH via Golub-Welsch (no new dependency):**

```
J = SymTridiagonal(zeros(n), [sqrt(i/2) for i in 1:n-1])
vals, vecs = eigen(J)
w_phys = sqrt(π) .* vecs[1,:].^2
# Probabilist convention: nodes = sqrt(2) .* vals, weights sum to 1
logweights = log.(w_phys ./ sqrt(π))
```

For `n=1`: node at 0, single weight 1.0.

**Smolyak combination:**

Multi-indices `α = (α₁,...,αd)` with each `αᵢ ≥ 1` and `Σαᵢ ≤ d + L - 1`:

```
coefficient c(q) = (-1)^(d+L-1-q) * binomial(d-1, d+L-1-q)
```

For each multi-index: tensor-product the 1D GH rules of orders `α₁,...,αd`. Each tensor-product point gets `logweight = Σ log|w_kᵢ| + log|c|`, sign = `sign(c) * Πsign(w_kᵢ)`. Collect all into `GHQuadratureNodes`. No deduplication in Phase 1.

**Global cache** (populated before parallel use):

```julia
const _SPARSEGRID_CACHE = Dict{Tuple{Int,Int}, GHQuadratureNodes{Float64}}()
get_sparse_grid(dim::Int, level::Int) -> GHQuadratureNodes{Float64}  # builds and caches
n_ghq_points(dim::Int, level::Int) -> Int                   # exported utility
```

---

### 2. `remeasure.jl` — RE Measure Abstraction

```julia
abstract type AbstractREMeasure end
# Interface: transform(re, z) -> η, logcorrection(re, z) -> scalar
```

**Phase 1: `GaussianRE`**

```julia
struct GaussianRE{T<:AbstractFloat, MT<:AbstractMatrix} <: AbstractREMeasure
    μ::Vector{T}   # concatenated prior means, length n_b
    L::MT          # block-diagonal lower-Cholesky, n_b × n_b
    n_b::Int
end
transform(re::GaussianRE, z) = re.μ + re.L * z
logcorrection(::GaussianRE, z) = zero(eltype(z))
```

Construction `build_gaussian_re_from_batch(batch_info, θ, const_cache, dm, ll_cache)`:
- Extracts Normal/MvNormal distribution parameters for each RE level via `dists_builder`
- For `Normal(μ, σ)`: μ_k = [mean], L_k = [[σ]]
- For `MvNormal`: μ_k = mean(d), L_k = `cholesky(Symmetric(d.Σ.mat + ε*I)).L`
- Assembles block-diagonal L_full
- Non-Normal/MvNormal: throws informative error mentioning "Phase 1"

**Phase 2 stubs** (no methods): `LogNormalRE`, `BoundedRE`, `TransportRE`

---

### 3. `kernel.jl` — Signed LogSumExp + Batch Kernel

**Signed logsumexp** returns `(log|result|, sign::Int8)`:

```julia
function signed_logsumexp(logvals::AbstractVector{T}, signs::AbstractVector{Int8}) where T
    amax = maximum(logvals)
    pos = neg = zero(T)
    for (lv, s) in zip(logvals, signs)
        e = exp(lv - amax)
        s > 0 ? (pos += e) : (neg += e)
    end
    diff = pos - neg
    diff > 0 && return (amax + log(diff), Int8(1))
    diff < 0 && return (amax + log(-diff), Int8(-1))
    return (T(-Inf), Int8(1))
end
```

**Batch kernel** `batch_loglik_ghq(dm, batch_info, θ, re_measure, sgrid, const_cache, ll_cache)`:
- For each node r: `b_r = transform(re, z_r)`, slice η per individual via `_build_eta_ind`, sum `_loglikelihood_individual`, accumulate `a[r] = logw[r] + cond + logcorrection`
- Returns `signed_logsumexp(a, s)[1]` (negative result → `@warn` + return `-Inf`)
- No prior logpdf term for `GaussianRE` (absorbed by change of variables)

---

### 4. `ghquadrature.jl` — FittingMethod + Outer Loop

**Struct** (mirrors Laplace kwargs):
```julia
struct GHQuadrature{...} <: FittingMethod
    level::Int; optimizer; optim_kwargs; adtype
    inner::LaplaceInnerOptions; multistart::LaplaceMultistartOptions
    lb; ub; maxiters::Int; ignore_model_bounds::Bool
end
```

**Outer loop:**
1. Validate RE distributions (Normal/MvNormal only)
2. `_build_laplace_batch_infos` (reuse)
3. `build_ll_cache` (reuse)
4. Pre-populate sparse grid cache for all unique `n_b`
5. Objective: `θ → -Σ_batch batch_loglik_ghq(...)` differentiated via `AutoForwardDiff`
6. Optimize → `GHQuadratureResult` with `eb_modes` for `get_random_effects`

Key difference from Laplace: **fully AD-differentiated** objective (no envelope theorem).

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Signed logsumexp cancellation at high levels | High | Warn when 6+ digits lost. Safe range: level 1–3. |
| Non-Gaussian RE (Beta, LogNormal, NPF) | High | Error before computation in `_ghq_validate_re_distributions`. |
| Dimension explosion (large batches) | Medium | `@warn` when `n_ghq_points(n_b, level) > 10_000`. Export utility. |
| ForwardDiff through `PDMat` Cholesky | High | Extract raw matrix via `d.Σ.mat`, call `cholesky(Symmetric(...))` directly. |
| GH eigenvalue accuracy | Low | Test `sum(exp.(logweights)) ≈ 1.0` within 1e-12 for orders 1–5. |

---

## Reused Infrastructure

| Function / Type | Location | Used for |
|-----------------|----------|----------|
| `_build_laplace_batch_infos` | `laplace.jl:400` | Batch construction |
| `_build_eta_ind` | `laplace.jl:1046` | η slicing from b vector |
| `_laplace_compute_bstar_batch!` | `laplace.jl:1229` | EB modes for `get_random_effects` |
| `_loglikelihood_individual` | `common.jl:896` | Individual likelihood |
| `build_ll_cache` | `common.jl` | ODE/covariate cache |
| `_symmetrize_psd_params` | `common.jl` | PSD param symmetrization |
| `LaplaceInnerOptions`, `LaplaceMultistartOptions` | `laplace.jl:1735,1742` | Optimizer config |
| `_LaplaceBatchInfo`, `_LaplaceREInfo` | `laplace.jl:231,239` | Batch structures |

---

## Phased Roadmap

**Phase 1 (complete):** Smolyak + GH + GaussianRE + signed logsumexp + `GHQuadrature` FittingMethod + `NormalizingPlanarFlow` RE via `CompositeRE`

**Phase 2 (complete):**
- ✅ `GHQuadratureMAP` — MAP-regularised variant of `GHQuadrature` (adds `logprior` to outer objective; same interface)
- ✅ UQ support — Wald, Profile, and mcmc_refit UQ for `GHQuadrature`/`GHQuadratureMAP` results; Wald default Hessian backend `:forwarddiff`; sandwich vcov per-batch supported
- ✅ Plotting support — `GHQuadrature`/`GHQuadratureMAP` results fully compatible with all plotting functions (`plot_fits`, `plot_residuals`, `plot_vpc`, `plot_random_effects_*`, `plot_uq_distributions`, etc.) via `_default_random_effects` dispatch
- ✅ Summary/accessor compatibility — all standard accessors (`get_params`, `get_objective`, `get_random_effects`, `get_loglikelihood`, `get_converged`, `get_iterations`, `get_raw`, `get_notes`) work for both result types

**Phase 3 (complete):**
- ✅ Parallelization — `EnsembleThreads` threads the batch loop; `Threads.@threads` with per-thread `ll_cache`; ForwardDiff-compatible via `Vector{T}` accumulation
- ✅ Node deduplication — merge duplicate Smolyak nodes by summing signed weights; near-zero combined weights discarded; reduces d=2 L=3 from 15 to fewer points; integration accuracy preserved
- ✅ `LogNormalRE` transport map — `η = exp(μ + σz)`, `logcorrection = 0`; handled via `CompositeRE` segment in `build_re_measure_from_batch`
- ✅ `BoundedRE` transport map (Beta) — `η = logistic(z)`, `logcorrection = logpdf(Beta(α,β),η) + log η + log(1-η) + z²/2 + log(2π)/2`; `DomainError` on param underflow → graceful `-Inf` via try-catch in `_ghq_batch_ll`
- ✅ Anisotropic grids — `level::NamedTuple` maps RE group name → level; tensor product of per-group Smolyak grids; `get_anisotropic_grid(dims, levels)` with `_ANISOTROPIC_CACHE`; unlisted RE groups default to level 1
- ✅ Additional 1D rules: `_gl_rule` (Gauss-Legendre on [-1,1]), `_cc_rule` (Clenshaw-Curtis on [-1,1]); `build_tensor_product_grid` for composing grids

**Phase 3 additions (complete):**
- ✅ `Gamma`, `Exponential`, `Weibull` RE — exp transport (`η = exp(z)`), inline ForwardDiff-safe logpdf corrections
- ✅ `TDist(ν)` RE — identity transport (`η = z`), heavy-tailed correction
- ✅ Generic `ContinuousUnivariateDistribution` fallback — transport selected by support: ℝ → identity, (0,∞) → exp, (a,b) → scaled logistic; covers `InverseGamma`, `Logistic`, `Gumbel`, `Truncated`, etc. without per-type code

**Phase 4:** Adaptive sparse grids (center at posterior mode), integration accuracy diagnostics, backend heuristics

---

## Incremental Implementation & Testing

Tests added to `test/estimation_ghq_tests.jl` after each step. User approves before next step.

---

### ✅ Step 1: `nodes.jl` — GH rule + Smolyak construction + cache

Tests:
- `gh_rule(1)`: node=0, logweight=0.0 (weight=1)
- `gh_rule(2)`: nodes≈±1, weights≈0.5 each
- `gh_rule(3)`: nodes≈0,±√3; weights≈2/3,1/6,1/6
- `sum(exp.(gh_rule(n).logweights)) ≈ 1.0` within 1e-12 for n=1..5
- `build_sparse_grid(1, 2)` matches `gh_rule(2)` nodes/weights
- `build_sparse_grid(2, 2)` correct point count
- Integration: `Σ exp(logw_r)*f(node_r)` for `f=1`, `f=z²`, `f=exp(Σz)` vs analytic
- `n_ghq_points(d, L) == size(build_sparse_grid(d,L).nodes, 2)`

---

### ✅ Step 2: `remeasure.jl` + `kernel.jl` — RE measure + signed logsumexp + batch kernel

Tests:
- `signed_logsumexp([0.0,0.0], Int8[1,-1])` → `(-Inf, 1)`
- `signed_logsumexp([log(3),log(1)], Int8[1,-1])` → `(log(2), 1)`
- `signed_logsumexp([log(1),log(3)], Int8[1,-1])` → `(log(2), -1)`
- Large value stability: `signed_logsumexp([1000.0, 999.5], Int8[1,-1])` correct
- `GaussianRE` from `Normal(0, 2)`: μ=[0], L=[[2]], transform([1])≈[2]
- `GaussianRE` from `MvNormal(zeros(2), Σ)`: L matches `cholesky(Σ).L`
- `logcorrection(::GaussianRE, z) == 0`
- Non-Gaussian → error mentioning "Phase 1"
- `batch_loglik_ghq` trivial model: finite, matches expected integral

---

### ✅ Step 3: `ghquadrature.jl` — Full `GHQuadrature` FittingMethod

Tests:
- `fit_model(dm, GHQuadrature(level=1))` on simple `y ~ Normal(a + η, σ)` model, 10 individuals
- All accessors: `get_params`, `get_objective`, `get_converged`, `get_iterations`, `get_random_effects`
- Level comparison: LL(level=2) ≥ LL(level=1) (better approximation)
- Parameter agreement with Laplace within 20%
- ForwardDiff vs FiniteDifferences gradient check (relative error < 1e-4)
- ODE model at `level=1` runs without error
