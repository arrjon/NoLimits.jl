"""
Benchmark for _laplace_logf_batch allocation profile.
Run before and after the ll=0.0 → zero(T) fix to measure improvement.
"""

using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using ForwardDiff
using Random

import NoLimits: _laplace_logf_batch, _build_laplace_batch_infos, build_ll_cache,
                 _laplace_default_b0

# ── Setup: 20 individuals, 6 obs each, gaussian model ────────────────────────

model = @Model begin
    @covariates begin
        t = Covariate()
        z = Covariate()
    end
    @fixedEffects begin
        a = RealNumber(0.0)
        b = RealNumber(0.4)
        σ = RealNumber(0.5, scale=:log)
        τ = RealNumber(0.7, scale=:log)
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, τ); column=:ID)
    end
    @formulas begin
        y ~ Normal(a + b * z + η, σ)
    end
end

rng = MersenneTwister(123)
n_id, n_obs = 20, 6
n = n_id * n_obs
df = DataFrame(
    ID = repeat(1:n_id, inner=n_obs),
    t  = repeat(collect(0.0:(n_obs-1)), n_id),
    z  = randn(rng, n),
    y  = randn(rng, n)
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)
# _laplace_logf_batch receives θ on the natural (untransformed) scale
θ  = NoLimits.get_θ0_untransformed(model.fixed.fixed)

pairing, batch_infos, const_cache = _build_laplace_batch_infos(dm, NamedTuple())
ll_cache = build_ll_cache(dm)

# Pick the first non-empty batch
bi = findfirst(info -> info.n_b > 0, batch_infos)
info = batch_infos[bi]
b_f64 = _laplace_default_b0(dm, info, θ, const_cache, ll_cache)

# ── Warm up ───────────────────────────────────────────────────────────────────

_laplace_logf_batch(dm, info, θ, b_f64, const_cache, ll_cache)

# Dual b (inner optimization path: ForwardDiff gradient w.r.t. b)
f_b = bv -> _laplace_logf_batch(dm, info, θ, bv, const_cache, ll_cache)
ForwardDiff.gradient(f_b, b_f64)

# Dual θ (outer optimization path: ForwardDiff gradient w.r.t. θ)
θ_vec = Vector(θ)
f_θ = θv -> _laplace_logf_batch(dm, info, ComponentArray(θv, getaxes(θ)), b_f64, const_cache, ll_cache)
ForwardDiff.gradient(f_θ, θ_vec)

# ── Float64 call ──────────────────────────────────────────────────────────────
alloc_f64 = @allocated _laplace_logf_batch(dm, info, θ, b_f64, const_cache, ll_cache)
t_f64 = @elapsed for _ in 1:200; _laplace_logf_batch(dm, info, θ, b_f64, const_cache, ll_cache); end

# ── Dual b (inner gradient) call ──────────────────────────────────────────────
alloc_grad_b = @allocated ForwardDiff.gradient(f_b, b_f64)
t_grad_b = @elapsed for _ in 1:200; ForwardDiff.gradient(f_b, b_f64); end

# ── Dual θ (outer gradient) call ─────────────────────────────────────────────
alloc_grad_θ = @allocated ForwardDiff.gradient(f_θ, θ_vec)
t_grad_θ = @elapsed for _ in 1:200; ForwardDiff.gradient(f_θ, θ_vec); end

import NoLimits: _laplace_grad_batch

# Also benchmark the full grad_batch (includes hessian + logdet trace gradient)
_laplace_grad_batch(dm, info, θ, b_f64, const_cache, ll_cache, nothing, bi)
alloc_grad_full = @allocated _laplace_grad_batch(dm, info, θ, b_f64, const_cache, ll_cache, nothing, bi)
t_grad_full = @elapsed for _ in 1:50; _laplace_grad_batch(dm, info, θ, b_f64, const_cache, ll_cache, nothing, bi); end

println("=== _laplace_logf_batch benchmark (n_id=$n_id, n_obs=$n_obs) ===")
println("Float64 call:           $(alloc_f64) bytes,   $(round(t_f64/200*1e6, digits=1)) μs/call")
println("FD gradient w.r.t. b:   $(alloc_grad_b) bytes,   $(round(t_grad_b/200*1e6, digits=1)) μs/call")
println("FD gradient w.r.t. θ:   $(alloc_grad_θ) bytes,   $(round(t_grad_θ/200*1e6, digits=1)) μs/call")
println("Full grad_batch:        $(alloc_grad_full) bytes,   $(round(t_grad_full/50*1e3, digits=2)) ms/call")
