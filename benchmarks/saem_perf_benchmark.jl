"""
SAEM performance benchmark: measures allocations and runtime before/after optimizations.
Run with: julia --project benchmarks/saem_perf_benchmark.jl
"""

using NoLimits
using DataFrames
using Distributions
using OrdinaryDiffEq
using SciMLBase
using Random
using Turing

# ── Build a representative model + dataset ──────────────────────────────────
# 20 individuals, simple linear RE model with Normal outcome
# builtin_stats=:closed_form to exercise the common workflow path

model = @Model begin
    @fixedEffects begin
        a  = RealNumber(1.0)
        b  = RealNumber(0.3)
        σ  = RealNumber(0.5, scale=:log)
        τ  = RealNumber(0.5, scale=:log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, τ); column=:ID)
    end
    @formulas begin
        y ~ Normal(a + b * t + η, σ)
    end
end

rng = Xoshiro(42)
n_ids = 20
n_obs = 8
df = DataFrame(
    ID  = repeat(1:n_ids, inner=n_obs),
    t   = repeat(range(0.0, 4.0; length=n_obs), outer=n_ids),
    y   = randn(rng, n_ids * n_obs),
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t,
               serialization=EnsembleThreads())

# ── SAEM configuration ────────────────────────────────────────────────────────
saem_cfg = SAEM(;
    sampler       = MH(),
    turing_kwargs = (n_samples=5, n_adapt=0, progress=false, verbose=false),
    mcmc_steps    = 5,
    max_store     = 20,
    maxiters      = 40,
    t0            = 5,
    kappa         = 0.65,
    progress      = false,
    builtin_stats = :closed_form,
    re_cov_params = (; η=:τ),
)

# ── Warm-up run ───────────────────────────────────────────────────────────────
println("Warm-up run (compiles all methods)...")
fit_model(dm, saem_cfg; rng=Xoshiro(1))
println("Warm-up done.")
println()

# ── Timed benchmark run ───────────────────────────────────────────────────────
println("=== SAEM benchmark (maxiters=40, 20 individuals, EnsembleThreads) ===")
println()

GC.gc()
t1 = @elapsed begin
    res = fit_model(dm, saem_cfg; rng=Xoshiro(2))
end

allocs = @allocated fit_model(dm, saem_cfg; rng=Xoshiro(2))

println("Wall time : $(round(t1; digits=3)) s")
println("Allocations: $(allocs) bytes  ($(round(allocs/1024^2; digits=2)) MB)")
println()
println("Converged : $(NoLimits.get_converged(res))")
println("Objective : $(round(NoLimits.get_objective(res); digits=4))")
println()
println("Parameters:")
display(NoLimits.get_params(res; scale=:untransformed))
