using NoLimits
using DataFrames
using Random
using Distributions
using Statistics

include(joinpath(@__DIR__, "laplace_fastpath_baseline_benchmark.jl"))

const STEP4_DATE = "2026-03-13"
const _STEP4_DENSE_CASES = Set((:gaussian_dense, :lognormal_dense, :bernoulli_dense, :poisson_dense))

function _fit_timed(dm, method; seed::Int)
    t = @elapsed res = fit_model(dm, method; rng=MersenneTwister(seed))
    return (; runtime_s=t, objective=get_objective(res))
end

function _step4_make_df_dense(case::Symbol; n_id::Int=20, n_obs::Int=6, seed::Int=123)
    rng = MersenneTwister(seed)
    n = n_id * n_obs
    ID = repeat(1:n_id, inner=n_obs)
    t = repeat(collect(0.0:(n_obs - 1)), n_id)
    z = randn(rng, n)
    η0 = 0.6 .* randn(rng, n_id)
    η1 = 0.4 .* randn(rng, n_id)

    if case == :gaussian_dense
        y = [0.2 + 0.6 * z[i] + η0[ID[i]] + 0.5 * z[i] * η1[ID[i]] + 0.3 * randn(rng) for i in 1:n]
        return DataFrame(ID=ID, t=t, z=z, y=y)
    elseif case == :lognormal_dense
        y = [exp(-0.1 + 0.5 * z[i] + η0[ID[i]] + 0.4 * z[i] * η1[ID[i]] + 0.25 * randn(rng)) for i in 1:n]
        return DataFrame(ID=ID, t=t, z=z, y=y)
    elseif case == :bernoulli_dense
        y = Vector{Int}(undef, n)
        for i in eachindex(y)
            p = inv(1 + exp(-(0.1 + 0.5 * z[i] + η0[ID[i]] + 0.4 * z[i] * η1[ID[i]])))
            y[i] = rand(rng) < p ? 1 : 0
        end
        return DataFrame(ID=ID, t=t, z=z, y=y)
    elseif case == :poisson_dense
        y = [rand(rng, Poisson(exp(-0.2 + 0.4 * z[i] + η0[ID[i]] + 0.4 * z[i] * η1[ID[i]]))) for i in 1:n]
        return DataFrame(ID=ID, t=t, z=z, y=y)
    else
        error("Unknown dense case: $(case)")
    end
end

function _step4_model_laplace_dense(case::Symbol)
    if case == :gaussian_dense
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0)
                b = RealNumber(0.4)
                σ = RealNumber(0.5, scale=:log)
                τ0 = RealNumber(0.7, scale=:log)
                τ1 = RealNumber(0.5, scale=:log)
            end
            @randomEffects begin
                η0 = RandomEffect(Normal(0.0, τ0); column=:ID)
                η1 = RandomEffect(Normal(0.0, τ1); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + b * z + η0 + z * η1, σ)
            end
        end
    elseif case == :lognormal_dense
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0)
                b = RealNumber(0.4)
                σ = RealNumber(0.4, scale=:log)
                τ0 = RealNumber(0.7, scale=:log)
                τ1 = RealNumber(0.5, scale=:log)
            end
            @randomEffects begin
                η0 = RandomEffect(Normal(0.0, τ0); column=:ID)
                η1 = RandomEffect(Normal(0.0, τ1); column=:ID)
            end
            @formulas begin
                y ~ LogNormal(a + b * z + η0 + z * η1, σ)
            end
        end
    elseif case == :bernoulli_dense
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0)
                b = RealNumber(0.4)
                τ0 = RealNumber(0.7, scale=:log)
                τ1 = RealNumber(0.5, scale=:log)
            end
            @randomEffects begin
                η0 = RandomEffect(Normal(0.0, τ0); column=:ID)
                η1 = RandomEffect(Normal(0.0, τ1); column=:ID)
            end
            @formulas begin
                p = logistic(a + b * z + η0 + z * η1)
                y ~ Bernoulli(p)
            end
        end
    elseif case == :poisson_dense
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0)
                b = RealNumber(0.4)
                τ0 = RealNumber(0.7, scale=:log)
                τ1 = RealNumber(0.5, scale=:log)
            end
            @randomEffects begin
                η0 = RandomEffect(Normal(0.0, τ0); column=:ID)
                η1 = RandomEffect(Normal(0.0, τ1); column=:ID)
            end
            @formulas begin
                λ = exp(a + b * z + η0 + z * η1)
                y ~ Poisson(λ)
            end
        end
    end
    error("Unknown dense case: $(case)")
end

function _step4_model_laplace_map_dense(case::Symbol)
    if case == :gaussian_dense
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0, prior=Normal(0.0, 1.0))
                b = RealNumber(0.4, prior=Normal(0.0, 1.0))
                σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.7))
                τ0 = RealNumber(0.7, scale=:log, prior=LogNormal(0.0, 0.7))
                τ1 = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.7))
            end
            @randomEffects begin
                η0 = RandomEffect(Normal(0.0, τ0); column=:ID)
                η1 = RandomEffect(Normal(0.0, τ1); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + b * z + η0 + z * η1, σ)
            end
        end
    elseif case == :lognormal_dense
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0, prior=Normal(0.0, 1.0))
                b = RealNumber(0.4, prior=Normal(0.0, 1.0))
                σ = RealNumber(0.4, scale=:log, prior=LogNormal(0.0, 0.7))
                τ0 = RealNumber(0.7, scale=:log, prior=LogNormal(0.0, 0.7))
                τ1 = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.7))
            end
            @randomEffects begin
                η0 = RandomEffect(Normal(0.0, τ0); column=:ID)
                η1 = RandomEffect(Normal(0.0, τ1); column=:ID)
            end
            @formulas begin
                y ~ LogNormal(a + b * z + η0 + z * η1, σ)
            end
        end
    elseif case == :bernoulli_dense
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0, prior=Normal(0.0, 1.0))
                b = RealNumber(0.4, prior=Normal(0.0, 1.0))
                τ0 = RealNumber(0.7, scale=:log, prior=LogNormal(0.0, 0.7))
                τ1 = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.7))
            end
            @randomEffects begin
                η0 = RandomEffect(Normal(0.0, τ0); column=:ID)
                η1 = RandomEffect(Normal(0.0, τ1); column=:ID)
            end
            @formulas begin
                p = logistic(a + b * z + η0 + z * η1)
                y ~ Bernoulli(p)
            end
        end
    elseif case == :poisson_dense
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0, prior=Normal(0.0, 1.0))
                b = RealNumber(0.4, prior=Normal(0.0, 1.0))
                τ0 = RealNumber(0.7, scale=:log, prior=LogNormal(0.0, 0.7))
                τ1 = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.7))
            end
            @randomEffects begin
                η0 = RandomEffect(Normal(0.0, τ0); column=:ID)
                η1 = RandomEffect(Normal(0.0, τ1); column=:ID)
            end
            @formulas begin
                λ = exp(a + b * z + η0 + z * η1)
                y ~ Poisson(λ)
            end
        end
    end
    error("Unknown dense case: $(case)")
end

@inline function _step4_make_df(case::Symbol)
    case in _STEP4_DENSE_CASES ? _step4_make_df_dense(case) : make_df(case)
end

@inline function _step4_model_laplace(case::Symbol)
    case in _STEP4_DENSE_CASES ? _step4_model_laplace_dense(case) : model_laplace(case)
end

@inline function _step4_model_laplace_map(case::Symbol)
    case in _STEP4_DENSE_CASES ? _step4_model_laplace_map_dense(case) : model_laplace_map(case)
end

@inline function _step4_case_requires_saveat(case::Symbol)
    return case == :ode_offset || case == :ode_eta
end

function run_step4_compare(; repeats::Int=2)
    cases = (:gaussian, :lognormal, :bernoulli, :poisson,
             :gaussian_dense, :lognormal_dense, :bernoulli_dense, :poisson_dense,
             :ode_offset, :ode_eta)
    rows = NamedTuple[]

    for case in cases
        df = _step4_make_df(case)
        m_l = _step4_model_laplace(case)
        m_m = _step4_model_laplace_map(case)
        if _step4_case_requires_saveat(case)
            m_l = set_solver_config(m_l; saveat_mode=:saveat)
            m_m = set_solver_config(m_m; saveat_mode=:saveat)
        end
        dm_l = DataModel(m_l, df; primary_id=:ID, time_col=:t)
        dm_m = DataModel(m_m, df; primary_id=:ID, time_col=:t)

        lap_auto = NoLimits.Laplace(; optim_kwargs=(maxiters=10,), inner_kwargs=(maxiters=25,), multistart_n=0, multistart_k=0, fastpath_mode=:auto)
        lap_off = NoLimits.Laplace(; optim_kwargs=(maxiters=10,), inner_kwargs=(maxiters=25,), multistart_n=0, multistart_k=0, fastpath_mode=:off)
        map_auto = NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=10,), inner_kwargs=(maxiters=25,), multistart_n=0, multistart_k=0, fastpath_mode=:auto)
        map_off = NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=10,), inner_kwargs=(maxiters=25,), multistart_n=0, multistart_k=0, fastpath_mode=:off)

        _ = _fit_timed(dm_l, lap_auto; seed=11)
        _ = _fit_timed(dm_l, lap_off; seed=11)
        _ = _fit_timed(dm_m, map_auto; seed=11)
        _ = _fit_timed(dm_m, map_off; seed=11)

        lap_auto_times = Float64[]
        lap_off_times = Float64[]
        map_auto_times = Float64[]
        map_off_times = Float64[]
        lap_diffs = Float64[]
        map_diffs = Float64[]

        for r in 1:repeats
            seed = 100 + r
            la = _fit_timed(dm_l, lap_auto; seed=seed)
            lo = _fit_timed(dm_l, lap_off; seed=seed)
            ma = _fit_timed(dm_m, map_auto; seed=seed)
            mo = _fit_timed(dm_m, map_off; seed=seed)

            push!(lap_auto_times, la.runtime_s)
            push!(lap_off_times, lo.runtime_s)
            push!(map_auto_times, ma.runtime_s)
            push!(map_off_times, mo.runtime_s)
            push!(lap_diffs, abs(la.objective - lo.objective))
            push!(map_diffs, abs(ma.objective - mo.objective))
        end

        lap_auto_t = median(lap_auto_times)
        lap_off_t = median(lap_off_times)
        map_auto_t = median(map_auto_times)
        map_off_t = median(map_off_times)

        push!(rows, (;
            case=String(case),
            n_rows=nrow(df),
            laplace_auto_time_s=lap_auto_t,
            laplace_off_time_s=lap_off_t,
            laplace_speedup=lap_auto_t > 0 ? lap_off_t / lap_auto_t : NaN,
            laplace_abs_obj_diff=median(lap_diffs),
            laplace_map_auto_time_s=map_auto_t,
            laplace_map_off_time_s=map_off_t,
            laplace_map_speedup=map_auto_t > 0 ? map_off_t / map_auto_t : NaN,
            laplace_map_abs_obj_diff=median(map_diffs),
        ))
    end
    return rows
end

function write_step4_markdown(rows; path::String=joinpath(@__DIR__, "laplace_fastpath_step4_auto_vs_off_2026-03-13.md"))
    open(path, "w") do io
        println(io, "# Step 4 Fast-Path Comparison (`auto` vs `off`)")
        println(io)
        println(io, "- Date: ", STEP4_DATE)
        println(io, "- Scope: Step 4 enables Newton-inner fast backend for fully eligible models, including dense (`n_b > 1`) batches.")
        println(io, "- Speedup is reported as `off_time / auto_time` (values > 1 mean `auto` is faster).")
        println(io)
        println(io, "| case | n_rows | laplace_auto_time_s | laplace_off_time_s | laplace_speedup | laplace_abs_obj_diff | laplace_map_auto_time_s | laplace_map_off_time_s | laplace_map_speedup | laplace_map_abs_obj_diff |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows
            println(io,
                    "| ", r.case,
                    " | ", r.n_rows,
                    " | ", round(r.laplace_auto_time_s, digits=6),
                    " | ", round(r.laplace_off_time_s, digits=6),
                    " | ", round(r.laplace_speedup, digits=4),
                    " | ", round(r.laplace_abs_obj_diff, digits=12),
                    " | ", round(r.laplace_map_auto_time_s, digits=6),
                    " | ", round(r.laplace_map_off_time_s, digits=6),
                    " | ", round(r.laplace_map_speedup, digits=4),
                    " | ", round(r.laplace_map_abs_obj_diff, digits=12),
                    " |")
        end
    end
    return path
end

function main()
    rows = run_step4_compare()
    out = write_step4_markdown(rows)
    println("Wrote step4 auto-vs-off report to: ", out)
    for r in rows
        println(r)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
