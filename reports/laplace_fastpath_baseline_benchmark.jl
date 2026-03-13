using NoLimits
using DataFrames
using Random
using Distributions
using Statistics

const BASELINE_DATE = "2026-03-12"

function make_df(kind::Symbol; n_id::Int=20, n_obs::Int=6, seed::Int=123)
    rng = MersenneTwister(seed)
    n = n_id * n_obs
    ID = repeat(1:n_id, inner=n_obs)
    t = repeat(collect(0.0:(n_obs - 1)), n_id)
    z = randn(rng, n)
    η = 0.6 .* randn(rng, n_id)

    if kind == :gaussian
        y = [0.2 + 0.6 * z[i] + η[ID[i]] + 0.3 * randn(rng) for i in 1:n]
        return DataFrame(ID=ID, t=t, z=z, y=y)
    elseif kind == :lognormal
        y = [exp(-0.1 + 0.4 * z[i] + η[ID[i]] + 0.25 * randn(rng)) for i in 1:n]
        return DataFrame(ID=ID, t=t, z=z, y=y)
    elseif kind == :bernoulli
        y = Vector{Int}(undef, n)
        for i in eachindex(y)
            p = inv(1 + exp(-(0.1 + 0.5 * z[i] + η[ID[i]])))
            y[i] = rand(rng) < p ? 1 : 0
        end
        return DataFrame(ID=ID, t=t, z=z, y=y)
    elseif kind == :poisson
        y = [rand(rng, Poisson(exp(-0.2 + 0.4 * z[i] + 0.7 * η[ID[i]]))) for i in 1:n]
        return DataFrame(ID=ID, t=t, z=z, y=y)
    elseif kind == :ode_offset
        y = [exp(-0.3 * t[i]) + η[ID[i]] + 0.2 * randn(rng) for i in 1:n]
        return DataFrame(ID=ID, t=t, y=y)
    elseif kind == :ode_eta
        y = [exp(-(0.3 + 0.2 * η[ID[i]]) * t[i]) + 0.2 * randn(rng) for i in 1:n]
        return DataFrame(ID=ID, t=t, y=y)
    else
        error("Unknown kind: $(kind)")
    end
end

function model_laplace(kind::Symbol)
    if kind == :gaussian
        return @Model begin
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
    elseif kind == :lognormal
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0)
                b = RealNumber(0.4)
                σ = RealNumber(0.4, scale=:log)
                τ = RealNumber(0.7, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @formulas begin
                y ~ LogNormal(a + b * z + η, σ)
            end
        end
    elseif kind == :bernoulli
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0)
                b = RealNumber(0.4)
                τ = RealNumber(0.7, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @formulas begin
                p = logistic(a + b * z + η)
                y ~ Bernoulli(p)
            end
        end
    elseif kind == :poisson
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0)
                b = RealNumber(0.4)
                τ = RealNumber(0.7, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @formulas begin
                λ = exp(a + b * z + η)
                y ~ Poisson(λ)
            end
        end
    elseif kind == :ode_offset
        return @Model begin
            @covariates begin
                t = Covariate()
            end
            @fixedEffects begin
                k = RealNumber(0.3, scale=:log)
                σ = RealNumber(0.3, scale=:log)
                τ = RealNumber(0.6, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @DifferentialEquation begin
                D(x1) ~ -k * x1
            end
            @initialDE begin
                x1 = 1.0
            end
            @formulas begin
                y ~ Normal(x1(t) + η, σ)
            end
        end
    elseif kind == :ode_eta
        return @Model begin
            @covariates begin
                t = Covariate()
            end
            @fixedEffects begin
                k = RealNumber(0.3, scale=:log)
                σ = RealNumber(0.3, scale=:log)
                τ = RealNumber(0.6, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @DifferentialEquation begin
                D(x1) ~ -(k + η) * x1
            end
            @initialDE begin
                x1 = 1.0
            end
            @formulas begin
                y ~ Normal(x1(t), σ)
            end
        end
    else
        error("Unknown kind: $(kind)")
    end
end

function model_laplace_map(kind::Symbol)
    if kind == :gaussian
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0, prior=Normal(0.0, 1.0))
                b = RealNumber(0.4, prior=Normal(0.0, 1.0))
                σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.7))
                τ = RealNumber(0.7, scale=:log, prior=LogNormal(0.0, 0.7))
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + b * z + η, σ)
            end
        end
    elseif kind == :lognormal
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0, prior=Normal(0.0, 1.0))
                b = RealNumber(0.4, prior=Normal(0.0, 1.0))
                σ = RealNumber(0.4, scale=:log, prior=LogNormal(0.0, 0.7))
                τ = RealNumber(0.7, scale=:log, prior=LogNormal(0.0, 0.7))
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @formulas begin
                y ~ LogNormal(a + b * z + η, σ)
            end
        end
    elseif kind == :bernoulli
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0, prior=Normal(0.0, 1.0))
                b = RealNumber(0.4, prior=Normal(0.0, 1.0))
                τ = RealNumber(0.7, scale=:log, prior=LogNormal(0.0, 0.7))
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @formulas begin
                p = logistic(a + b * z + η)
                y ~ Bernoulli(p)
            end
        end
    elseif kind == :poisson
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0, prior=Normal(0.0, 1.0))
                b = RealNumber(0.4, prior=Normal(0.0, 1.0))
                τ = RealNumber(0.7, scale=:log, prior=LogNormal(0.0, 0.7))
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @formulas begin
                λ = exp(a + b * z + η)
                y ~ Poisson(λ)
            end
        end
    elseif kind == :ode_offset
        return @Model begin
            @covariates begin
                t = Covariate()
            end
            @fixedEffects begin
                k = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.7))
                σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.7))
                τ = RealNumber(0.6, scale=:log, prior=LogNormal(0.0, 0.7))
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @DifferentialEquation begin
                D(x1) ~ -k * x1
            end
            @initialDE begin
                x1 = 1.0
            end
            @formulas begin
                y ~ Normal(x1(t) + η, σ)
            end
        end
    elseif kind == :ode_eta
        return @Model begin
            @covariates begin
                t = Covariate()
            end
            @fixedEffects begin
                k = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.7))
                σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.7))
                τ = RealNumber(0.6, scale=:log, prior=LogNormal(0.0, 0.7))
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @DifferentialEquation begin
                D(x1) ~ -(k + η) * x1
            end
            @initialDE begin
                x1 = 1.0
            end
            @formulas begin
                y ~ Normal(x1(t), σ)
            end
        end
    else
        error("Unknown kind: $(kind)")
    end
end

function _fit_with_timing(dm, method; warmup::Bool=true)
    warmup && fit_model(dm, method)
    t = @elapsed res = fit_model(dm, method)
    return (; runtime_s=t, objective=get_objective(res), converged=get_converged(res))
end

function run_baseline(; repeats::Int=2)
    cases = (:gaussian, :lognormal, :bernoulli, :poisson, :ode_offset, :ode_eta)
    rows = NamedTuple[]

    for case in cases
        df = make_df(case)
        m_l = model_laplace(case)
        m_m = model_laplace_map(case)
        m_l = (case == :ode_offset || case == :ode_eta) ? set_solver_config(m_l; saveat_mode=:saveat) : m_l
        m_m = (case == :ode_offset || case == :ode_eta) ? set_solver_config(m_m; saveat_mode=:saveat) : m_m
        dm_l = DataModel(m_l, df; primary_id=:ID, time_col=:t)
        dm_m = DataModel(m_m, df; primary_id=:ID, time_col=:t)

        lap = NoLimits.Laplace(; optim_kwargs=(maxiters=25,), inner_kwargs=(maxiters=20,), multistart_n=0, multistart_k=0)
        lmap = NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=25,), inner_kwargs=(maxiters=20,), multistart_n=0, multistart_k=0)

        lap_times = Float64[]
        lap_objs = Float64[]
        lap_conv = Bool[]
        map_times = Float64[]
        map_objs = Float64[]
        map_conv = Bool[]
        for r in 1:repeats
            lr = _fit_with_timing(dm_l, lap; warmup=r == 1)
            mr = _fit_with_timing(dm_m, lmap; warmup=r == 1)
            push!(lap_times, lr.runtime_s)
            push!(lap_objs, lr.objective)
            push!(lap_conv, Bool(lr.converged))
            push!(map_times, mr.runtime_s)
            push!(map_objs, mr.objective)
            push!(map_conv, Bool(mr.converged))
        end

        push!(rows, (;
            case=String(case),
            n_rows=nrow(df),
            laplace_time_s=median(lap_times),
            laplace_obj=median(lap_objs),
            laplace_conv=all(lap_conv),
            laplace_map_time_s=median(map_times),
            laplace_map_obj=median(map_objs),
            laplace_map_conv=all(map_conv),
        ))
    end
    return rows
end

function write_markdown(rows; path::String=joinpath(@__DIR__, "laplace_fastpath_baseline_results_2026-03-12.md"))
    io = IOBuffer()
    println(io, "# Laplace Fast-Path Baseline (Fallback Reference)")
    println(io)
    println(io, "- Date: ", BASELINE_DATE)
    println(io, "- Notes: This is the pre-fast-path fallback baseline. Use for `auto` vs `off` comparisons in subsequent steps.")
    println(io)
    println(io, "| case | n_rows | laplace_time_s | laplace_obj | laplace_conv | laplace_map_time_s | laplace_map_obj | laplace_map_conv |")
    println(io, "|---|---:|---:|---:|:---:|---:|---:|:---:|")
    for r in rows
        println(io,
                "| ", r.case,
                " | ", r.n_rows,
                " | ", round(r.laplace_time_s, digits=4),
                " | ", round(r.laplace_obj, digits=4),
                " | ", r.laplace_conv,
                " | ", round(r.laplace_map_time_s, digits=4),
                " | ", round(r.laplace_map_obj, digits=4),
                " | ", r.laplace_map_conv,
                " |")
    end
    open(path, "w") do f
        write(f, String(take!(io)))
    end
    return path
end

function main()
    rows = run_baseline()
    out = write_markdown(rows)
    println("Wrote baseline report to: ", out)
    for r in rows
        println(r)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
