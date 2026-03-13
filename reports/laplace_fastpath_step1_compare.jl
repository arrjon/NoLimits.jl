using NoLimits
using Random
using Statistics

include(joinpath(@__DIR__, "laplace_fastpath_baseline_benchmark.jl"))

const STEP1_DATE = "2026-03-12"

function run_step1_compare(; repeats::Int=1)
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

        lap_auto = NoLimits.Laplace(; optim_kwargs=(maxiters=10,), inner_kwargs=(maxiters=15,), multistart_n=0, multistart_k=0, fastpath_mode=:auto)
        lap_off = NoLimits.Laplace(; optim_kwargs=(maxiters=10,), inner_kwargs=(maxiters=15,), multistart_n=0, multistart_k=0, fastpath_mode=:off)
        map_auto = NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=10,), inner_kwargs=(maxiters=15,), multistart_n=0, multistart_k=0, fastpath_mode=:auto)
        map_off = NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=10,), inner_kwargs=(maxiters=15,), multistart_n=0, multistart_k=0, fastpath_mode=:off)

        lap_diffs = Float64[]
        map_diffs = Float64[]
        for r in 1:repeats
            seed = 100 + r
            res_l_auto = fit_model(dm_l, lap_auto; rng=MersenneTwister(seed))
            res_l_off = fit_model(dm_l, lap_off; rng=MersenneTwister(seed))
            push!(lap_diffs, abs(get_objective(res_l_auto) - get_objective(res_l_off)))

            res_m_auto = fit_model(dm_m, map_auto; rng=MersenneTwister(seed))
            res_m_off = fit_model(dm_m, map_off; rng=MersenneTwister(seed))
            push!(map_diffs, abs(get_objective(res_m_auto) - get_objective(res_m_off)))
        end

        push!(rows, (;
            case=String(case),
            n_rows=nrow(df),
            laplace_abs_obj_diff=median(lap_diffs),
            laplace_map_abs_obj_diff=median(map_diffs),
        ))
    end
    return rows
end

function write_step1_markdown(rows; path::String=joinpath(@__DIR__, "laplace_fastpath_step1_auto_vs_off_2026-03-12.md"))
    open(path, "w") do io
        println(io, "# Step 1 Fast-Path Comparison (`auto` vs `off`)")
        println(io)
        println(io, "- Date: ", STEP1_DATE)
        println(io, "- Scope: Step 1 adds configuration + logging only (no numerical fast-path backend).")
        println(io)
        println(io, "| case | n_rows | laplace_abs_obj_diff | laplace_map_abs_obj_diff |")
        println(io, "|---|---:|---:|---:|")
        for r in rows
            println(io,
                    "| ", r.case,
                    " | ", r.n_rows,
                    " | ", round(r.laplace_abs_obj_diff, digits=12),
                    " | ", round(r.laplace_map_abs_obj_diff, digits=12),
                    " |")
        end
    end
    return path
end

function main()
    rows = run_step1_compare()
    out = write_step1_markdown(rows)
    println("Wrote step1 auto-vs-off report to: ", out)
    for r in rows
        println(r)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
