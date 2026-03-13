using NoLimits
using Random
using Statistics

include(joinpath(@__DIR__, "laplace_fastpath_baseline_benchmark.jl"))

const STEP5_DATE = "2026-03-13"

function _fit_timed(dm, method; seed::Int)
    t = @elapsed res = fit_model(dm, method; rng=MersenneTwister(seed))
    return (; runtime_s=t, objective=get_objective(res))
end

function run_step5_compare(; repeats::Int=2)
    cases = (:gaussian, :ode_offset, :ode_eta)
    rows = NamedTuple[]

    for case in cases
        df = make_df(case)
        m_l = model_laplace(case)
        m_m = model_laplace_map(case)
        if case == :ode_offset || case == :ode_eta
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

function write_step5_markdown(rows; path::String=joinpath(@__DIR__, "laplace_fastpath_step5_auto_vs_off_2026-03-13.md"))
    open(path, "w") do io
        println(io, "# Step 5 Fast-Path Comparison (`auto` vs `off`)")
        println(io)
        println(io, "- Date: ", STEP5_DATE)
        println(io, "- Scope: Step 5 activates fastpath for eligible ODE offset models with a generic polish pass.")
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
    rows = run_step5_compare()
    out = write_step5_markdown(rows)
    println("Wrote step5 auto-vs-off report to: ", out)
    for r in rows
        println(r)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
