"""
Cost of computing the spectral gap vs the operations already on the hot path.
Run with:  julia --project test/ct_hmm_gap_cost.jl
"""

using LinearAlgebra, BenchmarkTools, ExponentialAction, Printf

function make_funnel_Q(n, fast, slow, back)
    Q = zeros(n, n)
    Q[1, 2] = fast;  Q[1, 1] = -fast
    for i in 2:(n-1)
        Q[i, i+1] = slow;  Q[i, i] = -slow
    end
    Q[n, 1] = back;  Q[n, n] = -back
    Q
end

function spectral_gap_eigvals(Q)
    λs = eigvals(Q)
    minimum(abs.(λs[abs.(λs) .> 1e-14]))
end

# Just the max-exit check (current shortcut gating — essentially free)
function max_exit(Q)
    maximum(-diag(Q))
end

# The stationary solve already done when shortcut fires
function stationary_solve(Q)
    n = size(Q, 1)
    A = Matrix{Float64}(transpose(Q))
    for j in 1:n; A[n, j] = 1.0; end
    b = zeros(n); b[n] = 1.0
    A \ b
end

# expv — the fallback when shortcut does NOT fire
function expv_propagate(Q, v, dt)
    expv(dt, transpose(Q), v)
end

println("=" ^ 68)
println(" Cost comparison (median times, BenchmarkTools)")
println("=" ^ 68)
println()

for n in [2, 3, 4, 5, 8, 10]
    Q  = make_funnel_Q(n, 1000.0, 0.01, 0.5)
    v0 = zeros(n); v0[1] = 1.0
    dt = 0.5

    t_gap  = @benchmark spectral_gap_eigvals($Q)      samples=1000 evals=10
    t_stat = @benchmark stationary_solve($Q)           samples=1000 evals=10
    t_expv = @benchmark expv_propagate($Q, $v0, $dt)  samples=1000 evals=10

    println("n = $n")
    @printf("  eigvals (spectral gap)  : %8.1f ns   allocs=%d\n",
            median(t_gap).time, t_gap.allocs)
    @printf("  stationary solve (A\\b)  : %8.1f ns   allocs=%d\n",
            median(t_stat).time, t_stat.allocs)
    @printf("  expv (current fallback) : %8.1f ns   allocs=%d\n",
            median(t_expv).time, t_expv.allocs)
    @printf("  gap / expv ratio        : %8.2f×\n",
            median(t_gap).time / median(t_expv).time)
    println()
end
