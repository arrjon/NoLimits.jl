"""
Diagnostic: accuracy of the equilibrium shortcut in _ct_hmm_probabilities_hidden_states.

The shortcut fires when  max_exit_rate * Δt > _CT_HMM_EQUIL_THRESHOLD (= 200)  and
returns the stationary distribution π instead of computing  exp(Q Δt) v.

Convergence to π depends on the *spectral gap*  (smallest |non-zero eigenvalue| of Q),
NOT on max_exit.  When a Q matrix has both very fast and very slow rates the shortcut
can trigger while slow states are still far from equilibrium.

This script quantifies the L∞ error  ‖exp(Q Δt) v - π‖∞  at the exact moment the
shortcut first fires  (i.e.  Δt = 200 / max_exit)  for several families of Q.

Run with:
    julia --project test/ct_hmm_equil_approx_accuracy.jl
"""

using LinearAlgebra
using ExponentialAction    # provides expv
using Printf

# ── helpers ──────────────────────────────────────────────────────────────────

const THRESHOLD = 200.0   # mirrors _CT_HMM_EQUIL_THRESHOLD

function stationary(Q)
    n = size(Q, 1)
    A = Matrix{Float64}(transpose(Q))
    for j in 1:n; A[n, j] = 1.0; end
    b = zeros(n); b[n] = 1.0
    π = A \ b
    any(x -> x < -sqrt(eps()), π) && return nothing
    return π ./ sum(π)
end

# Exact propagation via matrix exponential (reference)
function exact_propagate(Q, v, dt)
    p = expv(dt, transpose(Q), v)
    return p ./ sum(p)   # renormalise for safety
end

# Spectral gap of Q (smallest |non-zero eigenvalue|)
function spectral_gap(Q)
    λs = eigvals(Q)
    return minimum(abs.(λs[abs.(λs) .> 1e-14]))
end

# ─────────────────────────────────────────────────────────────────────────────
# Scenario builder helpers
# ─────────────────────────────────────────────────────────────────────────────

# 2-state ergodic chain with rates q12 and q21
function two_state_Q(q12, q21)
    [-(q12)  q12
      q21   -(q21)]
end

# 3-state cyclic chain  1→2→3→1 with rates λ12, λ23, λ31
function three_state_cyclic_Q(λ12, λ23, λ31)
    [-λ12   λ12    0.0
      0.0  -λ23   λ23
      λ31   0.0  -λ31]
end

# n-state "funnel" Q:  1→2 is fast, 2→3→…→n is slow chain, n→1 moderate
function funnel_Q(n, fast_rate, slow_rate, back_rate)
    Q = zeros(n, n)
    Q[1, 2] = fast_rate;  Q[1, 1] = -fast_rate
    for i in 2:(n-1)
        Q[i, i+1] = slow_rate;  Q[i, i] = -slow_rate
    end
    Q[n, 1] = back_rate;  Q[n, n] = -back_rate
    return Q
end

# ─────────────────────────────────────────────────────────────────────────────
# Analysis: sweep Δt from threshold-trigger to 10× threshold for a given Q,v
# and report L∞ error of the stationary approximation vs exact propagation.
# ─────────────────────────────────────────────────────────────────────────────
function analyse(label, Q, v0)
    π = stationary(Q)
    if π === nothing
        println("  [$label] non-ergodic Q — skipped")
        return
    end
    max_exit = maximum(-diag(Q))
    gap      = spectral_gap(Q)
    dt_trig  = THRESHOLD / max_exit   # Δt where shortcut first fires

    println("  $label")
    @printf("    max_exit_rate  = %.4g\n", max_exit)
    @printf("    spectral_gap   = %.4g\n", gap)
    @printf("    gap/max_exit   = %.4g  (ratio; 1 = uniform rates)\n",
            gap / max_exit)
    @printf("    Δt_trigger     = %.4g  (shortcut first fires here)\n", dt_trig)
    println()

    # Δt grid: just at trigger, 2×, 5×, 10×, 100×
    for mult in [1.0, 2.0, 5.0, 10.0, 100.0]
        dt   = mult * dt_trig
        p    = exact_propagate(Q, v0, dt)
        err  = maximum(abs.(p .- π))
        @printf("    Δt = %5.1f × Δt_trig  (%9.3g) │ max|exp(QΔt)v - π|∞ = %.3e\n",
                mult, dt, err)
    end
    println()
end

# ─────────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────────

println("=" ^ 72)
println(" Equilibrium-shortcut accuracy: ‖exp(Q Δt) v − π‖∞")
println(" Shortcut fires when max_exit * Δt > ", THRESHOLD)
println("=" ^ 72)
println()

println("── Scenario A: 2-state ergodic, equal rates (baseline) ──────────────")
analyse("q12=q21=300", two_state_Q(300.0, 300.0), [1.0, 0.0])

println("── Scenario B: 2-state ergodic, DISPARATE rates ─────────────────────")
for (q12, q21) in [(300.0, 1.0), (300.0, 0.1), (300.0, 0.01), (300.0, 0.001)]
    analyse("q12=$(q12), q21=$(q21)", two_state_Q(q12, q21), [1.0, 0.0])
end

println("── Scenario C: 3-state cyclic, one slow link ────────────────────────")
for slow in [1.0, 0.1, 0.01, 0.001]
    Q = three_state_cyclic_Q(500.0, slow, 200.0)
    analyse("λ12=500, λ23=$(slow), λ31=200  v=[1,0,0]", Q, [1.0, 0.0, 0.0])
end

println("── Scenario D: 3-state cyclic, worst-case initial condition ─────────")
# v0 = uniform start is easy; worst case is the state that is hardest to mix
for slow in [0.1, 0.01]
    Q = three_state_cyclic_Q(500.0, slow, 200.0)
    for (vname, v0) in [("v=[1,0,0]", [1.0,0.0,0.0]),
                        ("v=[0,1,0]", [0.0,1.0,0.0]),
                        ("v=[0,0,1]", [0.0,0.0,1.0])]
        analyse("λ12=500, λ23=$(slow), λ31=200  $vname", Q, v0)
    end
end

println("── Scenario E: n-state funnel, slow interior chain ──────────────────")
for (n, slow) in [(3, 0.01), (4, 0.01), (5, 0.01), (5, 0.001)]
    Q  = funnel_Q(n, 1000.0, slow, 0.5)
    v0 = zeros(n); v0[1] = 1.0
    analyse("funnel n=$n, fast=1000, slow=$(slow), back=0.5", Q, v0)
end

# ─────────────────────────────────────────────────────────────────────────────
# Summary / diagnosis
# ─────────────────────────────────────────────────────────────────────────────
println("=" ^ 72)
println(" SUMMARY")
println("=" ^ 72)
println("""
The shortcut  exp(Q Δt) v ≈ π  is valid iff  gap × Δt >> 1,
where  gap = smallest |non-zero eigenvalue| of Q.

The trigger condition  max_exit × Δt > 200  is sufficient only when
the spectral gap is comparable to max_exit.  In two-state chains this
always holds because  gap = q12 + q21 ≥ max_exit.  In cyclic 3-state
chains with the rates tested it also happens to hold.

The approximation FAILS when a "funnel" topology exists:
  one very fast entry rate → slow bottleneck chain of length k

In that case:
  gap  ≈  slow_rate  <<  max_exit
  Δt at trigger ≈ 200 / max_exit  (very small)
  gap × Δt_trigger ≈ 200 × slow_rate / max_exit  ≪ 1

Observed errors at trigger Δt = 200 / max_exit:
  n=3 funnel, slow=0.01, max=1000  →  ~1.8%   error
  n=4 funnel, slow=0.01, max=1000  →  ~50%    error
  n=5 funnel, slow=0.01, max=1000  →  ~67%    error
  n=5 funnel, slow=0.001, max=1000 →  ~67%    error  (still ~65% at 100× Δt!)

Proposed fix:
  Replace the threshold condition with:
      gap * Δt > THRESHOLD
  where  gap  is the spectral gap of Q.
  This guarantees  ‖exp(Q Δt) v − π‖  < ε  uniformly.

  Cost: one eigenvalue computation per (Q, Δt) pair where max_exit > THRESHOLD/Δt.
  Since those are the only calls that would trigger the shortcut, the overhead
  is minimal (and avoids the far more expensive expv call in the true path).
""")
println("=" ^ 72)
