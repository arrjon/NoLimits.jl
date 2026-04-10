# kernel.jl
# Batch-level log-likelihood computation via Smolyak sparse-grid quadrature.
# Implements signed logsumexp and the subject/batch integration kernel.

# ---------------------------------------------------------------------------
# Signed LogSumExp
# ---------------------------------------------------------------------------

"""
    signed_logsumexp(logvals, signs) -> (log_abs_result, result_sign::Int8)

Numerically stable computation of log|Σ_r s_r exp(a_r)| where s_r ∈ {+1, -1}.

Returns `(log|result|, sign(result))` as `(Float64, Int8)`.

The sum is split into positive and negative contributions, each stabilised by
the maximum over all `logvals`:

    pos = Σ_{s_r=+1} exp(a_r - amax)
    neg = Σ_{s_r=-1} exp(a_r - amax)
    result = exp(amax) * (pos - neg)

If |result| ≈ 0 (catastrophic cancellation), returns `(-Inf, +1)`.

This situation can occur at high Smolyak levels where the inclusion-exclusion
coefficients are large in magnitude and nearly cancel. Levels 1–3 are
numerically stable for typical NLME models.
"""
function signed_logsumexp(logvals::AbstractVector{T}, signs::AbstractVector{Int8}) where {T<:Number}
    isempty(logvals) && return (T(-Inf), Int8(1))

    amax = maximum(logvals)
    (isinf(amax) && amax < 0) && return (T(-Inf), Int8(1))

    pos = zero(T)
    neg = zero(T)
    @inbounds for i in eachindex(logvals, signs)
        shifted = exp(logvals[i] - amax)
        if signs[i] > 0
            pos += shifted
        else
            neg += shifted
        end
    end

    diff = pos - neg
    if diff > zero(T)
        return (amax + log(diff), Int8(1))
    elseif diff < zero(T)
        return (amax + log(-diff), Int8(-1))
    else
        return (T(-Inf), Int8(1))
    end
end

# ---------------------------------------------------------------------------
# Batch log-likelihood via sparse-grid quadrature
# ---------------------------------------------------------------------------

"""
    batch_loglik_ghq(dm, batch_info, θ, re_measure, sgrid, const_cache, ll_cache)
    -> Float64 (or Dual)

Estimate the batch marginal log-likelihood

    log ∫ p(y_batch | b, θ) p(b | θ) db

using the Smolyak sparse-grid quadrature rule `sgrid` and the RE measure
`re_measure`.

The integral is computed as:

    log L_batch ≈ signed_logsumexp_r [ log|W_r| + Σᵢ ℓᵢ(T(z_r), θ) + log c(z_r) ]

where `T(z_r) = transform(re_measure, z_r)` and `log c(z_r) = logcorrection(re_measure, z_r)`.

**For `GaussianRE`**: `logcorrection = 0`, and the Gauss-Hermite weights already
encode the N(z; 0, I) measure that integrates against the prior exactly via the
change of variables b = μ + Lz. The prior logpdf term is NOT added separately.

**ForwardDiff compatibility**: When `θ` carries Dual tags, the gradient flows
through `μ(θ)` and `L(θ)` in `re_measure`, and through `_loglikelihood_individual`.
The nodes `z_r` are precomputed Float64 constants.

Returns `-Inf` (promoting to the accumulator type) if:
- Any individual likelihood evaluates to `-Inf`
- The signed logsumexp result is negative (numerical instability warning)
- The result is non-finite
"""
function batch_loglik_ghq(
    dm::DataModel,
    batch_info::_LaplaceBatchInfo,
    θ::ComponentArray,
    re_measure::AbstractREMeasure,
    sgrid::GHQuadratureNodes{Float64},
    const_cache::LaplaceConstantsCache,
    ll_cache::_LLCache,
)
    R  = size(sgrid.nodes, 2)
    θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)

    # Determine accumulator element type from the RE measure.
    # When θ carries ForwardDiff.Dual tags, re_measure.μ has Dual elements,
    # so T = Dual. The Array{T} allocation handles type promotion seamlessly.
    T = eltype(re_measure)

    a_vals = Vector{T}(undef, R)

    @inbounds for r in 1:R
        z_r = sgrid.nodes[:, r]                       # Float64 column, no alloc
        b_r = transform(re_measure, z_r)              # T-valued: μ + L * z_r

        # Sum conditional log-likelihoods over all individuals in batch
        cond  = zero(T)
        valid = true
        for i in batch_info.inds
            η_i = _build_eta_ind(dm, i, batch_info, b_r, const_cache, θ_re)
            lli  = _loglikelihood_individual(dm, i, θ_re, η_i, ll_cache)
            if !isfinite(lli)
                valid = false
                break
            end
            cond += T(lli)
        end

        a_vals[r] = if valid
            T(sgrid.logweights[r]) + cond + logcorrection(re_measure, z_r)
        else
            T(-Inf)
        end
    end

    log_val, result_sign = signed_logsumexp(a_vals, sgrid.signs)

    if result_sign < 0
        @warn "GHQuadrature: batch marginal likelihood estimate is negative " *
              "(signed logsumexp returned negative result). " *
              "This indicates numerical instability — consider reducing `level` " *
              "or checking your model specification."
        return T(-Inf)
    end

    isfinite(log_val) || return T(-Inf)
    return log_val
end
