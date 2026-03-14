# nodes.jl
# Gauss-Hermite quadrature rules and Smolyak sparse-grid construction.
# All nodes are in the probabilist's convention: integrates f(z) under N(0,I).

using LinearAlgebra

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

"""
    GHQuadratureNodes{T}

Precomputed Smolyak sparse-grid quadrature rule in `dim` dimensions at
accuracy `level`.

Fields:
- `nodes`:      `(dim, R)` matrix — each column is one quadrature point z_r ∈ ℝ^dim
- `logweights`: `(R,)` vector — log|w_r| for each point (weights can be negative)
- `signs`:      `(R,)` Int8 vector — net sign (+1 or -1) of w_r

Usage: approximate ∫ f(z) N(z; 0, I) dz ≈ signed_sum_r w_r f(z_r)
where w_r = signs[r] * exp(logweights[r]).
"""
struct GHQuadratureNodes{T<:AbstractFloat}
    dim::Int
    level::Int
    nodes::Matrix{T}
    logweights::Vector{T}
    signs::Vector{Int8}
end

# ---------------------------------------------------------------------------
# 1D Gauss-Hermite rule (probabilist's convention)
# ---------------------------------------------------------------------------

"""
    _gh_rule(n::Int) -> (nodes::Vector{Float64}, logweights::Vector{Float64})

Compute `n`-point Gauss-Hermite quadrature in the probabilist's convention via
the Golub-Welsch (Jacobi matrix eigenvalue) method.

Convention: ∫ f(x) N(x; 0, 1) dx ≈ Σ exp(logweights[k]) * f(nodes[k])
Weights sum to 1.0 exactly (log weights sum to log(1) = 0 approximately).

For n=1: single node at 0, weight 1.0.
"""
function _gh_rule(n::Int)
    if n == 1
        return [0.0], [0.0]  # node=0, logweight=log(1)=0
    end

    # Jacobi tridiagonal matrix for physicist's GH
    β = [sqrt(i / 2.0) for i in 1:(n-1)]
    J = SymTridiagonal(zeros(n), β)

    # Eigendecomposition — eigenvectors columns of vecs, eigenvalues = physicist nodes
    F = eigen(J)
    x_phys = F.values
    vecs   = F.vectors

    # Physicist weights: w_k^phys = sqrt(π) * vecs[1,k]^2
    w_phys = sqrt(π) .* vecs[1, :] .^ 2

    # Convert to probabilist's convention:
    #   nodes_prob = sqrt(2) * x_phys
    #   w_prob = w_phys / sqrt(π)   (so they sum to 1)
    nodes_prob = sqrt(2.0) .* x_phys
    w_prob     = w_phys ./ sqrt(π)

    # Sort by node value (ascending)
    ord = sortperm(nodes_prob)
    return nodes_prob[ord], log.(w_prob[ord])
end

# ---------------------------------------------------------------------------
# Multi-index enumeration
# ---------------------------------------------------------------------------

# Enumerate all multi-indices α ∈ Z_{≥1}^dim with |α|₁ ≤ q_max (in-place).
function _enumerate_multiindices!(result::Vector{Vector{Int}}, dim::Int, q_max::Int,
                                   current::Vector{Int}, current_sum::Int)
    if length(current) == dim
        push!(result, copy(current))
        return
    end
    remaining = dim - length(current)
    max_here  = q_max - current_sum - (remaining - 1)   # leave at least 1 for each remaining
    for αi in 1:max_here
        push!(current, αi)
        _enumerate_multiindices!(result, dim, q_max, current, current_sum + αi)
        pop!(current)
    end
end

function _smolyak_multiindices(dim::Int, level::Int)
    q_max  = dim + level - 1
    result = Vector{Vector{Int}}()
    _enumerate_multiindices!(result, dim, q_max, Int[], 0)
    return result
end

# ---------------------------------------------------------------------------
# Smolyak sparse-grid construction
# ---------------------------------------------------------------------------

"""
    build_sparse_grid(dim::Int, level::Int) -> GHQuadratureNodes{Float64}

Build the Smolyak sparse-grid quadrature rule in `dim` dimensions at accuracy
`level` using Gauss-Hermite 1D rules (probabilist's convention).

The rule approximates:
    ∫ f(z) N(z; 0, I_dim) dz ≈ Σ_r signs[r] * exp(logweights[r]) * f(nodes[:,r])

Duplicate nodes (which arise from the Smolyak tensor-product construction) are
merged by summing their signed weights. Near-zero combined weights (|w| < eps)
are discarded.
"""
function build_sparse_grid(dim::Int, level::Int)
    @assert dim  >= 1 "dim must be ≥ 1"
    @assert level >= 1 "level must be ≥ 1"

    q_max  = dim + level - 1

    # Precompute 1D GH rules for orders 1..level
    rules_nodes = Vector{Vector{Float64}}(undef, level)
    rules_lw    = Vector{Vector{Float64}}(undef, level)
    for k in 1:level
        rules_nodes[k], rules_lw[k] = _gh_rule(k)
    end

    # Storage for assembled points
    all_nodes   = Vector{Vector{Float64}}()
    all_lw      = Vector{Float64}()
    all_signs   = Vector{Int8}()

    for α in _smolyak_multiindices(dim, level)
        q     = sum(α)
        exp_  = (q_max - q)
        coeff = (iseven(exp_) ? 1 : -1) * binomial(dim - 1, q_max - q)
        coeff == 0 && continue

        coeff_sign  = coeff > 0 ? Int8(1) : Int8(-1)
        log_abs_c   = log(abs(coeff))

        # Number of points in this tensor product
        n_pts = prod(length(rules_nodes[αi]) for αi in α)

        # Iterate over tensor-product indices via mixed-radix counting
        ns = [length(rules_nodes[αi]) for αi in α]
        for flat_idx in 0:(n_pts-1)
            # Decompose flat_idx into per-dimension indices
            node_val = Vector{Float64}(undef, dim)
            lw_sum   = log_abs_c
            net_sign = coeff_sign
            rem      = flat_idx
            for d in 1:dim
                ki    = rem % ns[d] + 1
                rem   = rem ÷ ns[d]
                nd    = rules_nodes[α[d]][ki]
                lw_d  = rules_lw[α[d]][ki]
                node_val[d] = nd
                lw_sum     += lw_d
                # weights are always positive (GH weights > 0), sign unaffected
            end
            push!(all_nodes, node_val)
            push!(all_lw,    lw_sum)
            push!(all_signs, net_sign)
        end
    end

    R = length(all_lw)
    nodes_mat = Matrix{Float64}(undef, dim, R)
    for r in 1:R
        nodes_mat[:, r] = all_nodes[r]
    end

    # ── Deduplication ────────────────────────────────────────────────────────
    # Sort indices lexicographically by node coordinate.
    perm = sortperm(1:R, by = r -> nodes_mat[:, r])

    dedup_nodes = Vector{Vector{Float64}}()
    dedup_lw    = Vector{Float64}()
    dedup_signs = Vector{Int8}()

    r = 1
    while r <= R
        # Find the end of the run of identical nodes.
        run_end = r
        while run_end + 1 <= R && nodes_mat[:, perm[run_end+1]] == nodes_mat[:, perm[r]]
            run_end += 1
        end

        # Accumulate the combined signed weight for this node.
        combined_w = 0.0
        for i in r:run_end
            ri = perm[i]
            combined_w += all_signs[ri] * exp(all_lw[ri])
        end

        # Keep only if the combined weight is non-negligible.
        if abs(combined_w) > eps(Float64)
            push!(dedup_nodes, copy(nodes_mat[:, perm[r]]))
            push!(dedup_lw,    log(abs(combined_w)))
            push!(dedup_signs, combined_w > 0.0 ? Int8(1) : Int8(-1))
        end

        r = run_end + 1
    end

    R_new = length(dedup_lw)
    out_mat = Matrix{Float64}(undef, dim, R_new)
    for r in 1:R_new
        out_mat[:, r] = dedup_nodes[r]
    end

    return GHQuadratureNodes{Float64}(dim, level, out_mat, dedup_lw, dedup_signs)
end

# ---------------------------------------------------------------------------
# Global cache
# ---------------------------------------------------------------------------

const _SPARSEGRID_CACHE = Dict{Tuple{Int,Int}, GHQuadratureNodes{Float64}}()

"""
    get_sparse_grid(dim::Int, level::Int) -> GHQuadratureNodes{Float64}

Return the Smolyak sparse-grid for the given `dim` and `level`, building and
caching it on first call. Thread-safe only when all needed `(dim, level)` pairs
are populated before concurrent use (call this for all needed pairs at setup time).
"""
function get_sparse_grid(dim::Int, level::Int)
    key = (dim, level)
    haskey(_SPARSEGRID_CACHE, key) && return _SPARSEGRID_CACHE[key]
    sg = build_sparse_grid(dim, level)
    _SPARSEGRID_CACHE[key] = sg
    return sg
end

"""
    n_ghq_points(dim::Int, level::Int) -> Int

Return the number of quadrature points (after deduplication) in the Smolyak
sparse grid for `dim` dimensions at accuracy `level`.

Useful for checking grid size before fitting:
    n_ghq_points(5, 3)  # after deduplication
"""
function n_ghq_points(dim::Int, level::Int)
    return size(get_sparse_grid(dim, level).nodes, 2)
end

# ---------------------------------------------------------------------------
# Additional 1D quadrature rules
# ---------------------------------------------------------------------------

"""
    _gl_rule(n::Int) -> (nodes::Vector{Float64}, logweights::Vector{Float64})

Compute `n`-point Gauss-Legendre quadrature on [-1, 1] via the Golub-Welsch
(Jacobi matrix eigenvalue) method.

Convention: ∫₋₁¹ f(x) dx ≈ Σ exp(logweights[k]) * f(nodes[k])
Weights sum to 2 (the interval length), so `logweights` are log of absolute
GL weights.

For n=1: single node at 0, logweight = log(2).
"""
function _gl_rule(n::Int)
    if n == 1
        return [0.0], [log(2.0)]
    end
    β = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:n-1]
    J = SymTridiagonal(zeros(n), β)
    F = eigen(J)
    x = F.values                          # GL nodes on (-1, 1)
    w = 2.0 .* F.vectors[1, :] .^ 2      # GL weights, sum to 2
    ord = sortperm(x)
    return x[ord], log.(w[ord])
end

"""
    _cc_rule(n::Int) -> (nodes::Vector{Float64}, logweights::Vector{Float64})

Compute `n+1`-point Clenshaw-Curtis quadrature on [-1, 1].  The nodes are the
`n+1` Chebyshev points of the second kind: `x_j = cos(πj/n)` for `j=0,...,n`.
Returns nodes sorted in ascending order.

Convention: ∫₋₁¹ f(x) dx ≈ Σ exp(logweights[k]) * f(nodes[k])
Weights are positive and sum to 2 (the interval length).

For n=1: two-point rule (endpoints ±1, weight 1 each).

Clenshaw-Curtis nodes at level n are a superset of those at all even sub-levels,
making CC useful for adaptive / nested quadrature (Phase 4).
"""
function _cc_rule(n::Int)
    @assert n >= 1 "_cc_rule: n must be ≥ 1"

    if n == 1
        return [-1.0, 1.0], [log(1.0), log(1.0)]  # two endpoints, w=1 each
    end

    N = n
    # Nodes: x_j = cos(πj/N) for j=0,...,N  (x_0=1, x_N=-1)
    nodes_desc = [cos(π * j / N) for j in 0:N]

    # Weights via the standard explicit formula (Waldvogel 2006):
    #   w_j = (c_j / N) * (1 - Σ_{l=1}^{floor(N/2)} b_l * cos(2πlj/N) / (4l²-1))
    # where c_j = 1 if j∈{0,N}, c_j = 2 otherwise;
    #       b_l = 1 if 2l == N (endpoint of the sum), b_l = 2 otherwise.
    c = [j == 0 || j == N ? 1.0 : 2.0 for j in 0:N]
    half = N ÷ 2
    w = zeros(N + 1)
    for j in 0:N
        s = 0.0
        for l in 1:half
            b_l = (2l == N) ? 1.0 : 2.0
            s += b_l * cos(2π * l * j / N) / (4l^2 - 1)
        end
        w[j+1] = (c[j+1] / N) * (1.0 - s)
    end

    ord = sortperm(nodes_desc)
    return nodes_desc[ord], log.(w[ord])
end

# ---------------------------------------------------------------------------
# Tensor product of sparse grids (for anisotropic quadrature)
# ---------------------------------------------------------------------------

"""
    build_tensor_product_grid(grids) -> GHQuadratureNodes{Float64}

Build the tensor product of two or more sparse-grid rules.  The resulting rule
integrates the same class of functions as the individual rules in independent
subspaces:

    ∫ f(z) N(z₁; 0, I_{d₁}) N(z₂; 0, I_{d₂}) dz₁ dz₂
    ≈ Σ_{r₁,r₂} (w₁_r₁ · w₂_r₂) f([z₁_r₁; z₂_r₂])

The resulting `GHQuadratureNodes.level` is set to `0` (sentinel: product grid,
no single accuracy level).  The `dim` field is the sum of all input dimensions.

Use this to build anisotropic grids with different accuracy levels per
RE group: `build_tensor_product_grid([get_sparse_grid(d₁,l₁), get_sparse_grid(d₂,l₂)])`.
"""
function build_tensor_product_grid(grids::AbstractVector{<:GHQuadratureNodes{T}}) where {T}
    isempty(grids) && error("build_tensor_product_grid: at least one grid required")
    length(grids) == 1 && return grids[1]
    result = grids[1]
    for k in 2:length(grids)
        result = _tensor_product_two(result, grids[k])
    end
    return result
end

function _tensor_product_two(sg1::GHQuadratureNodes{T}, sg2::GHQuadratureNodes{T}) where {T}
    R1 = size(sg1.nodes, 2)
    R2 = size(sg2.nodes, 2)
    R  = R1 * R2
    d1 = sg1.dim
    d2 = sg2.dim
    d  = d1 + d2

    nodes_out      = Matrix{T}(undef, d, R)
    logweights_out = Vector{T}(undef, R)
    signs_out      = Vector{Int8}(undef, R)

    r = 0
    @inbounds for r2 in 1:R2, r1 in 1:R1
        r += 1
        nodes_out[1:d1, r]       = sg1.nodes[:, r1]
        nodes_out[d1+1:d, r]     = sg2.nodes[:, r2]
        logweights_out[r]        = sg1.logweights[r1] + sg2.logweights[r2]
        signs_out[r]             = sg1.signs[r1] * sg2.signs[r2]
    end

    # level=0 is the sentinel for "product grid" (no single Smolyak level)
    return GHQuadratureNodes{T}(d, 0, nodes_out, logweights_out, signs_out)
end

# ---------------------------------------------------------------------------
# Anisotropic grid cache
# ---------------------------------------------------------------------------

# Key: (dims, levels) — dims[k] = RE-group dimension, levels[k] = GH level for that group
const _ANISOTROPIC_CACHE = Dict{Tuple{Vector{Int}, Vector{Int}}, GHQuadratureNodes{Float64}}()

"""
    get_anisotropic_grid(dims::Vector{Int}, levels::Vector{Int}) -> GHQuadratureNodes{Float64}

Return (building and caching on first call) the tensor-product sparse grid
for a batch with multiple RE groups, each with its own dimension and accuracy
level.

- `dims[k]`   : dimension (number of free RE parameters) of the k-th RE group
- `levels[k]` : Smolyak accuracy level for the k-th RE group

The returned grid is the tensor product of `get_sparse_grid(dims[k], levels[k])`
for k = 1,...,length(dims).  Call this for all needed `(dims, levels)` pairs
at setup time before parallel use.
"""
function get_anisotropic_grid(dims::Vector{Int}, levels::Vector{Int})
    @assert length(dims) == length(levels) >= 1
    key = (copy(dims), copy(levels))
    haskey(_ANISOTROPIC_CACHE, key) && return _ANISOTROPIC_CACHE[key]
    grids = [get_sparse_grid(d, l) for (d, l) in zip(dims, levels)]
    sg = build_tensor_product_grid(grids)
    _ANISOTROPIC_CACHE[key] = sg
    return sg
end

"""
    n_anisotropic_grid_points(dims::Vector{Int}, levels::Vector{Int}) -> Int

Return the total number of quadrature points in the tensor-product anisotropic
grid: equal to `prod(n_ghq_points(dims[k], levels[k]) for k)`.

Useful for checking grid sizes before fitting with anisotropic levels.
"""
function n_anisotropic_grid_points(dims::Vector{Int}, levels::Vector{Int})
    return size(get_anisotropic_grid(dims, levels).nodes, 2)
end
