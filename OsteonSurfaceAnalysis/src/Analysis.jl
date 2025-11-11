module Analysis

using LinearAlgebra
using Statistics

export analysis_Tdelay_pairs, compute_curvature, compute_curvature_4th

function generate_Tdelay_pairs(proj_points)
    Tdelay_proj_point_pairs = []
    for cont in eachindex(proj_points)[1:end-1]
        Tdelay_proj_point_pairs_per_cont = []
        for ang in eachindex(proj_points[1])
            push!(Tdelay_proj_point_pairs_per_cont, [proj_points[cont][ang][1], proj_points[cont+1][ang][2]])
        end
        push!(Tdelay_proj_point_pairs, Tdelay_proj_point_pairs_per_cont)
    end
    return Tdelay_proj_point_pairs
end

function compute_Tdelay_gradients(Tdelay_proj_points)
    line_∇ = []
    for cont in eachindex(Tdelay_proj_points)
        line_∇_per_cont = []
        for ang in eachindex(Tdelay_proj_points[1])
            p1 = Tdelay_proj_points[cont][ang][1]
            p2 = Tdelay_proj_points[cont][ang][2]
            ∇ = (p2[2] - p1[2]) / (p2[1] - p1[1])
            push!(line_∇_per_cont, ∇)
        end
        push!(line_∇, line_∇_per_cont)
    end
    return line_∇
end

function angle_between_vectors(v1, v2)
    cosθ = dot(v1, v2) / (norm(v1) * norm(v2))
    return acos(clamp(cosθ, -1.0, 1.0))
end

function analysis_Tdelay_pairs(proj_points)
    Tdelay_proj_point_pairs = generate_Tdelay_pairs(proj_points)
    Tdelay_line_∇ = compute_Tdelay_gradients(Tdelay_proj_point_pairs)
    # compute α here
    α = zeros(size(Tdelay_proj_point_pairs,1), size(Tdelay_proj_point_pairs[1],1))
    for cont in axes(α,1)
        for xy_ang in axes(α,2)
            v1 = Tdelay_proj_point_pairs[cont][xy_ang][1] .- Tdelay_proj_point_pairs[cont][xy_ang][2]
            v2 = [0.0, v1[2]]
            α_val = angle_between_vectors(v1, v2)
            if Tdelay_line_∇[cont][xy_ang] < 0.0
                α[cont,xy_ang] = -rad2deg(α_val)
            else
                α[cont,xy_ang] = rad2deg(α_val)
            end
        end
    end
    return Tdelay_proj_point_pairs, Tdelay_line_∇, α
end

"""
    kappa = compute_curvature(ϕ, dx, dy, dz; eps=1e-12)

Compute mean curvature κ of the level set ϕ(x,y,z) on a regular 3D grid,
using 2nd-order central differences.

κ = ( ϕx^2 ϕyy - 2 ϕx ϕy ϕxy + ϕy^2 ϕxx
    + ϕx^2 ϕzz - 2 ϕx ϕz ϕxz + ϕz^2 φxx
    + ϕy^2 ϕzz - 2 ϕy ϕz ϕyz + ϕz^2 φyy ) / |∇φ|^3

This method of calculating curvature comes from Oscher 2003 Eq(1.8). 
The spacing (dx,dy,dz) are the grid steps in x,y,z.
"""
function compute_curvature(ϕ::AbstractArray{<:Real,3},
                           dx::Real, dy::Real, dz::Real; eps=1e-12)

    nx, ny, nz = size(ϕ)
    kappa = fill!(similar(ϕ, Float64), 0.0)

    inv2dx = 1.0/(2*dx); inv2dy = 1.0/(2*dy); inv2dz = 1.0/(2*dz)
    invdx2 = 1.0/(dx*dx); invdy2 = 1.0/(dy*dy); invdz2 = 1.0/(dz*dz)
    inv4dxdy = 1.0/(4*dx*dy); inv4dxdz = 1.0/(4*dx*dz); inv4dydz = 1.0/(4*dy*dz)

    @inbounds for k in 2:nz-1, j in 2:ny-1, i in 2:nx-1
        ϕc = ϕ[i, j, k]

        # First derivatives (central)
        ϕx = (ϕ[i+1, j,   k  ] - ϕ[i-1, j,   k  ]) * inv2dx
        ϕy = (ϕ[i,   j+1, k  ] - ϕ[i,   j-1, k  ]) * inv2dy
        ϕz = (ϕ[i,   j,   k+1] - ϕ[i,   j,   k-1]) * inv2dz

        # Second derivatives (central)
        ϕxx = (ϕ[i+1, j,   k  ] - 2ϕc + ϕ[i-1, j,   k  ]) * invdx2
        ϕyy = (ϕ[i,   j+1, k  ] - 2ϕc + ϕ[i,   j-1, k  ]) * invdy2
        ϕzz = (ϕ[i,   j,   k+1] - 2ϕc + ϕ[i,   j,   k-1]) * invdz2

        # Mixed derivatives (central)
        #ϕxy = (ϕ[i+1, j+1, k] - ϕ[i+1, j-1, k] - ϕ[i-1, j+1, k] + ϕ[i-1, j-1, k]) * inv4dxdy
        #ϕxz = (ϕ[i+1, j, k+1] - ϕ[i+1, j, k-1] - ϕ[i-1, j, k+1] + ϕ[i-1, j, k-1]) * inv4dxdz
        #ϕyz = (ϕ[i, j+1, k+1] - ϕ[i, j+1, k-1] - ϕ[i, j-1, k+1] + ϕ[i, j-1, k-1]) * inv4dydz
        ϕxy = (ϕx[i,   j+1, k  ] - ϕx[i,   j-1, k  ]) * inv2dy
        ϕxz = (ϕx[i,   j,   k+1] - ϕx[i,   j,   k-1]) * inv2dz
        ϕyz = (ϕy[i,   j,   k+1] - ϕy[i,   j,   k-1]) * inv2dz


        # |∇ϕ|
        gradmag = sqrt(ϕx*ϕx + ϕy*ϕy + ϕz*ϕz) + eps  # eps avoids divide-by-zero
        denom = gradmag^3

        # Numerator
        num  =  (ϕx^2)*ϕyy - 2*ϕx*ϕy*ϕxy + (ϕy^2)*ϕxx
        num +=  (ϕx^2)*ϕzz - 2*ϕx*ϕz*ϕxz + (ϕz^2)*ϕxx
        num +=  (ϕy^2)*ϕzz - 2*ϕy*ϕz*ϕyz + (ϕz^2)*ϕyy

        kappa[i, j, k] = num / denom
    end

    return kappa
end


"""
    κ = compute_curvature_4th(ϕ, dx, dy, dz; eps=1e-12)

Compute the level-set mean curvature κ of ϕ(x,y,z) on a rectangular 3D grid with spacings dx, dy, and dz,
using 4th-order central differences on the interior
and 2nd-order central differences on a 2-cell boundary band.

From Oscher 2003, the formula is given by κ = ∇ϕ / ||∇ϕ|| which expands to:
κ = ( ϕx^2 ϕyy - 2 ϕx ϕy ϕxy + ϕy^2 ϕxx
    + ϕx^2 ϕzz - 2 ϕx ϕz ϕxz + ϕz^2 ϕxx
    + ϕy^2 ϕzz - 2 ϕy ϕz ϕyz + ϕz^2 ϕyy ) / |∇ϕ|^3

`eps` is added in to avoid division by zero.
"""
function compute_curvature_4th(ϕ::AbstractArray{<:Real,3},
                               dx::Real, dy::Real, dz::Real; eps=1e-12)
    nx, ny, nz = size(ϕ)
    kappa = fill!(similar(ϕ, Float64), 0.0)

    nx ≥ 5 && ny ≥ 5 && nz ≥ 5 || error("Need at least 5 points in each dim for 4th-order stencils.")

    inv12dx  = 1.0/(12*dx);  inv12dy  = 1.0/(12*dy);  inv12dz  = 1.0/(12*dz)
    inv12dx2 = 1.0/(12*dx*dx); inv12dy2 = 1.0/(12*dy*dy); inv12dz2 = 1.0/(12*dz*dz)
    inv2dx   = 1.0/(2*dx);   inv2dy   = 1.0/(2*dy);   inv2dz   = 1.0/(2*dz)
    invdx2   = 1.0/(dx*dx);  invdy2   = 1.0/(dy*dy);  invdz2   = 1.0/(dz*dz)
    inv4dxdy = 1.0/(4*dx*dy); inv4dxdz = 1.0/(4*dx*dz); inv4dydz = 1.0/(4*dy*dz)

    # ---- 1D 4th-order stencils at a single index (central, needs ±1,±2) ----
    Dx4(i,j,k)  = (-ϕ[i+2,j,k] + 8ϕ[i+1,j,k] - 8ϕ[i-1,j,k] + ϕ[i-2,j,k]) * inv12dx
    Dy4(i,j,k)  = (-ϕ[i,j+2,k] + 8ϕ[i,j+1,k] - 8ϕ[i,j-1,k] + ϕ[i,j-2,k]) * inv12dy
    Dz4(i,j,k)  = (-ϕ[i,j,k+2] + 8ϕ[i,j,k+1] - 8ϕ[i,j,k-1] + ϕ[i,j,k-2]) * inv12dz

    Dxx4(i,j,k) = (-ϕ[i+2,j,k] + 16ϕ[i+1,j,k] - 30ϕ[i,j,k] + 16ϕ[i-1,j,k] - ϕ[i-2,j,k]) * inv12dx2
    Dyy4(i,j,k) = (-ϕ[i,j+2,k] + 16ϕ[i,j+1,k] - 30ϕ[i,j,k] + 16ϕ[i,j-1,k] - ϕ[i,j-2,k]) * inv12dy2
    Dzz4(i,j,k) = (-ϕ[i,j,k+2] + 16ϕ[i,j,k+1] - 30ϕ[i,j,k] + 16ϕ[i,j,k-1] - ϕ[i,j,k-2]) * inv12dz2

    # Mixed derivatives via composition of 4th-order 1D operators (still 4th-order):
    # e.g. ϕ_xy(i,j,k) = D4x( Dy4(ϕ)(·,j,k) ) at i.
    function Dxy4(i,j,k)
        g_im2 = Dy4(i-2,j,k); g_im1 = Dy4(i-1,j,k); g_ip1 = Dy4(i+1,j,k); g_ip2 = Dy4(i+2,j,k)
        (-g_ip2 + 8g_ip1 - 8g_im1 + g_im2) * inv12dx
    end
    function Dxz4(i,j,k)
        g_im2 = Dz4(i-2,j,k); g_im1 = Dz4(i-1,j,k); g_ip1 = Dz4(i+1,j,k); g_ip2 = Dz4(i+2,j,k)
        (-g_ip2 + 8g_ip1 - 8g_im1 + g_im2) * inv12dx
    end
    function Dyz4(i,j,k)
        g_jm2 = Dz4(i,j-2,k); g_jm1 = Dz4(i,j-1,k); g_jp1 = Dz4(i,j+1,k); g_jp2 = Dz4(i,j+2,k)
        (-g_jp2 + 8g_jp1 - 8g_jm1 + g_jm2) * inv12dy
    end

    # ===================== 4th-order interior =====================
    @inbounds for k in 3:nz-2, j in 3:ny-2, i in 3:nx-2
        ϕx, ϕy, ϕz = Dx4(i,j,k), Dy4(i,j,k), Dz4(i,j,k)
        ϕxx, ϕyy, ϕzz = Dxx4(i,j,k), Dyy4(i,j,k), Dzz4(i,j,k)
        ϕxy, ϕxz, ϕyz = Dxy4(i,j,k), Dxz4(i,j,k), Dyz4(i,j,k)

        gradmag = sqrt(ϕx*ϕx + ϕy*ϕy + ϕz*ϕz) + eps
        denom = gradmag^3

        num  =  (ϕx^2)*ϕyy - 2*ϕx*ϕy*ϕxy + (ϕy^2)*ϕxx
        num +=  (ϕx^2)*ϕzz - 2*ϕx*ϕz*ϕxz + (ϕz^2)*ϕxx
        num +=  (ϕy^2)*ϕzz - 2*ϕy*ϕz*ϕyz + (ϕz^2)*ϕyy

        kappa[i,j,k] = num / denom
    end

    # ===================== 2nd-order boundary band =====================
    # Use your original 2nd-order stencils on i/j/k ∈ {2, nx-1} etc., and also first/last layer.
    @inbounds begin
        inv2dx = 1.0/(2*dx); inv2dy = 1.0/(2*dy); inv2dz = 1.0/(2*dz)
        invdx2 = 1.0/(dx*dx); invdy2 = 1.0/(dy*dy); invdz2 = 1.0/(dz*dz)
        inv4dxdy = 1.0/(4*dx*dy); inv4dxdz = 1.0/(4*dx*dz); inv4dydz = 1.0/(4*dy*dz)

        # Helper loop that guards against out-of-bounds and fills any index not done above
        function fill_second_order!(i,j,k)
            if 3 ≤ i ≤ nx-2 && 3 ≤ j ≤ ny-2 && 3 ≤ k ≤ nz-2
                return  # already 4th-order
            end
            2 ≤ i ≤ nx-1 && 2 ≤ j ≤ ny-1 && 2 ≤ k ≤ nz-1 || return  # need neighbors

            ϕc = ϕ[i,j,k]
            ϕx = (ϕ[i+1,j,k] - ϕ[i-1,j,k]) * inv2dx
            ϕy = (ϕ[i,j+1,k] - ϕ[i,j-1,k]) * inv2dy
            ϕz = (ϕ[i,j,k+1] - ϕ[i,j,k-1]) * inv2dz

            ϕxx = (ϕ[i+1,j,k] - 2ϕc + ϕ[i-1,j,k]) * invdx2
            ϕyy = (ϕ[i,j+1,k] - 2ϕc + ϕ[i,j-1,k]) * invdy2
            ϕzz = (ϕ[i,j,k+1] - 2ϕc + ϕ[i,j,k-1]) * invdz2

            ϕxy = (ϕ[i+1,j+1,k] - ϕ[i+1,j-1,k] - ϕ[i-1,j+1,k] + ϕ[i-1,j-1,k]) * inv4dxdy
            ϕxz = (ϕ[i+1,j,k+1] - ϕ[i+1,j,k-1] - ϕ[i-1,j,k+1] + ϕ[i-1,j,k-1]) * inv4dxdz
            ϕyz = (ϕ[i,j+1,k+1] - ϕ[i,j+1,k-1] - ϕ[i,j-1,k+1] + ϕ[i,j-1,k-1]) * inv4dydz

            gradmag = sqrt(ϕx*ϕx + ϕy*ϕy + ϕz*ϕz) + eps
            denom = gradmag^3

            num  =  (ϕx^2)*ϕyy - 2*ϕx*ϕy*ϕxy + (ϕy^2)*ϕxx
            num +=  (ϕx^2)*ϕzz - 2*ϕx*ϕz*ϕxz + (ϕz^2)*ϕxx
            num +=  (ϕy^2)*ϕzz - 2*ϕy*ϕz*ϕyz + (ϕz^2)*ϕyy

            kappa[i,j,k] = num / denom
        end

        # All points that have the 2nd-order neighborhood
        for k in 2:nz-1, j in 2:ny-1, i in 2:nx-1
            fill_second_order!(i,j,k)
        end
    end

    return kappa
end

"""
    ensure_ccw(X, Y)

Takes coordinate vectors `X` and `Y` representing a closed 2D curve and
returns `(X2, Y2)` oriented **counterclockwise** (anti-clockwise).

If the points are already CCW, they are returned unchanged.
If they are clockwise, they are reversed.

Returns
-------
`(X2, Y2, flipped)` where `flipped::Bool` is `true` if the order was reversed.
"""
function ensure_ccw(X::AbstractVector, Y::AbstractVector)
    @assert length(X) == length(Y) "X and Y must be same length"
    N = length(X)
    @assert N ≥ 3 "Need at least 3 points"

    # Compute signed area (shoelace formula)
    A2 = 0.0
    @inbounds for i in 1:N
        j = (i == N) ? 1 : i + 1
        A2 += X[i] * Y[j] - X[j] * Y[i]
    end

    if A2 > 0     # already CCW
        return (X, Y, false)
    else           # CW: reverse
        return (reverse(X), reverse(Y), true)
    end
end

"""
    κ = local_curvature(x, y; k=5, method=:constrained, sigma=nothing,
                        weights=:gaussian, signed=true)

Estimate local curvature κ at each vertex of a **closed** 2D polyline given by
`(x[i], y[i])`, using a least-squares quadratic fit in a tangent-aligned frame.

Arguments
---------
- `x, y` : Vectors (same length N) of coordinates.

Keyword args
-----------
- `k::Int=5` : # of neighbors on each side to include in the local fit.
- `method::Symbol=:constrained` :
      `:constrained`  → fit v ≈ a u^2   (enforces v(0)=v'(0)=0) ⇒ κ = 2a
      `:unconstrained`→ fit v ≈ a u^2 + b u + c               ⇒ κ = 2a/(1+b^2)^(3/2)
- `sigma` : bandwidth for Gaussian weights in u. If `nothing`, choose
            `sigma = 2*std(u)` per window (fallback to max(|u|) if degenerate).
- `weights::Symbol=:gaussian` : `:gaussian` or `:uniform`.
- `signed::Bool=true` : if `true`, keep curvature sign (left turn positive
                        under the “left-normal is +v” convention).

Returns
-------
- `κ::Vector{Float64}` : curvature at each vertex (NaN where fit is ill-conditioned).

Notes
-----
- Indices wrap around (closed curve).
- If you have very noisy data, consider increasing `k`.
- For sharp corners, any quadratic will smear the peak; treat separately if needed.
"""
function local_curvature(x::AbstractVector, y::AbstractVector;
                         k::Int=5, method::Symbol=:constrained,
                         sigma=nothing, weights::Symbol=:none,
                         signed::Bool=true)

    @assert length(x) == length(y) "x and y must have same length"
    N = length(x)
    @assert N ≥ 5 "Need at least 5 points"

    # FIX: wrap using mod1 (1..N)
    wrap(i) = mod1(i, N)

    function tangent(i)
        ip = wrap(i+1); im = wrap(i-1)
        tx = x[ip] - x[im]
        ty = y[ip] - y[im]
        n = hypot(tx, ty)
        return n == 0 ? (1.0, 0.0) : (tx/n, ty/n)
    end

    function to_local(i, j, tx, ty)
        dx = x[j] - x[i]
        dy = y[j] - y[i]
        u =  tx*dx + ty*dy
        v = -ty*dx + tx*dy
        return (u, v)
    end

    x,y = ensure_ccw(x, y)

    κ = fill(NaN, N)

    for i in 1:N
        left  = [wrap(i - s) for s in k:-1:1]
        right = [wrap(i + s) for s in 1:k]
        neigh = vcat(left, right)

        tx, ty = tangent(i)

        U = Vector{Float64}(undef, 2k)
        V = Vector{Float64}(undef, 2k)
        @inbounds for (m, j) in enumerate(neigh)
            u, v = to_local(i, j, tx, ty)
            U[m] = u; V[m] = v
        end

        w = ones(2k)
        if weights == :gaussian
            σ = isnothing(sigma) ? (2*std(U)) : float(sigma)
            if !(σ > 0)
                σ = maximum(abs, U); σ = (σ > 0) ? σ : 1.0
            end
            @inbounds for m in 1:2k
                w[m] = exp(-(U[m]^2)/(σ^2))
            end
        end

        if method == :constrained
            num = 0.0; den = 0.0
            @inbounds for m in 1:2k
                u = U[m]; v = V[m]; wm = w[m]; u2 = u*u
                num += wm * u2 * v
                den += wm * u2 * u2
            end
            κ[i] = den > eps() ? 2 * (num/den) : NaN

        elseif method == :unconstrained
            A = ones(2k, 3)
            @inbounds for m in 1:2k
                u = U[m]
                A[m,1] = u*u
                A[m,2] = u
            end
            sw = sqrt.(w)
            coeff = (A .* sw) \ (V .* sw)
            a, b, _ = coeff
            κ[i] = 2a / (1 + b^2)^(3/2)
        else
            error("Unknown method: $method")
        end

        if !signed
            κ[i] = abs(κ[i])
        end
    end

    return κ
end

end # end of module