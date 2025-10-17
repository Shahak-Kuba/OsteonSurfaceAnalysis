module Analysis

using LinearAlgebra

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
        ϕxy = (ϕ[i+1, j+1, k] - ϕ[i+1, j-1, k] - ϕ[i-1, j+1, k] + ϕ[i-1, j-1, k]) * inv4dxdy
        ϕxz = (ϕ[i+1, j, k+1] - ϕ[i+1, j, k-1] - ϕ[i-1, j, k+1] + ϕ[i-1, j, k-1]) * inv4dxdz
        ϕyz = (ϕ[i, j+1, k+1] - ϕ[i, j+1, k-1] - ϕ[i, j-1, k+1] + ϕ[i, j-1, k-1]) * inv4dydz

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
    kappa = compute_curvature_4th(ϕ, dx, dy, dz; eps=1e-12)

Compute the level-set mean curvature κ of ϕ(x,y,z) on a regular 3D grid,
using **4th-order central differences** on the interior (i=3..nx-2, etc.)
and **2nd-order central differences** on a 2-cell boundary band.

The formula is (Osher/Sethian-style):
κ = ( ϕx^2 ϕyy - 2 ϕx ϕy ϕxy + ϕy^2 ϕxx
    + ϕx^2 ϕzz - 2 ϕx ϕz ϕxz + ϕz^2 ϕxx
    + ϕy^2 ϕzz - 2 ϕy ϕz ϕyz + ϕz^2 ϕyy ) / |∇ϕ|^3

`eps` avoids division by zero.
Requires at least 5 grid points along each axis for the 4th-order interior.
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

end # end of module