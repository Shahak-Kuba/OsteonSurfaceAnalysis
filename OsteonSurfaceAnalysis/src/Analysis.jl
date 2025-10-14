module Analysis

using LinearAlgebra

export analysis_Tdelay_pairs, compute_curvature

function generate_Tdelay_pairs(proj_points_right, proj_points_left, tvals)
    Tdelay_proj_points_right = []
    Tdelay_proj_points_left = []
    reshaped_proj_points_right = reshape(proj_points_right, (length(tvals),Int(length(proj_points_right)/length(tvals))))
    reshaped_proj_points_left = reshape(proj_points_left, (length(tvals),Int(length(proj_points_right)/length(tvals))))

    for jj in axes(reshaped_proj_points_left,2)
        for ii in axes(reshaped_proj_points_left,1)[1:end-1]
            push!(Tdelay_proj_points_right, [reshaped_proj_points_right[ii,jj][1], reshaped_proj_points_right[ii+1,jj][2]])
            push!(Tdelay_proj_points_left, [reshaped_proj_points_left[ii,jj][1], reshaped_proj_points_left[ii+1,jj][2]])
        end
    end

    return Tdelay_proj_points_right, Tdelay_proj_points_left
end

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

function compute_Tdelay_gradients(Tdelay_proj_points_right, Tdelay_proj_points_left)
    # calculating line gradients
    line_∇_right = []
    line_∇_left = []
    for ii in eachindex(Tdelay_proj_points_right)
        xy1_right = Tdelay_proj_points_right[ii][1]
        xy2_right = Tdelay_proj_points_right[ii][2]

        xy1_left = Tdelay_proj_points_left[ii][1]
        xy2_left = Tdelay_proj_points_left[ii][2]

        grad_right = (xy2_right[2] - xy1_right[2]) / (xy2_right[1] - xy1_right[1]) 
        grad_left = (xy2_left[2] - xy1_left[2]) / (xy2_left[1] - xy1_left[1]) 

        push!(line_∇_right, grad_right)
        push!(line_∇_left, grad_left)
    end

    return line_∇_right, line_∇_left
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

function analysis_Tdelay_pairs(proj_points_right, proj_points_left, tvals, intersecting_points)
    Tdelay_proj_points_right, Tdelay_proj_points_left = generate_Tdelay_pairs(proj_points_right, proj_points_left, tvals);
    line_∇_right, line_∇_left = compute_Tdelay_gradients(Tdelay_proj_points_right, Tdelay_proj_points_left)
    # caclulate angle w.r.t vertical axis
    α = zeros(size(intersecting_points,1)-1, size(intersecting_points[1],1))
    # calculating vertical angle TO BE FIXED
    #for ii in axes(α,1)
    #    for jj in 1:Int(size(α,2)/2)
    #        idx = (jj-1) * size(α,1) + ii 
    #        pt1 = proj_points_right[idx][1] .- proj_points_right[idx][2]
    #        pt2 = proj_points_left[idx][1] .- proj_points_left[idx][2]
    #        α[ii,jj] = rad2deg(atan(pt1[1], pt1[2]) )
    #        α[ii,Int(size(α,2)/2) + jj] = -rad2deg(atan(pt2[1], pt2[2]))
    #    end
    #end
    return Tdelay_proj_points_right, Tdelay_proj_points_left, line_∇_right, line_∇_left, α
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

end # end of module