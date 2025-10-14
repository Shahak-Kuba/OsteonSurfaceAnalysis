module Analysis

using LinearAlgebra

export analysis_Tdelay_pairs

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
            v1 = Tdelay_proj_point_pairs[1][1][1] .- Tdelay_proj_point_pairs[1][1][2]
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

end # end of module