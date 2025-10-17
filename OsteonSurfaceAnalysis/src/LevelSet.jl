module LevelSet

export edt, edt_S, ϕ_func, compute_ϕ_at_t, compute_ϕ_stack, compute_ϕ_at_t_3D, compute_ϕ_stack_3D, estimate_Ocy_formation_time

using DistanceTransforms
# EDT & signed DT, ϕ over time
function edt(mask::BitArray)
    return sqrt.(DistanceTransforms.transform(boolean_indicator(mask)))
end

function edt_S(mask::BitArray)
    return edt(mask) .- edt(.!mask)
end


# Linear interpolation between outer and inner signed distance functions
ϕ_func = (t,S_DTʰ,S_DTᶜ) -> (1-t) .* S_DTʰ - (t) .* S_DTᶜ

# --------------------------- 2D Implementation of level set function computation ------------------------------
# computer level set function at a single time value
function compute_ϕ_at_t(outer, inner, tval::Float64)
    H, W, Z = size(outer)
    ϕ = zeros(Float32, H, W, Z)
    outer_dt_S = similar(ϕ, Float32, H, W, Z)
    inner_dt_S = similar(ϕ, Float32, H, W, Z)
    for z in 1:Z
        outer_dt_S[:,:,z] .= edt_S(outer[:,:,z])
        inner_dt_S[:,:,z] .= edt_S(inner[:,:,z])
        ϕ[:,:,z] .= ϕ_func(tval, outer_dt_S[:,:,z], inner_dt_S[:,:,z])
    end
    return ϕ
end

# computer level set function for multiple time values
function compute_ϕ_stack(outer, inner, tvals)
    H, W, Z = size(outer)
    ϕ = zeros(Float32, H, W, Z, length(tvals))

    outer_dt_S = similar(ϕ, Float32, H, W, Z)
    inner_dt_S = similar(ϕ, Float32, H, W, Z)

    for z in 1:Z
        outer_dt_S[:,:,z] .= edt_S(outer[:,:,z])
        inner_dt_S[:,:,z] .= edt_S(inner[:,:,z])
    end

    for (ti, t) in enumerate(tvals)
        for z in 1:Z
            ϕ[:,:,z,ti] .= ϕ_func(t, outer_dt_S[:,:,z], inner_dt_S[:,:,z])
        end
    end
    return ϕ
end

# --------------------------- 3D Implementation of level set function computation ------------------------------
function compute_ϕ_at_t_3D(outer, inner, tval::Float64)
    outer_dt_S = edt_S(outer)
    inner_dt_S = edt_S(inner)
    return ϕ_func(tval, outer_dt_S, inner_dt_S)
end

# computer level set function for multiple time values
function compute_ϕ_stack_3D(outer, inner, tvals)
    H, W, Z = size(outer)
    ϕ = zeros(Float32, H, W, Z, length(tvals))

    outer_dt_S = edt_S(outer)
    inner_dt_S = edt_S(inner)

    for (ti, t) in enumerate(tvals)
        ϕ[:,:,:,ti] .= ϕ_func(t, outer_dt_S, inner_dt_S)
    end
    return ϕ
end

function estimate_Ocy_formation_time(outer, inner, Ocy_pos_voxel)
    # function to find closest (x,y,z) coord in HC_DT_S gridspace to Ocy_pos
    t_form = zeros(size(Ocy_pos_voxel))
    CL_DT_S = LevelSet.edt_S(outer)
    HC_DT_S = LevelSet.edt_S(inner)
    for ii in eachindex(t_form)
        x = Ocy_pos_voxel[ii][1]; y = Ocy_pos_voxel[ii][2]; z = Ocy_pos_voxel[ii][3];
        t_form[ii] = HC_DT_S[x,y,z] / (HC_DT_S[x,y,z] + CL_DT_S[x,y,z])
    end
    return t_form
end

# --------------------------- running it with python for unequal dx,dy,dz values ------------------------------



end # end of module