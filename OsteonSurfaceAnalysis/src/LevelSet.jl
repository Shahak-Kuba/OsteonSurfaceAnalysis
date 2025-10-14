module LevelSet

export edt, edt_S, ϕ_func, compute_ϕ_at_t, compute_ϕ_stack

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
# TBD

# --------------------------- finding the 0 level contour ------------------------------


end # end of module