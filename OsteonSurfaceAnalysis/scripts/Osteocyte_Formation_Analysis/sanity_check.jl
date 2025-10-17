using Pkg
Pkg.add(["PythonCall","CondaPkg"])

using CondaPkg
CondaPkg.add(["python","numpy","scipy"])       # installs numpy too

Pkg.build("PythonCall")
# restart Julia here (important)

using PythonCall
nd = pyimport("scipy.ndimage")
nd.distance_transform_edt([0 1; 1 1])  # quick smoke test

# cache the scipy.ndimage module safely

"""
Euclidean distance (µm) to foreground (mask==true), with anisotropic spacing.
Converts BitArray to dense Bool array to avoid PythonCall issues.
"""
function edt(mask::AbstractArray{Bool,3}; dx=0.379, dy=0.379, dz=0.4)
    A = Array{Bool}(mask)              # avoid BitArray conversions
    D = nd.distance_transform_edt(.!A; sampling=(dx,dy,dz))
    return Array(D)
end

"""
Signed EDT (µm): positive outside, negative inside.
"""
function edt_S(mask::AbstractArray{Bool,3}; dx=0.379, dy=0.379, dz=0.4)
    A = Array{Bool}(mask)
    din  = nd.distance_transform_edt(A;   sampling=(dx,dy,dz))  # dist to outside
    dout = nd.distance_transform_edt(.!A; sampling=(dx,dy,dz))  # dist to inside
    return dout .- din
end

ϕ_func = (t,S_DTʰ,S_DTᶜ) -> (1-t) .* S_DTʰ - (t) .* S_DTᶜ

function compute_ϕ_stack_3D_py(outer, inner, tvals)
    H, W, Z = size(outer)
    ϕ = zeros(Float32, H, W, Z, length(tvals))

    outer_dt_S = pyconvert(Array{Float32, 3},edt_S(PythonCall.PyArray(outer)))
    inner_dt_S = pyconvert(Array{Float32, 3},edt_S(PythonCall.PyArray(inner)))

    for (ti, t) in enumerate(tvals)
        ϕ[:,:,:,ti] .= ϕ_func(t, outer_dt_S, inner_dt_S)
    end
    return ϕ
end

test = compute_ϕ_stack_3D_py(a_outer, a_inner, tvals)

ϕ = test