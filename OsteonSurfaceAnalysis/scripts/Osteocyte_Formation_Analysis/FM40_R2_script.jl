using OsteonSurfaceAnalysis, FileIO, GLMakie, Statistics
processed_paths = readdir("./DATA/FM40-1-R2/Processed_Images/"; join=true)

paths = processed_paths
using CSV, DataFrames
Ocy_data = CSV.read("./DATA/FM40-1-R2/cells_FM40-1-R2_.csv", DataFrame);
Ocy_data = DataFrame(permutedims(Matrix(Ocy_data)), :auto);
# getting osteocyte (x,y,z) positions
Ocy_pos = []
Ocy_pos_voxel = []
for idx in eachindex(Ocy_data.x3)[2:end]
    x = parse(Float64, Ocy_data.x3[idx])
    y = parse(Float64, Ocy_data.x4[idx])
    z = parse(Float64, Ocy_data.x5[idx])
    println("x: ", x, " y: ", y, " z: ", z)
    push!(Ocy_pos,(x,y,z))
    push!(Ocy_pos_voxel,(Int64(round(x/0.379)),Int64(round(y/0.379)),Int64(round(z/0.4))))
end


downsample = 1
dx = 0.379; dy = 0.379; dz = 0.4;
Δz = dz;
ΔT = 0.2;

using CondaPkg
using PythonCall
nd = pyimport("scipy.ndimage")

"""
Euclidean distance (µm) to foreground (mask==true), with anisotropic spacing.
Converts BitArray to dense Bool array to avoid PythonCall issues.
"""
function edt_py(mask::AbstractArray{Bool,3}; dx=0.379, dy=0.379, dz=0.4)
    A = Array{Bool}(mask)              # avoid BitArray conversions
    D = nd.distance_transform_edt(.!A; sampling=(dx,dy,dz))
    return Array(D)
end

"""
Signed EDT (µm): positive outside, negative inside.
"""
function edt_S_py(mask::AbstractArray{Bool,3}; dx=0.379, dy=0.379, dz=0.4)
    A = Array{Bool}(mask)
    din  = nd.distance_transform_edt(A;   sampling=(dx,dy,dz))  # dist to outside
    dout = nd.distance_transform_edt(.!A; sampling=(dx,dy,dz))  # dist to inside
    return dout .- din
end

ϕ_func = (t,S_DTʰ,S_DTᶜ) -> (1-t) .* S_DTʰ - (t) .* S_DTᶜ

function compute_ϕ_at_t_3D_py(outer, inner, tval::Float64)
    outer_dt_S = pyconvert(Array{Float32, 3},edt_S_py(PythonCall.PyArray(outer)))
    inner_dt_S = pyconvert(Array{Float32, 3},edt_S_py(PythonCall.PyArray(inner)))
    return ϕ_func(tval, outer_dt_S, inner_dt_S)
end

function compute_ϕ_stack_3D_py(outer, inner, tvals)
    H, W, Z = size(outer)
    ϕ = zeros(Float32, H, W, Z, length(tvals))

    outer_dt_S = pyconvert(Array{Float32, 3},edt_S_py(PythonCall.PyArray(outer)))
    inner_dt_S = pyconvert(Array{Float32, 3},edt_S_py(PythonCall.PyArray(inner)))

    for (ti, t) in enumerate(tvals)
        ϕ[:,:,:,ti] .= ϕ_func(t, outer_dt_S, inner_dt_S)
    end
    return ϕ
end

function estimate_Ocy_formation_time(outer, inner, Ocy_pos_voxel)
    # function to find closest (x,y,z) coord in HC_DT_S gridspace to Ocy_pos
    t_form = zeros(size(Ocy_pos_voxel))
    CL_DT_S = pyconvert(Array{Float32, 3},edt_S_py(PythonCall.PyArray(outer)))
    HC_DT_S = pyconvert(Array{Float32, 3},edt_S_py(PythonCall.PyArray(inner)))
    for ii in eachindex(t_form)
        x = Ocy_pos_voxel[ii][1]; y = Ocy_pos_voxel[ii][2]; z = Ocy_pos_voxel[ii][3];
        t_form[ii] = CL_DT_S[x,y,z] / (CL_DT_S[x,y,z] + HC_DT_S[x,y,z])
    end
    return t_form
end

a_outer, a_inner = build_outer_inner(paths)

# estimating formation times
t_form = estimate_Ocy_formation_time(a_outer,a_inner,Ocy_pos_voxel)

# ordereing formation time 
t_form_index = sortperm(t_form)
t_form_ordered = t_form[t_form_index]
Ocy_pos_ordered = Ocy_pos[t_form_index]
Ocy_pos_voxel_ordered = Ocy_pos_voxel[t_form_index]

tvals = []
Ocy_pos_in_domain = []
Ocy_pos_voxel_in_domain = []

for (ti,t) in enumerate(t_form_ordered)
    if 0.0 <= t <= 1.0
        push!(tvals, t_form_ordered[ti])
        push!(Ocy_pos_in_domain, Ocy_pos_ordered[ti])
        push!(Ocy_pos_voxel_in_domain, Ocy_pos_voxel_ordered[ti])
    end
end

idx = collect(1:10:61)
tvals2 = tvals[idx]

#tvals = collect(0:ΔT:1.0)
outer_dt_S = pyconvert(Array{Float32, 3},edt_S_py(PythonCall.PyArray(a_outer)))
inner_dt_S = pyconvert(Array{Float32, 3},edt_S_py(PythonCall.PyArray(a_inner)))

ϕ = compute_ϕ_stack_3D_py(a_outer, a_inner, tvals2) # for some reason code breaks here
#ϕ = ϕ_func(0, outer_dt_S, inner_dt_S)

K2 = OsteonSurfaceAnalysis.compute_curvature(ϕ[:,:,:,1],dx,dy,dz)

H,W,D = size(ϕ[:,:,:,1])
x = (collect(1:H).-1).*dx
y = (collect(1:W).-1).*dy
f1 = Figure(size=(800,800))
a1 = Axis3(f1[1, 1], title = "Computed curvature from ϕ at z = $(6*dz)")
surface!(a1, x, y, K2[:,:,6], colormap=:jet)

f1 = Figure(size=(800,800))
a1 = Axis3(f1[1, 1], title = "Computed curvature from ϕ")
contour!(a1, 0 .. H*dx, 0 .. W*dy, 0 .. D*dz, ϕ[:,:,:,1], levels = [0.0], isorange = 3, colormap=:jet, colorrange=(-0.1,0.1))
contour!(a1, 0 .. H*dx, 0 .. W*dy, 0 .. D*dz, ϕ[:,:,:,end], levels = [0.0], isorange = 3, colormap=:jet, colorrange=(-0.1,0.1))
scatter!(a1, Ocy_pos, markersize=12, color=:green)

"""
    curvature_at_contour(K, x, y, zgrid, zsel, X, Y; onbounds = :throw)

Return a vector `k` with curvature values sampled from the z-slice of `K`
corresponding to `zsel`, bilinearly interpolated at coordinates `(X[i], Y[i])`.

Arguments
- K     :: AbstractArray{T,3}   — size (length(x), length(y), length(zgrid))
- x,y   :: AbstractVector{<:Real}  — monotonically increasing grid axes
- zgrid :: AbstractVector{<:Real}  — z axis (same length as size(K,3))
- zsel  :: Integer or Real       — either a z **index** (1..end) or a z **coordinate**
- X,Y   :: AbstractVector{<:Real} — contour coordinates (same length)

Keyword
- onbounds :: Symbol — what to do if (X,Y) is outside the [x,y] domain:
    :throw (default), :clamp (use nearest valid cell and clamp), or :NaN.

Notes
- If `zsel` is a Real, the nearest z-layer in `zgrid` is used.
- Uses bilinear interpolation on the chosen z-slice. If X/Y land exactly on
  grid nodes, this reduces to exact node sampling.
"""
function curvature_at_contour(K::AbstractArray{T,3},
                              x::AbstractVector{<:Real},
                              y::AbstractVector{<:Real},
                              zgrid::AbstractVector{<:Real},
                              zsel,
                              X::AbstractVector{<:Real},
                              Y::AbstractVector{<:Real};
                              onbounds::Symbol = :throw) where {T}

    nx, ny, nz = size(K)
    @assert nx == length(x) "size(K,1) must equal length(x)"
    @assert ny == length(y) "size(K,2) must equal length(y)"
    @assert nz == length(zgrid) "size(K,3) must equal length(zgrid)"
    @assert length(X) == length(Y) "X and Y must have same length"
    @assert issorted(x) && issorted(y) && issorted(zgrid) "x, y, zgrid must be sorted"

    # Resolve z index
    k = if zsel isa Integer
        1 <= zsel <= nz || throw(BoundsError("z index $zsel not in 1:$nz"))
        zsel
    else
        # nearest index to the given z coordinate
        j = searchsortedfirst(zgrid, zsel)
        j <= 1 ? 1 :
        j > nz ? nz :
        (abs(zgrid[j] - zsel) < abs(zgrid[j-1] - zsel) ? j : j-1)
    end

    Ks = @view K[:, :, k]  # 2D slice at the chosen z
    res = Vector{T}(undef, length(X))

    # Helper to locate the x/y cell index just to the "left/below" of value v
    # returning an index in 1:(n-1), plus the local interpolation weight t ∈ [0,1]
    local function cell_and_t(grid::AbstractVector{<:Real}, v::Real, onbounds::Symbol)
        n = length(grid)
        # Fast paths for bounds
        if v <= grid[1]
            if onbounds === :throw
                throw(BoundsError("value $v < grid minimum $(grid[1])"))
            elseif onbounds === :NaN
                return (1, NaN)
            else # :clamp
                return (1, 0.0)
            end
        elseif v >= grid[n]
            if onbounds === :throw
                throw(BoundsError("value $v > grid maximum $(grid[n])"))
            elseif onbounds === :NaN
                return (n-1, NaN)
            else # :clamp — clamp to last cell, t=1
                return (n-1, 1.0)
            end
        end

        i_hi = searchsortedfirst(grid, v)       # 2..n
        i_lo = i_hi - 1                         # 1..n-1
        g0, g1 = grid[i_lo], grid[i_hi]
        t = (v - g0) / (g1 - g0)
        return (i_lo, t)
    end

    @inbounds for i in eachindex(X)
        xi, yi = X[i], Y[i]
        ix, tx = cell_and_t(x, xi, onbounds)
        iy, ty = cell_and_t(y, yi, onbounds)

        if isnan(tx) || isnan(ty)
            res[i] = T(NaN)
            continue
        end

        # Corners: (ix,iy)=lower-left
        v00 = Ks[ix,   iy  ]
        v10 = Ks[ix+1, iy  ]
        v01 = Ks[ix,   iy+1]
        v11 = Ks[ix+1, iy+1]

        # Bilinear interpolation
        res[i] = (1 - tx)*(1 - ty)*v00 +
                 tx       *(1 - ty)*v10 +
                 (1 - tx)*ty       *v01 +
                 tx       *ty       *v11
    end

    return res
end

set_theme!(theme_black(), fontsize = 30)
H,W,D = size(ϕ[:,:,:,1])
x = collect(1.0:H)
y = collect(1.0:W)
z = collect(1.0:D)
f1 = Figure(size=(800,800))
a1 = Axis3(f1[1, 1], title = "Computed curvature from ϕ")
dx = 0.379; dy = 0.379; dz = 0.4;
for ti in axes(ϕ,4)
    K2 = OsteonSurfaceAnalysis.compute_curvature(ϕ[:,:,:,ti],dx,dy,dz)
    z_layer = Ocy_pos_voxel_in_domain[idx[ti]][3]
    #K4 = OsteonSurfaceAnalysis.compute_curvature_4th(ϕ[:,:,:,ti],dx,dy,dz)
    X,Y = OsteonSurfaceAnalysis.compute_zero_contour_xy_coords(ϕ, z_layer, ti);
    k2 = curvature_at_contour(K2,x,y,z,z_layer,X,Y)
    #k4 = curvature_at_contour(K4,x,y,z,z_layer,X,Y)
    println("average second order curvature on 0 contour: ", mean(k2))
    #println("average fourth order curvature on 0 contour: ", mean(k4))
    #println("average difference between approximations on 0 contour: ", mean(k4 .- k2))
    lines!(a1, X*dx, Y*dy, ones(length(X)).*z_layer*dz,linewidth=4,color=k2, colormap=:dense, colorrange = (-0.01, 0.01))
end
scatter!(a1, Ocy_pos_in_domain[1:10:61], markersize=15,color=:red)
Colorbar(f1[1,2], colormap=:dense, colorrange = (-0.4, 0.4))


Ocy_count = collect(1:length(tvals))
Ocy_form_rate = Ocy_count ./ tvals
f1 = Figure(size=(800,800))
a1 = GLMakie.Axis(f1[1, 1], title = "Ocy form rate", xlabel="t", ylabel="# formed Ocy / t")
lines!(a1, tvals, Ocy_form_rate, linewidth=3,color=:white)
scatter!(a1, tvals, Ocy_form_rate, markersize=15,color=:white)