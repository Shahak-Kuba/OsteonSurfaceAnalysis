using OsteonSurfaceAnalysis, FileIO, GLMakie
data_path = readdir("././DATA/"; join=true)
output_path = data_path[3]

r1 = 10; r2 = 100; 
output_path_r1 = output_path * "/small_r.bmp"
circle_mask(r1, output_path_r1)
output_path_r2 = output_path * "/large_r.bmp"
circle_mask(r2, output_path_r2)

path_HCa = output_path_r1
path_On = output_path_r2
processes_output_path = output_path * "/circle_example.png"
OsteonSurfaceAnalysis.generate_RG_img_from_data(path_HCa, path_On, processes_output_path)

path = [processes_output_path]
paths = repeat(path,20)
downsample = 1
Δz = 1.0;
Δθ = pi/6;
ΔT = 0.2;

a_outer, a_inner = build_outer_inner(paths)
tvals = collect(0:ΔT:1.0)
ϕ = OsteonSurfaceAnalysis.compute_ϕ_stack_3D(a_outer, a_inner, tvals)

top_center_2D = OsteonSurfaceAnalysis.compute_xy_center(ϕ, length(paths), length(tvals))
bottom_center_2D = OsteonSurfaceAnalysis.compute_xy_center(ϕ, 1, length(tvals))

top_center_3D = (top_center_2D[1], top_center_2D[2], Δz * length(paths))
bottom_center_3D = (bottom_center_2D[1], bottom_center_2D[2], 0.0)





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



H,W,D = size(ϕ[:,:,:,1])
x = collect(1.0:H)
y = collect(1.0:W)
z = collect(1.0:D)
z_layer = 3

K4 = OsteonSurfaceAnalysis.compute_curvature_4th(ϕ[:,:,:,ti],dx,dy,dz)
f1 = Figure(size=(800,800))
a1 = GLMakie.Axis3(f1[1, 1], title = "Computed curvature from ϕ")
X = [xi for xi in x, yi in y, zi in z]
Y = [yi for xi in x, yi in y, zi in z]
Z = [zi for xi in x, yi in y, zi in z]



f2 = Figure(size=(800,800))
a1 = GLMakie.Axis(f2[1, 1], title = "Computed radius from ϕ(x,t) = 0 and 1 / κ")
centroid = OsteonSurfaceAnalysis.compute_xy_center(ϕ, 10, size(ϕ,4))
dx = 0.379; dy = 0.379; dz = 0.4;
ti = 2;
K2 = OsteonSurfaceAnalysis.compute_curvature(ϕ[:,:,:,ti],dx,dy,dz)
z_layer = 10
X,Y = OsteonSurfaceAnalysis.compute_zero_contour_xy_coords(ϕ, z_layer, ti);
R = sqrt.((X .- centroid[1]).^2 .+ (Y .- centroid[2]).^2)
k2 = curvature_at_contour(K2,x,y,z,z_layer,X,Y)
lines!(a1, 1 ./ k2, linewidth = 3, color = :blue)
lines!(a1, R, linewidth = 3, color = :red)


f1 = Figure(size=(800,800))
a1 = Axis(f1[1, 1], title = "Computed curvature from ϕ")
dx = 1; dy = 1; dz = 1;
centroid = OsteonSurfaceAnalysis.compute_xy_center(ϕ, 10, size(ϕ,4))
for ti in axes(ϕ,4)
    #K2 = OsteonSurfaceAnalysis.compute_curvature(ϕ[:,:,:,ti],dx,dy,dz)
    #K4 = OsteonSurfaceAnalysis.compute_curvature_4th(ϕ[:,:,:,ti],dx,dy,dz)
    X,Y = OsteonSurfaceAnalysis.compute_zero_contour_xy_coords(ϕ, z_layer, ti);
    curvature = OsteonSurfaceAnalysis.Analysis.local_curvature(X,Y; k = 20)
    R = sqrt.((X .- centroid[1]).^2 .+ (Y .- centroid[2]).^2)
    #k2 = curvature_at_contour(K2,x,y,z,z_layer,X,Y)
    #k4 = curvature_at_contour(K4,x,y,z,z_layer,X,Y)
    lines!(a1, X.*dx, Y.*dy, linewidth=4,color=curvature, colormap=:jet, colorrange = (0.01, 0.1))
end
Colorbar(f1[1,2], colormap=:jet, colorrange = (0.01, 0.1))

f1 = Figure(size=(800,800))
a1 = Axis3(f1[1, 1], title = "Computed curvature from ϕ")
lines!(a1, X, Y, linewidth=4,color=k, colormap=:jet, colorrange = (minimum(k), maximum(k)))


# approximating curvature new method
X,Y = OsteonSurfaceAnalysis.compute_zero_contour_xy_coords(ϕ, z_layer, ti);
curvature = OsteonSurfaceAnalysis.Analysis.local_curvature(X,Y; k = 20)

