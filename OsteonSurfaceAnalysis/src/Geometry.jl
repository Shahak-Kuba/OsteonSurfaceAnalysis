module Geometry

using LinearAlgebra
using Statistics
import Contour as CTR

export compute_zero_contour_xy_coords, Ω, compute_xy_center,
Plane, compute_planes_and_intersections, proj_3D_onto_XZ
# --------------------------- finding the 2D 0 level contour at z_layer ------------------------------
function compute_zero_contour_xy_coords(ϕ,z_layer,tval_idx)
    if length(size(ϕ)) > 3
        H,W,D = size(ϕ[:,:,:,tval_idx])
        ϕ_at_t = ϕ[:,:,z_layer,end]
    else
        H,W,D = size(ϕ)
        ϕ_at_t = ϕ[:,:,z_layer]
    end

    x = collect(1:H)
    y = collect(1:W)
    cset = CTR.contours(x,y,ϕ_at_t,[0.0])
    line = first(CTR.lines(first(CTR.levels(cset))))
    X,Y = CTR.coordinates(line)

    # check direction
    if (Y[2] - Y[1] > 0.0 && X[2] - X[1] > 0.0) ||  (Y[2] - Y[1] > 0.0 && X[2] - X[1] < 0.0)
        return reverse(X), reverse(Y)
    else
        return X,Y
    end
end

# polygon area
function Ω(x, y)
    A = 0.0
    n = length(x)
    for ii in 1:n
        j = ii == n ? 1 : ii + 1
        A += x[ii]*y[j] - y[ii]*x[j]
    end
    return abs(A) / 2
end

function compute_xy_center(ϕ,z_layer,tval)
    x,y = compute_zero_contour_xy_coords(ϕ,z_layer,tval)
    if x[1] != x[end] || y[1] != y[end]
        x = vcat(x, x[1])
        y = vcat(y, y[1])
    end
    A = Ω(x, y)
    x_centroid = 1 / (6 * A) * sum((x[1:end-1] + x[2:end]) .* (x[1:end-1] .* y[2:end] - x[2:end] .* y[1:end-1]))
    y_centroid = 1 / (6 * A) * sum((y[1:end-1] + y[2:end]) .* (x[1:end-1] .* y[2:end] - x[2:end] .* y[1:end-1]))
    return (x_centroid, y_centroid)
end

# planes, intersections, plane mappings (kept bodies the same)
# Functions to generate z-cutting planes and calculate which points from the contours intersect with the plane
# ----- tuple helpers -----
dot3(a,b) = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
minus(a,b) = (a[1]-b[1], a[2]-b[2], a[3]-b[3])
plus(a,b)  = (a[1]+b[1], a[2]+b[2], a[3]+b[3])
scale(a,s) = (a[1]*s, a[2]*s, a[3]*s)
cross3(a,b) = (a[2]*b[3]-a[3]*b[2], a[3]*b[1]-a[1]*b[3], a[1]*b[2]-a[2]*b[1])
norm3(a) = sqrt(dot3(a,a))
normalize3(v) = (s = norm3(v); s==0 ? (0.0,0.0,0.0) : scale(v, 1/s))

struct Plane{T}
    p0::NTuple{3,T}   # a point on the plane
    n::NTuple{3,T}    # (not necessarily unit) normal
end

# Rodrigues rotation of a vector v around unit axis û by angle θ
function rotate_about_axis(v::NTuple{3,Float64}, û::NTuple{3,Float64}, θ::Float64)
    c, s = cos(θ), sin(θ)
    term1 = scale(v, c)
    term2 = scale(cross3(û, v), s)
    term3 = scale(û, dot3(û, v) * (1 - c))
    plus(plus(term1, term2), term3)
end


"""
    plane_through_centers(top, bot; θ=0.0, ref=(0.0,0.0,1.0))

Return a plane `Plane(p0, n)` that:
- contains the line through `top` and `bot`
- is obtained by rotating an initial plane around that line by angle `θ` (radians)

`ref` picks the initial orientation before rotation (any vector not parallel to the line).
"""
function plane_through_centers(top::NTuple{3,Float64}, bot::NTuple{3,Float64};
                               θ::Float64=0.0, ref::NTuple{3,Float64}=(0.0,0.0,1.0))
    # Axis of rotation = line through the centers
    axis = minus(top, bot)
    û = normalize3(axis)
    if norm3(û) == 0
        error("Top and bottom centers coincide; axis undefined.")
    end

    # Build a vector inside the plane that is orthogonal to the axis:
    # take ref, remove its component along the axis
    v0 = minus(ref, scale(û, dot3(ref, û)))
    if norm3(v0) == 0
        # ref was parallel to the axis; pick another
        ref2 = abs(û[3]) < 0.9 ? (0.0,0.0,1.0) : (1.0,0.0,0.0)
        v0 = minus(ref2, scale(û, dot3(ref2, û)))
    end
    v0 = normalize3(v0)

    # Initial plane normal is n0 = axis × v0  (so plane contains the axis)
    n0 = cross3(û, v0)

    # Rotate the plane normal around the axis by θ
    nθ = rotate_about_axis(n0, û, θ)

    # Plane through any point on the line (use top)
    return Plane{Float64}(top, nθ)
end

function intersect_segment_with_plane(p::NTuple{3,Float64},
                                      q::NTuple{3,Float64},
                                      pl::Plane{Float64}; eps=1e-12)
    pq = minus(q, p)
    denom = dot3(pl.n, pq)
    num   = -dot3(pl.n, minus(p, pl.p0))
    if abs(denom) < eps
        return false, (0.0,0.0,0.0)   # parallel/coplanar case ignored
    end
    t = num/denom
    if t < -eps || t > 1+eps
        return false, (0.0,0.0,0.0)
    end
    return true, plus(p, scale(pq, clamp(t, 0.0, 1.0)))
end

function intersect_polylines_with_plane(polys::Vector{Vector{NTuple{3,Float64}}},
                                        pl::Plane{Float64}; closed::Bool=false)
    hits = NTuple{3,Float64}[]
    for poly in polys
        n = length(poly); n < 2 && continue
        segs = closed ? [(i, i % n + 1) for i in 1:n] : [(i, i+1) for i in 1:n-1]
        for (i,j) in segs
            hit, X = intersect_segment_with_plane(poly[i], poly[j], pl)
            hit && push!(hits, X)
        end
    end
    hits
end

function contours_to_3d_polylines(cset, zval)
    polys = Vector{Vector{NTuple{3,Float64}}}()
    for lvl in CTR.levels(cset)
        for ln in CTR.lines(lvl)
            xs, ys = CTR.coordinates(ln)
            push!(polys, [(xs[i], ys[i], zval) for i in eachindex(xs)])
        end
    end
    polys
end

function angle_between_vectors(v1, v2)
    cosθ = dot(v1, v2) / (norm(v1) * norm(v2))
    return acos(clamp(cosθ, -1.0, 1.0))
end

function compute_planes_and_intersections(ϕ, Δz, tvals, θvals, top_center, bottom_center)
    H,W,D = size(ϕ[:,:,:,1])
    x = collect(1:H)
    y = collect(1:W)
    intersecting_points_per_contour = []
    cutting_planes = []

    for (ti,t) in enumerate(tvals)
        ϕ_bottom = ϕ[:,:,1,ti]
        ϕ_top = ϕ[:,:,end,ti]
        cset_top = CTR.contours(x,y,ϕ_top, [0])
        cset_bot = CTR.contours(x,y,ϕ_bottom, [0])
        intersecting_points_per_theta = []
        unordered_cutting_planes = []
        for θ in θvals
            pl = plane_through_centers(top_center, bottom_center; θ)
            push!(unordered_cutting_planes, pl)
       
            # Build 3D polylines for all relevant contours (from top & bottom slices, or multiple z’s)
            polys3d = vcat(contours_to_3d_polylines(cset_top, Δz),
                contours_to_3d_polylines(cset_bot, 0.0))
            hits3d = intersect_polylines_with_plane(polys3d, pl; closed=true)
            if length(hits3d) > 4
                hits3d = unique(hits3d)
            end
            if norm(hits3d[1] .- hits3d[3]) < norm(hits3d[1] .- hits3d[4])
                pair_1 = [hits3d[1]; hits3d[3]]
                pair_2 = [hits3d[2]; hits3d[4]]
            else
                pair_1 = [hits3d[1]; hits3d[4]]
                pair_2 = [hits3d[2]; hits3d[3]]
            end
            if angle_between_vectors(pair_1[1][1:2] .- top_center[1:2], (1,0)) - pi/2 > 0
                push!(intersecting_points_per_theta, pair_1, pair_2)
            else
                push!(intersecting_points_per_theta, pair_2, pair_1)
            end
        end
        # ordering points by their (x,y) position
        # Order all the first points in each pair counterclockwise based on (x, y)
        pts = [pair[1] for pair in intersecting_points_per_theta]
        center_x = mean(getindex.(pts, 1))
        center_y = mean(getindex.(pts, 2))
        angles = atan.((getindex.(pts, 2) .- center_y), (getindex.(pts, 1) .- center_x))
        ordered_indices = reverse(sortperm(angles))
        ordered_pts = intersecting_points_per_theta[ordered_indices]
        ordered_cutting_planes = repeat(unordered_cutting_planes,inner=2)[ordered_indices]
        push!(intersecting_points_per_contour, ordered_pts)
        push!(cutting_planes, ordered_cutting_planes)
        #push!(intersecting_points_per_contour, intersecting_points_per_theta)
    end
    return intersecting_points_per_contour, cutting_planes
end

# --------------------------------------- Projecting 3D cutting planes onto X-Z axis ------------------------
# Build a right-handed orthonormal basis {u,v} for a plane (u×v ≈ n̂)
function plane_basis(pl::Plane{Float64})
    n̂ = normalize3(pl.n)
    # pick helper not parallel to n̂
    h = abs(n̂[3]) < 0.9 ? (0.0,0.0,1.0) : (1.0,0.0,0.0)
    u = normalize3(cross3(h, n̂))   # in-plane axis 1
    v = cross3(n̂, u)               # in-plane axis 2 (already unit)
    return u, v, n̂
end

# Orthogonal projection of a point to the plane
function project_to_plane(p::NTuple{3,Float64}, pl::Plane{Float64})
    _, _, n̂ = plane_basis(pl)
    r = minus(p, pl.p0)
    d = dot3(r, n̂)
    return minus(p, scale(n̂, d))   # p - d*n̂
end

"""
    map_points_plane_to_plane(points, src, dst; θ=0.0, s=1.0, project=true)

Map 3D `points` lying on (or near) the source `src::Plane` to the target `dst::Plane`
so that their **in-plane coordinates** are preserved (distribution maintained).

Options:
- `θ`  : extra in-plane rotation (radians) applied in the *target* plane
- `s`  : uniform in-plane scale factor (1.0 keeps distances)
- `project` : if true, orthogonally project input points onto `src` first

Returns the new points on `dst` as `Vector{NTuple{3,Float64}}`.
"""
function map_points_plane_to_plane(points::Vector{NTuple{3,Float64}},
                                   src::Plane{Float64},
                                   dst::Plane{Float64};
                                   θ::Float64=0.0, s::Float64=1.0,
                                   project::Bool=true)

    us, vs, _ = plane_basis(src)
    ud, vd, _ = plane_basis(dst)

    c, si = cos(θ), sin(θ)

    qpts = NTuple{3,Float64}[]
    for p in points
        # 1) make sure we’re on the src plane (optional but robust)
        ps = project ? project_to_plane(p, src) : p

        # 2) source-plane local coords (u,v)
        r = minus(ps, src.p0)
        u = dot3(r, us)
        v = dot3(r, vs)

        # 3) optional in-plane rotation & scale (in target plane)
        up =  s*( c*u - si*v)
        vp =  s*( si*u + c*v)

        # 4) rebuild in target plane
        q = plus(dst.p0, plus(scale(ud, up), scale(vd, vp)))
        push!(qpts, q)
    end
    return qpts
end

function proj_3D_onto_XZ(intersecting_points, cutting_planes, top_center, bottom_center)
    plotting_pl = plane_through_centers((0.0,0.0,1.0), (0.0,0.0,0.0); θ=0.0)
    proj_points = []

    for cont in eachindex(intersecting_points)
        proj_points_per_cont = []
        for ang in eachindex(intersecting_points[1])
            pl = cutting_planes[cont][ang]
            points = intersecting_points[cont][ang]
            proj_pair = map_points_plane_to_plane(points, pl, plotting_pl)
            x = [proj_pair[1][1], proj_pair[2][1]]; y = [proj_pair[1][3], proj_pair[2][3]] .- minimum([proj_pair[1][3], proj_pair[2][3]]);
            push!(proj_points_per_cont, [(x[1],y[1]), (x[2],y[2])])
        end
        push!(proj_points, proj_points_per_cont)
    end
    return proj_points #proj_points_right, proj_points_left #proj_centers (NEEDS to be fixed)
end



end # end of module