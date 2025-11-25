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

function compute_EDT_S_py(outer, inner)
    outer_dt_S = pyconvert(Array{Float32, 3},edt_S_py(PythonCall.PyArray(outer)))
    inner_dt_S = pyconvert(Array{Float32, 3},edt_S_py(PythonCall.PyArray(inner)))
    return outer_dt_S, inner_dt_S
end

ϕ_func = (t,S_DTʰ,S_DTᶜ) -> (1-t) .* S_DTʰ - (t) .* S_DTᶜ

function compute_ϕ_stack_3D(outer_dt_S, inner_dt_S, tvals)
    H, W, Z = size(outer_dt_S)
    ϕ = zeros(Float32, H, W, Z, length(tvals))

    for (ti, t) in enumerate(tvals)
        ϕ[:,:,:,ti] .= ϕ_func(t, outer_dt_S, inner_dt_S)
    end
    return ϕ
end

function estimate_Ocy_formation_time(outer_dt_S, inner_dt_S, Ocy_pos_voxel)
    t_form = zeros(size(Ocy_pos_voxel))
    for ii in eachindex(t_form)
        x = Ocy_pos_voxel[ii][1]; y = Ocy_pos_voxel[ii][2]; z = Ocy_pos_voxel[ii][3];
        t_form[ii] = outer_dt_S[x,y,z] / (outer_dt_S[x,y,z] + inner_dt_S[x,y,z])
    end
    return t_form
end

function unwrap_top_level_set_contour(ϕ, t_index, dx, dy)
    X,Y = OsteonSurfaceAnalysis.compute_zero_contour_xy_coords(ϕ,1,t_index)
    x = X .* dx; y = Y .* dy;
    
    # computing centroid based of Haversian Canal
    centroid = OsteonSurfaceAnalysis.compute_xy_center(ϕ,1,size(ϕ,4))
    #println(centroid)
    x_c = centroid[1] * dx; y_c = centroid[2] * dy;

    # shifting osteon to the center
    x = x .- x_c; y = y .- y_c

    # converting to polar #### SONETHING IS NOT WORKING HERE ######
    R = sqrt.(x.^2 + y.^2);
    function theta_from_coords(x, y)
        θ = atan.(y, x)  # returns angle in range -π to π
        for (ii, angle) in enumerate(θ)
            #println(angle)
            if angle < 0
                θ[ii] += 2π      # shift negative angles into 0 to 2π
            end
        end
        return θ
    end
    θ = theta_from_coords(x, y)
    return R, θ
end


a_outer, a_inner = build_outer_inner(paths)
outer_dt_S, inner_dt_S = compute_EDT_S_py(a_outer, a_inner) # only compute this once to save time

# estimating formation times
t_form = estimate_Ocy_formation_time(outer_dt_S,inner_dt_S,Ocy_pos_voxel)

# ordereing formation time 
t_form_index = sortperm(t_form)
t_form_ordered = t_form[t_form_index]
Ocy_pos_ordered = Ocy_pos[t_form_index]
Ocy_pos_voxel_ordered = Ocy_pos_voxel[t_form_index]

tvals = t_form_ordered

#ϕ = compute_ϕ_stack_3D(outer_dt_S, inner_dt_S, tvals) # for some reason code breaks here

function compute_curvature_near_osteocyte(tvals,outer_dt_S,inner_dt_S,Ocy_pos_voxel_ordered,dx,dy)
    mean_available_κ = []
    κ_at_osteocyte = []

    function nearest_index(X, Y, x, y)
        @assert length(X) == length(Y) "X and Y must be the same length"
        @assert !isempty(X) "X and Y cannot be empty"
        # Compute squared distances (faster than using sqrt)
        d2 = @. (X - x)^2 + (Y - y)^2
        # Return index of the minimum distance
        i = argmin(d2)
        return i
    end

    for (idx,t_formed) in enumerate(tvals)
        # compute level set at that time
        ϕ = ϕ_func(t_formed, outer_dt_S, inner_dt_S)
        z_layer = Ocy_pos_voxel_ordered[idx][3]
        osteocyte_x = Ocy_pos_voxel_ordered[idx][1].*dx
        osteocyte_y = Ocy_pos_voxel_ordered[idx][2].*dy
        # find zero level contour
        X,Y = OsteonSurfaceAnalysis.compute_zero_contour_xy_coords(ϕ, z_layer, idx);
        # choose k depending on at what scale youre measuring curvature
        contour_curvature = OsteonSurfaceAnalysis.compute_2D_curvature(X.*dx,Y.*dy; k=20);
        push!(mean_available_κ, mean(contour_curvature))
        push!(κ_at_osteocyte, contour_curvature[nearest_index(X.*dx, Y.*dy, osteocyte_x, osteocyte_y)])
        
    end
    return κ_at_osteocyte, mean_available_κ
end

κ_at_osteocyte, mean_available_κ = compute_curvature_near_osteocyte(tvals,outer_dt_S,inner_dt_S,Ocy_pos_voxel_ordered,dx,dy)


set_theme!(theme_black(), fontsize = 30)
GLMakie.activate!()
f1 = Figure(size=(1900,600))
a1 = Axis(f1[1, 1], title = "κ vs t_form", xlabel="t_form", ylabel="κ")
a2 = Axis(f1[1, 2], title = "...", xlabel="κ - mean_κ", ylabel="t_form", limits=(-0.1, 0.1, 0, 1))
a3 = Axis(f1[1, 3], title = "mean_κ vs t_form", xlabel="t_form", ylabel="κ - mean_κ")
scatter!(a1, tvals, κ_at_osteocyte, markersize=15, color=:red)
scatter!(a2, κ_at_osteocyte .- mean_available_κ, tvals, markersize=15, color=:red)
scatter!(a3, tvals, mean_available_κ, markersize=15, color=:red)



GLMakie.activate!()
f1 = Figure(size=(800,800))
a1 = Axis(f1[1, 1], title = "Unwrapped osteon lamella: top level", xlabel="θ [rad]", ylabel="R [μm]")
for ii in axes(ϕ,4)
    R, θ = unwrap_top_level_set_contour(ϕ, ii, dx, dy)
    lines!(a1, θ[sortperm(θ)], R[sortperm(θ)], linewidth = 3)
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
    z_layer = Ocy_pos_voxel_in_domain[idx[ti]][3]
    X,Y = OsteonSurfaceAnalysis.compute_zero_contour_xy_coords(ϕ, z_layer, ti);
    R = sqrt.(X.^2 + Y.^2)
    curvature = OsteonSurfaceAnalysis.Analysis.local_curvature(X,Y; k=50)
    lines!(a1, X*dx, Y*dy, ones(length(X)).*z_layer*dz,linewidth=4,color=curvature, colormap=:dense, colorrange = (-0.01, 0.01))
end
scatter!(a1, Ocy_pos_in_domain[1:10:61], markersize=15,color=:red)
Colorbar(f1[1,2], colormap=:dense, colorrange = (-0.01, 0.01))


f1 = Figure(size=(800,800))
a1 = GLMakie.Axis(f1[1, 1], title = "Computed curvature from ϕ")
dx = 0.379; dy = 0.379; dz = 0.4;
z_layer = 70
ϕ = ϕ_func(0.2, outer_dt_S, inner_dt_S)
X,Y = OsteonSurfaceAnalysis.compute_zero_contour_xy_coords(ϕ, z_layer, 1);
R = sqrt.((X.*dx).^2 + (Y.*dy).^2)
k2 = OsteonSurfaceAnalysis.compute_2D_curvature(X.*dx,Y.*dy; k=50);
lines!(a1, 1 ./ k2, linewidth = 3, color = :blue)
lines!(a1, R, linewidth = 3, color = :red)


for ti in axes(ϕ,4)
    K2 = OsteonSurfaceAnalysis.compute_curvature(ϕ[:,:,:,ti],dx,dy,dz)
    z_layer = 70#Ocy_pos_voxel_in_domain[idx[ti]][3]
    #K4 = OsteonSurfaceAnalysis.compute_curvature_4th(ϕ[:,:,:,ti],dx,dy,dz)
    X,Y = OsteonSurfaceAnalysis.compute_zero_contour_xy_coords(ϕ, z_layer, ti);
    R = sqrt.(X.^2 + Y.^2)
    k2 = curvature_at_contour(K2,x,y,z,z_layer,X,Y)
    lines!(a1, k2, linewidth = 3)

end


Ocy_count = collect(1:length(tvals))
t_between_formation = diff([0;tvals])
f1 = Figure(size=(800,800))
a1 = GLMakie.Axis(f1[1, 1], title = "non dim time between osteocyte formation", xlabel="Osteocyte #", ylabel="dt")
lines!(a1, Ocy_count, t_between_formation, linewidth=3,color=:white)
scatter!(a1, Ocy_count, t_between_formation, markersize=15,color=:white)