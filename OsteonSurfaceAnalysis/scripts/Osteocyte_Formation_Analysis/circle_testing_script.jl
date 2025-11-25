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












f1 = Figure(size=(800,800))
a1 = GLMakie.Axis(f1[1, 1], title = "Computed curvature from ϕ")
dx = 0.379; dy = 0.379; dz = 0.4;
z_layer = 10
#ϕ = ϕ_func(0.2, outer_dt_S, inner_dt_S)
X,Y = OsteonSurfaceAnalysis.compute_zero_contour_xy_coords(ϕ, z_layer, 5);
R = sqrt.(((X .- mean(X)).*dx).^2 + ((Y .- mean(Y)).*dy).^2)

#scatter!(a1, (X .- mean(X)).*dx, (Y .- mean(Y)).*dy)

k = OsteonSurfaceAnalysis.compute_2D_curvature(X.*dx,Y.*dy; k=20);
lines!(a1, 1 ./ vcat(k...), linewidth = 3, color = :blue)
lines!(a1, R, linewidth = 3, color = :red)

f2 = Figure(size=(800,800))
a2 = GLMakie.Axis(f2[1, 1], title = "zero contour")
scatter!(a2, (X .- mean(X)).*dx, (Y .- mean(Y)).*dy, color = :green)








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

