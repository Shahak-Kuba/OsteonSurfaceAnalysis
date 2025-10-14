using OsteonSurfaceAnalysis, FileIO, GLMakie

processed_paths = readdir("./DATA/FM40-1-R2/Processed_Images/"; join=true)

paths = [processed_paths[1], processed_paths[end]]
downsample = 1
Δz = 80.0;
Δθ = pi/5;
ΔT = 0.1;

θs = collect(0.0:Δθ:pi); 
if θs[end] ≈ pi; pop!(θs); end

a_outer, a_inner = build_outer_inner(paths)
tvals = collect(0:ΔT:1.0)
ϕ = OsteonSurfaceAnalysis.compute_ϕ_stack(a_outer, a_inner, tvals)

top_center_2D = OsteonSurfaceAnalysis.compute_xy_center(ϕ, 2, length(tvals))
bottom_center_2D = OsteonSurfaceAnalysis.compute_xy_center(ϕ, 1, length(tvals))

top_center_3D = (top_center_2D[1], top_center_2D[2], Δz)
bottom_center_3D = (bottom_center_2D[1], bottom_center_2D[2], 0.0)

intersecting_points, cutting_planes = OsteonSurfaceAnalysis.compute_planes_and_intersections(ϕ,Δz,tvals,θs, top_center_3D, bottom_center_3D)

proj_points = OsteonSurfaceAnalysis.proj_3D_onto_XZ(intersecting_points, cutting_planes, top_center_3D, bottom_center_3D)

Tdelay_proj_points, line_∇, α = OsteonSurfaceAnalysis.analysis_Tdelay_pairs(proj_points)





# Plotting Results
GLMakie.activate!()
set_theme!(theme_black(), fontsize = 24)

# contour plots
f1 = Figure(size=(800,800))
a1 = Axis3(f1[1, 1], title = "ϕ contours")
OsteonSurfaceAnalysis.plot_3d_contours!(a1,ϕ,Δz, tvals)

# contour plot with intersections
f2 = Figure(size=(800,800))
a1 = Axis3(f2[1, 1], title = "ϕ contours")
OsteonSurfaceAnalysis.plot_3d_contours_w_intersections!(a1,ϕ,Δz, tvals, top_center_3D, bottom_center_3D, intersecting_points)

# example slice plot
β = repeat(θs, 2);
for i in size(θs,1)+1:length(β)
    β[i] += π
end
f3 = Figure(size = (800, 1600))
a_1 = GLMakie.Axis(f3[1, 1], title = "θ = $(round(rad2deg(β[1]))) deg")
a_2 = GLMakie.Axis(f3[1, 2], title = "θ = $(round(rad2deg(β[2]))) deg")
a_3 = GLMakie.Axis(f3[2, 1], title = "θ = $(round(rad2deg(β[3]))) deg")
a_4 = GLMakie.Axis(f3[2, 2], title = "θ = $(round(rad2deg(β[4]))) deg")
a_5 = GLMakie.Axis(f3[3, 1], title = "θ = $(round(rad2deg(β[5]))) deg")
a_6 = GLMakie.Axis(f3[3, 2], title = "θ = $(round(rad2deg(β[6]))) deg")
a_7 = GLMakie.Axis(f3[4, 1], title = "θ = $(round(rad2deg(β[7]))) deg")
a_8 = GLMakie.Axis(f3[4, 2], title = "θ = $(round(rad2deg(β[8]))) deg")
ax = [a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8]
OsteonSurfaceAnalysis.plot_example_slices!(ax, Tdelay_proj_points, line_∇)
Colorbar(f3[:,3],  colormap=:jet, colorrange=(-10,10), label="gradient")

# α vs β plot
f4 = Figure(size = (1200, 800))
a1 = GLMakie.Axis(f4[1, 1], title = "T delay = $(ΔT)", xlabel="β", ylabel="α")
OsteonSurfaceAnalysis.plot_α_β!(a1, α, β, tvals)
Legend(f4[1,2], a1, "ϕ times")

display(f1)
display(f2)
display(f3)
display(f4)