data_path = readdir("./DATA/"; join=true)
output_path = data_path[3]

path_HCa = readdir("./DATA/FM40-1-R2/HCa"; join=true)[1]
paths_On = readdir("./DATA/FM40-1-R2/On"; join=true)
path_On_top = paths_On[1]
path_On_bot = paths_On[end]
processes_output_path_top = output_path * "/Synthetic_Osteon_top.png"
OsteonSurfaceAnalysis.generate_RG_img_from_data(path_HCa, path_On_top, processes_output_path_top)
processes_output_path_bot = output_path * "/Synthetic_Osteon_bot.png"
OsteonSurfaceAnalysis.generate_RG_img_from_data(path_HCa, path_On_bot, processes_output_path_bot)

paths = [processes_output_path_top, processes_output_path_bot]
downsample = 1
Δz = 50.0;
Δθ = pi/4;
ΔT = 0.2;

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

proj_points_right, proj_points_left = OsteonSurfaceAnalysis.proj_3D_onto_XZ(intersecting_points, cutting_planes, top_center_3D, bottom_center_3D)

Tdelay_proj_points_right, Tdelay_proj_points_left, line_∇_right, line_∇_left, α = OsteonSurfaceAnalysis.analysis_Tdelay_pairs(proj_points_right, proj_points_left, tvals, intersecting_points)





# Plotting Results
GLMakie.activate!()
set_theme!(theme_black(), fontsize = 36)

# contour plots
f1 = Figure(size=(800,800))
a1 = Axis3(f1[1, 1], title = "ϕ contours")
OsteonSurfaceAnalysis.plot_3d_contours!(a1,ϕ,Δz, tvals)

# contour plot with intersections
f2 = Figure(size=(800,800))
a1 = Axis3(f2[1, 1], title = "ϕ contours")
OsteonSurfaceAnalysis.plot_3d_contours_w_intersections!(a1,ϕ,Δz, tvals, top_center_3D, bottom_center_3D, intersecting_points)

# example slice plot
reshaped_proj_points_left = reshape(proj_points_left, (length(tvals),Int(length(proj_points_right)/length(tvals))))
f3 = Figure(size = (1400, 800))
a_1 = GLMakie.Axis(f3[1, 1], title = "θ = $(round(rad2deg(θs[1]))) deg")#, limits=(0,1000,-50,0))
a_2 = GLMakie.Axis(f3[1, 2], title = "θ = $(round(rad2deg(θs[2]))) deg")#, limits=(0,1000,-50,0))
a_3 = GLMakie.Axis(f3[2, 1], title = "θ = $(round(rad2deg(θs[3]))) deg")#, limits=(0,1000,-50,0))
a_4 = GLMakie.Axis(f3[2, 2], title = "θ = $(round(rad2deg(θs[4]))) deg")#, limits=(0,1000,-50,0))
ax = [a_1,a_2,a_3,a_4]
OsteonSurfaceAnalysis.plot_example_slices!(ax,reshaped_proj_points_left, Tdelay_proj_points_right, Tdelay_proj_points_left, line_∇_right, line_∇_left, cutting_planes)
Colorbar(f3[:,3],  colormap=:jet, colorrange=(-10,10), label="gradient")

# α vs β plot
β = repeat(θs, 2);
for i in size(θs,1)+1:length(β)
    β[i] += π
end
f4 = Figure(size = (1200, 800))
a1 = GLMakie.Axis(f4[1, 1], title = "T delay = $(ΔT)", xlabel="β", ylabel="α")
OsteonSurfaceAnalysis.plot_α_β!(a1, α, β, tvals)
Legend(f4[1,2], a1, "ϕ times")

display(f1)
display(f2)
display(f3)
display(f4)
