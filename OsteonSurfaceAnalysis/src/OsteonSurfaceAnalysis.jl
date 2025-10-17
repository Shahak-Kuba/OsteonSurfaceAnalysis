module OsteonSurfaceAnalysis

using FileIO, Images, ImageBinarization, ImageMorphology, ImageSegmentation, ImageFiltering, Statistics
using DistanceTransforms
using GLMakie
using CSV, DataFrames
import Contour as CTR

include("Imaging.jl")
include("LevelSet.jl")
include("Geometry.jl")
include("Analysis.jl")
include("Plotting.jl")

using .Imaging
using .LevelSet
using .Geometry
using .Analysis
using .Plotting

export
# Imaging
process_mask_data, extract_sample_name, build_outer_inner, circle_mask,#make_tdelay_mask,
# Level set / distance fields
edt, edt_S, compute_ϕ_at_t, compute_ϕ_stack, compute_ϕ_at_t_3D_py, compute_ϕ_stack_3D_py
# Geometry & intersections
compute_zero_contour_xy_coords, Ω, compute_xy_center, compute_planes_and_intersections, proj_3D_onto_XZ,
#Analysis
analysis_Tdelay_pairs, compute_curvature, compute_curvature_4th, estimate_Ocy_formation_time, 
# Plotting entrypoints
plot_3d_contours!, plot_3d_contours_w_intersections!, plot_example_slices!, plot_α_β!

end # module