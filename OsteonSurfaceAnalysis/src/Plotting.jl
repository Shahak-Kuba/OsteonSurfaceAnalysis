module Plotting

using GLMakie
import Contour as CTR

export plot_3d_contours!, plot_3d_contours_w_intersections!, plot_example_slices!, plot_α_β!

function plot_3d_contours!(ax, ϕ, Δz, tvals)
    H,W,_,_ = size(ϕ)
    x = collect(1:H); y = collect(1:W)
    for (ti,t) in enumerate(tvals)
        ϕ_bottom = ϕ[:,:,1,ti]
        cset_bot = CTR.contours(x,y,ϕ_bottom, [0])
        line_bot = first(CTR.lines(first(CTR.levels(cset_bot))))
        x_bot, y_bot = CTR.coordinates(line_bot)

        ϕ_top = ϕ[:,:,2,ti]
        cset_top = CTR.contours(x,y,ϕ_top, [0])
        line_top = first(CTR.lines(first(CTR.levels(cset_top))))
        x_top, y_top = CTR.coordinates(line_top)

        lines!(ax, x_bot, y_bot, zeros(size(x_bot)), linewidth=3, color=:red)
        lines!(ax, x_top, y_top, Δz.*ones(size(x_top)), linewidth=3, color=:blue)
    end
end

function plot_3d_contours_w_intersections!(ax, ϕ, Δz, tvals, center_top, center_bot, intersecting_points_per_contour)
    H,W,_,_ = size(ϕ)
    x = collect(1:H); y = collect(1:W)
    for (ti,t) in enumerate(tvals)
        ϕ_bottom = ϕ[:,:,1,ti]
        cset_bot = CTR.contours(x,y,ϕ_bottom, [0])
        line_bot = first(CTR.lines(first(CTR.levels(cset_bot))))
        x_bot, y_bot = CTR.coordinates(line_bot)

        ϕ_top = ϕ[:,:,2,ti]
        cset_top = CTR.contours(x,y,ϕ_top, [0])
        line_top = first(CTR.lines(first(CTR.levels(cset_top))))
        x_top, y_top = CTR.coordinates(line_top)

        lines!(ax, x_bot, y_bot, zeros(size(x_bot)), linewidth=3, color=:red)
        lines!(ax, x_top, y_top, Δz.*ones(size(x_top)), linewidth=3, color=:blue)
    end
    scatter!(ax, center_bot[1], center_bot[2], 0, markersize=30)
    scatter!(ax, center_top[1], center_top[2], Δz, markersize=30)
    # plotting intersections
    for ii in axes(intersecting_points_per_contour,1)[1:end-1]
        for jj in axes(intersecting_points_per_contour[1],1)
            scatter!(ax,intersecting_points_per_contour[ii][jj][1][1], intersecting_points_per_contour[ii][jj][1][2], intersecting_points_per_contour[ii][jj][1][3], markersize=25, color=:green)
            scatter!(ax,intersecting_points_per_contour[ii+1][jj][2][1], intersecting_points_per_contour[ii+1][jj][2][2], intersecting_points_per_contour[ii+1][jj][2][3], markersize=25, color=:cyan)
            lines!(ax, [intersecting_points_per_contour[ii][jj][1][1], intersecting_points_per_contour[ii+1][jj][2][1]], [intersecting_points_per_contour[ii][jj][1][2], intersecting_points_per_contour[ii+1][jj][2][2]], [intersecting_points_per_contour[ii][jj][1][3], intersecting_points_per_contour[ii+1][jj][2][3]], linewidth=3, color = :orange)
        end
    end
end

function plot_example_slices!(axes, Tdelay_proj_points, line_∇)
    for ang in eachindex(axes)
        ax = axes[ang]; # 1 axes is 1 angle
        for cont in eachindex(Tdelay_proj_points) 
            x = [Tdelay_proj_points[cont][ang][1][1], Tdelay_proj_points[cont][ang][2][1]]
            y = [Tdelay_proj_points[cont][ang][1][2], Tdelay_proj_points[cont][ang][2][2]]
            lines!(ax, x, y, linewidth=3,color=line_∇[cont][ang].*ones(size(x)), colormap=:jet, colorrange=(-10,10))
        end
    end
end

function plot_α_β!(ax, α, β, tvals)
    for jj in 1:2:size(α,1)
        scatter!(ax, rad2deg.(β), α[jj,:], markersize = 15)
        lines!(ax, rad2deg.(β), α[jj,:], linewidth = 3, label = "T = $(tvals[jj])")
    end
end

end # end of module