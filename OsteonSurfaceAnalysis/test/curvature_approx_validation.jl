# Generating discrete points of a circle

R = 10;
θ = collect(LinRange(0,2π,30));
splice!(θ, length(θ)) # removing 2π
x = R .* cos.(θ)
y = R .* sin.(θ)
N = length(x)

using OsteonSurfaceAnalysis
estimated_curvature = OsteonSurfaceAnalysis.Analysis.compute_2D_curvature_SK(x,y; k=2);

using CairoMakie
f = Figure(size=(500,500))
ax = Axis(f[1,1])
lines!(ax, vec(1 ./ estimated_curvature), linewidth=3, label="R_est = 1/κ_est")
lines!(ax, ones(length(estimated_curvature)) .* R, linewidth=3, label="True R")

display(f)





wrap(i) = mod1(i, N)

function tangent(x,y,i)
    ip = wrap(i+1); im = wrap(i-1)
    tx = x[ip] - x[im]
    ty = y[ip] - y[im]
    n = hypot(tx, ty)
    return n == 0 ? (1.0, 0.0) : (tx/n, ty/n)
end

function rotate(p::Tuple{<:Real,<:Real}, θ::Real; about=(0.0, 0.0))
    x, y   = p
    cx, cy = about
    c = cos(θ); s = sin(θ)
    dx, dy = x - cx, y - cy
    return (cx + c*dx - s*dy,  cy + s*dx + c*dy)
end

curvature = zeros(length(x),1)
ii = 20; k = 1;
left  = [wrap(ii - s) for s in k:-1:1]
right = [wrap(ii + s) for s in 1:k]
ii_considered = vcat(left, ii, right)
x_central = x[ii]
y_central = y[ii]
x_considered = x[ii_considered]
y_considered = y[ii_considered]


# calculating tangent at ii
tx, ty = tangent(x,y,ii)

rotation_θ = atan(ty,tx)
rad2deg(rotation_θ )
# move the points so that the ii point is at the origin
x_considered_center = x_considered .- x_central
y_considered_center = y_considered .- y_central
points = [(X,Y) for (X,Y) in zip(x_considered_center,y_considered_center)]

# rotate the points
rotated_points = []
for point in points
    push!(rotated_points,rotate(point, -rotation_θ))
end

A = zeros(length(points), 3)
for jj in eachindex(rotated_points)
    A[jj, :] .= [rotated_points[jj][1]^2, rotated_points[jj][1], 1]
end

 y_array = [y for (x,y) in rotated_points]

a,b,c = (A'*A)\(A'*y_array)

f = Figure(size=(500,500))
ax = Axis(f[1,1])
scatter!(ax, x_considered, y_considered, color=:red)
scatter!(ax, [p[1] for p in rotated_points], [p[2] for p in rotated_points], color=:green)

# generating parabola with fitted coefficients
parab(x, a,b,c) = a*x.^2 .+ b*x .+ c
X = LinRange(-R,R,100)
Y = parab.(X, a,b,c)
lines!(ax, X, Y, color=:black, linewidth=2)


num = abs(2*a)
# since I center the points about a central point (0,0) then 2ax = 0 
# + eps is to avoid division by 0
eps = 1e-10
denom = ( 1 + b^2 )^(3/2) + eps 

# estimating curvature from parabola
curvature= num / denom