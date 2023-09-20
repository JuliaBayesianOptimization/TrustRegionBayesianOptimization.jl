using Pkg
Pkg.activate("examples/")

# add it by running `dev ../TrustRegionBayesianOptimization.jl/`
# -> need to manually add before (they have no versions):
# add https://github.com/samuelbelko/SurrogatesBase.jl.git#param-abstract-type
# add https://github.com/JuliaBayesianOptimization/SurrogatesAbstractGPs.jl.git
# add https://github.com/JuliaBayesianOptimization/AbstractBayesianOptimization.jl.git
using TrustRegionBayesianOptimization

# -- for plotting --
using Plots
using LinearAlgebra
# plotlyjs()
gr()
# ----

# lb = left bottom point in domain, ub = top right point in domain
lb, ub = [-2.0, -1.0], [2.0, 2.0]

rosenbrock(x::Vector; kwargs...) = rosenbrock(x[1], x[2]; kwargs...)
# non-convex function
rosenbrock(x1, x2) = 100 * (x2 - x1^2)^2 + (1 - x1)^2

minima(::typeof(rosenbrock)) = [[1, 1]], 0
mins, fmin = minima(rosenbrock)


function p()
    plt = contour(-2:0.1:2, -1:0.1:2, (x, y) -> -rosenbrock([x, y]), levels = 500,
                  fill = true)
    plt = scatter!((x -> x[1]).(get_hist(oh)[1]), (x -> x[2]).(get_hist(oh)[1]),
                   label = "eval. hist")
    plt = scatter!((x -> x[1]).(mins), (y -> y[2]).(mins), label = "true minima",
                   markersize = 10, shape = :diamond)
    plt = scatter!([get_solution(oh)[1][1]], [get_solution(oh)[1][2]],
                   label = "observed min.", shape = :rect)
    plt
end

# g, sense::Sense, lb, ub, max_evaluations
oh = OptimizationHelper(rosenbrock, Min, lb, ub, 200)

# oh, n_surrogates, batch_size, n_init_for_local
dsm = Turbo(oh, 3, 8, 10)
policy = TurboPolicy(oh)

# run initial sampling, create initial trust regions and local models
initialize!(dsm, oh)

# savefig(p(), "plot_before_optimization.png")
#display(p())

# Optimize
optimize!(dsm, policy, oh)

# savefig(p(), "plot_after_optimization.png")
display(p())

observed_dist = minimum((m -> norm(get_solution(oh)[1] .- m)).(mins))
observed_regret = abs(get_solution(oh)[2] - fmin)
