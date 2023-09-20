"""
An implementation of Trust Region Bayesian Optimization (TuRBO) algorithm from
Scalable Global Optimization via Local Bayesian Optimization appearing in NeurIPS 2019.
"""
module TrustRegionBayesianOptimization

using Printf

# from https://github.com/JuliaBayesianOptimization/AbstractBayesianOptimization.jl.git
using AbstractBayesianOptimization
# from https://github.com/JuliaBayesianOptimization/SurrogatesAbstractGPs.jl.git
# have to add https://github.com/samuelbelko/SurrogatesBase.jl.git#param-abstract-type
# as well since it is not registered an it is a dependency for SurrogatesAbstractGPs
using SurrogatesAbstractGPs
using ParameterHandling
using KernelFunctions
# TODO: change to QuasiMonteCarlo samplers
using Sobol


# building blocks: decision support model & policy
export Turbo, TurboPolicy
export initialize!, optimize!, next_batch!
# helper utilities from AbstractBayesianOptimization
export OptimizationHelper, Min, Max, get_hist, get_solution

include("DecisionSupportModel/Turbo.jl")
include("TurboPolicy.jl")

end # module TurboAlgorithm
