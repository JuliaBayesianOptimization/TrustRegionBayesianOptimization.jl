include("TurboTR.jl")
include("../defaults.jl")



"""
`TuRBO` with an `AbstractGPSurrogate` local model.

We assume that the domain is `[0,1]^dim` and we are maximizing.
"""
struct Turbo{J,D<:Real,R<:Real} <: AbstractDecisionSupportModel
    oh::OptimizationHelper
    # number of surrogates
    n_surrogates::Int
    batch_size::Int
    # mum of initial samples for each local model
    n_init_for_local::Int
    # use kernel_creator for constructing surrogates at initialization and init. after restarting a TR
    kernel_creator::Function
    # for hyperparameter optimization of local models
    θ_initial::NamedTuple
    # save TR options for use in initialize!(..) and in later restarts of TRs
    tr_options::NamedTuple
    # TODO: type of sobol seq -> Move to QuasiMonteCarlo
    sobol_generator::J
    # TODO: verbosity levels?
    verbose::Bool
    surrogates::Vector{AbstractGPSurrogate}
    trs::Vector{TurboTR{D,R}}
end


"""
Turbo is never deliberately stopping the optimization loop.
"""
AbstractBayesianOptimization.is_done(dsm::Turbo; verbose = true) = false


# TODO: document: compute_θ_initial has to include lengthscales hyperparameters
# kernel_creator, compute_θ_initial from defaults.jl
function Turbo(
    oh::OptimizationHelper,
    n_surrogates,
    batch_size,
    n_init_for_local;
    kernel_creator = kernel_creator,
    θ_initial = compute_θ_initial(get_dimension(oh)),
    tr_config::TurboTRConfig{U} = compute_tr_config(
        get_dimension(oh),
        batch_size
    ),
    verbose = true,
) where {U}
    D = get_domain_eltype(oh)
    R = get_range_type(oh)
    U == D || throw(
        ErrorException(
            "tr_config has to use parameteric type that coincides with the type of elements in the domain, as provided in optimization helper",
        ),
    )
    dimension = get_dimension(oh)

    # TODO: how many samples do we need to skip for Sobol for better uniformity?
    # -> Move to QuasiMonteCarlo
    sobol_gen = SobolSeq(dimension)
    # skip first 2^10 -1 samples
    skip(sobol_gen, 10)

    # create placeholders for surrogates and trs
    surrogates = [
        GPSurrogate(
            Vector{Vector{D}}(),
            Vector{R}();
            kernel_creator = kernel_creator,
            hyperparameter = ParameterHandling.value(dsm.θ_initial),
        ) for i = 1:n_surrogates
    ]
    trs = [
        TurboTR(
            tr_config,
            Vector{D}(undef, dimension),
            Vector{D}(undef, dimension),
            Vector{D}(undef, dimension),
            Vector{D}(undef, dimension),
            0,
            0,
            Vector{D}(undef, dimension),
            R(-Inf), # convert -Inf to the appropriate type
            false,
        ) for i = 1:n_surrogates
    ]
    return Turbo(
        oh,
        n_surrogates,
        batch_size,
        n_init_for_local,
        kernel_creator,
        θ_initial,
        merged_tr_options,
        sobol_gen,
        verbose,
        surrogates,
        trs,
    )
end


function AbstractBayesianOptimization.initialize!(dsm::Turbo, oh::OptimizationHelper)
    for i = 1:(dsm.n_surrogates)
        initialize_local!(dsm, oh, i)
    end
    return nothing
end

function evaluation_budget_for(n, oh::OptimizationHelper)
    return get_max_evaluations(oh) - get_evaluation_counter(oh) >= n
end

"""
Initialize i-th local model and its trust region.

We use it also for restarting a TR after its convergence.
"""
function initialize_local!(dsm::Turbo, oh::OptimizationHelper, i)
    evaluation_budget_for(dsm.n_init_for_local, oh) ||
        throw(ErrorException("cannot initialize, no evaluation budget left"))

    # TODO: make initial sampler a parameter of Turbo
    init_xs = [next!(dsm.sobol_generator) for _ = 1:(dsm.n_init_for_local)]
    init_ys = evaluate_objective!(oh, init_xs)

    dsm.surrogates[i] = AbstractGPSurrogate(
        init_xs,
        init_ys,
        kernel_creator = dsm.kernel_creator,
        hyperparameters = ParameterHandling.value(dsm.θ_initial),
    )
    dsm.verbose && @info @sprintf "initialized %2i" i

    update_hyperparameters!(dsm.surrogates[i], BoundedHyperparameters(dsm.θ_initial))
    dsm.verbose && @info @sprintf "initial hyperparam. opt. on %2i" i

    # TODO: in noisy observations, set center to max. of posterior mean
    # set center to observed maximizer, observed in a local model
    dsm.trs[i].center .= init_xs[argmax(init_ys)]
    dsm.trs[i].observed_maximizer .= init_xs[argmax(init_ys)]
    dsm.trs[i].observed_maximum = maximum(init_ys)

    # method from TurboTR.jl
    dsm.trs[i].lengths .= compute_lengths(
        dsm.tr_options.base_length,
        dsm.surrogates[i].hyperparameters.lengthscales,
        get_dimension(oh),
    )
    dsm.trs[i].lb, dsm.trs[i].ub = compute_lb_ub(center, lengths)
    return nothing
end

"""
Process new evaluations `ys` at points `xs`, i.e, update local models and adapt trust regions.
"""
function AbstractBayesianOptimization.update!(dsm::Turbo, oh::OptimizationHelper, xs, ys)
    length(xs) == length(ys) || throw(ErrorException("xs, ys have different lengths"))
    for i = 1:(dsm.n_surrogates)
        # filter out points in the i-th trust region,
        tr_pairs = filter((x, y) -> is_in_tr(dsm.trs[i], x), zip(xs, ys))
        tr_xs = ((x, y) -> x).(tr_pairs)
        tr_ys = ((x, y) -> y).(tr_pairs)
        length(tr_xs) == length(tr_ys) || throw(ErrorException("tr_xs, tr_ys have different lengths"))

        if !isempty(tr_xs)
            # add points in i-th trust region to i-th surrogate
            add_point!(dsm.surrogates[i], tr_xs, tr_ys)
            # each time we add a batch of points, run hyperparameter optimization
            update_hyperparameters!(
                dsm.surrogates[i],
                BoundedHyperparameters(dsm.θ_initial),
            )
            dsm.verbose && @info @sprintf "hyperparmeter optimization run on %2i" i
            # update corresponding TR - counters, base_length, lengths, tr_is_done
            @assert !isempty(tr_ys)
            update_TR!(
                dsm.trs[i],
                tr_xs,
                tr_ys,
                dsm.surrogates[i].hyperparameters.lengthscales,
                get_dimension(oh),
            )
        end
        # restart TR if it converged
        if dsm.trs[i].tr_is_done
            dsm.verbose && @info @sprintf "restarting trust region %2i" i
            # initalize_local! performs hyperparamter optimization on initial sample
            initialize_local!(dsm, oh, i)
        end
    end
    return nothing
end
