include("TurboTR.jl")
include("../defaults.jl")

mutable struct TurboState
    is_done::Bool
end

"""
`TuRBO` with an `GPSurrogate` local model.

We assume that the domain is `[0,1]^dim` and we are maximizing.
"""
struct Turbo{J,D<:Real,R<:Real} <: AbstractDecisionSupportModel
    oh::OptimizationHelper
    state::TurboState
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
    tr_config::TurboTRConfig{D}
    # TODO: type of sobol seq -> Move to QuasiMonteCarlo
    sobol_generator::J
    # TODO: verbosity levels?
    verbose::Bool
    surrogates::Vector{GPSurrogate{Vector{D},R}}
    trs::Vector{TurboTR{D,R}}
end

"""
Turbo will deliberately stop the optimization loop if it has no evaluation budget left for
initialization of a restarted trust region.
"""
AbstractBayesianOptimization.is_done(dsm::Turbo; verbose = true) = dsm.state.is_done

# TODO: document: compute_θ_initial has to include lengthscales hyperparameters
# kernel_creator, compute_θ_initial from defaults.jl
function Turbo(
    oh::OptimizationHelper,
    n_surrogates,
    batch_size,
    n_init_for_local;
    kernel_creator = kernel_creator,
    θ_initial = compute_θ_initial(get_dimension(oh)),
    tr_config::TurboTRConfig{U} = compute_tr_config(get_dimension(oh), batch_size),
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

    return Turbo(
        oh,
        TurboState(false),
        n_surrogates,
        batch_size,
        n_init_for_local,
        kernel_creator,
        θ_initial,
        tr_config,
        sobol_gen,
        verbose,
        Vector{GPSurrogate{Vector{D},R}}(undef, n_surrogates),
        Vector{TurboTR{D, R}}(undef, n_surrogates),
    )
end


function AbstractBayesianOptimization.initialize!(dsm::Turbo, oh::OptimizationHelper)
    for i = 1:(dsm.n_surrogates)
        initialize_local!(dsm, oh, i)
    end
    return nothing
end

function is_evaluation_budget_for(n, oh::OptimizationHelper)
    return n <= get_max_evaluations(oh) - get_evaluation_counter(oh)
end

"""
Initialize i-th local model and its trust region.

We use it also for restarting a TR after its convergence.
"""
function initialize_local!(dsm::Turbo{J,D,R}, oh::OptimizationHelper, i) where {J,D,R}
    if ! is_evaluation_budget_for(dsm.n_init_for_local, oh)
        dsm.state.is_done = true
        dsm.verbose || @info "Cannot initialize new trust region, no evaluation budget left."
        return nothing
    end

    # TODO: make initial sampler a parameter of Turbo
    init_xs = [next!(dsm.sobol_generator) for _ = 1:(dsm.n_init_for_local)]
    init_ys = evaluate_objective!(oh, init_xs)

    dsm.surrogates[i] = GPSurrogate(
        init_xs,
        init_ys,
        kernel_creator = dsm.kernel_creator,
        hyperparameters = ParameterHandling.value(dsm.θ_initial),
    )
    dsm.verbose && @info @sprintf "initialized surrogate %2i" i

    update_hyperparameters!(dsm.surrogates[i], BoundedHyperparameters(dsm.θ_initial))
    dsm.verbose && @info @sprintf "initial hyperparam. opt. on %2i" i

    # TODO: in noisy observations, set center to max. of posterior mean
    # set center to observed maximizer, observed in a local model
    center = init_xs[argmax(init_ys)]
    lengths = compute_lengths(
        DEFAULT_INIT_BASE_LENGTH,
        dsm.surrogates[i].hyperparameters.lengthscales,
        get_dimension(oh),
    )
    # method from TurboTR.jl
    lb, ub = compute_lb_ub(center, lengths)

    dsm.trs[i] = TurboTR(
        dsm.tr_config,
        DEFAULT_INIT_BASE_LENGTH,
        lengths,
        center,
        lb,
        ub,
        0,
        0,
        init_xs[argmax(init_ys)],
        maximum(init_ys),
        false,
    )
    dsm.verbose && @info @sprintf "initialized tr %2i" i
    return nothing
end

"""
Process new evaluations `ys` at points `xs`, i.e, update local models and adapt trust regions.
"""
function AbstractBayesianOptimization.update!(dsm::Turbo, oh::OptimizationHelper, xs, ys)
    length(xs) == length(ys) || throw(ErrorException("xs, ys have different lengths"))
    for i = 1:(dsm.n_surrogates)
        # filter out points in the i-th trust region,
        tr_pairs = filter(t -> is_in_tr(dsm.trs[i], t[1]), collect(zip(xs, ys)))
        tr_xs = (t -> t[1]).(tr_pairs)
        tr_ys = (t -> t[2]).(tr_pairs)
        length(tr_xs) == length(tr_ys) ||
            throw(ErrorException("tr_xs, tr_ys have different lengths"))

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