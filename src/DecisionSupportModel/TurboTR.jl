struct TurboTRConfig{D<:Real}
    # base side length of a hyperrectangle trust region
    base_length::D
    length_min::D
    length_max::D
    failure_tolerance::Int
    success_tolerance::Int
end

"""
Maintain the state of one trust region.
"""
mutable struct TurboTR{D<:Real,R<:Real}
    const config::TurboTRConfig{D}
    # lengths for each dim are rescaled wrt lengthscales in fitted GP while maintaining
    # volume (base_length)^dim
    lengths::Vector{D}
    center::Vector{D}
    lb::Vector{D}
    ub::Vector{D}
    failure_counter::Int
    success_counter::Int
    observed_maximizer::Vector{D}
    observed_maximum::R
    tr_is_done::Bool
end

function is_in_tr( tr::TurboTR, x)
    all(tr.lb .<= x .<= tr.ub)
end

function compute_lb_ub(center, lengths)
    # intersection of TR with [0,1]^dim
    lb = max.(0, min.(center .- 1 / 2 .* lengths, 1))
    ub = max.(0, min.(center .+ 1 / 2 .* lengths, 1))
    lb, ub
end

function compute_lengths(base_length, lengthscales, dimension)
    # stability trick as in https://github.com/uber-research/TuRBO/blob/de0db39f481d9505bb3610b7b7aa0ebf7702e4a5/turbo/turbo_1.py#L184
    lengthscales = lengthscales ./ (sum(lengthscales) / dimension)
    lengthscales .* base_length ./ prod(lengthscales)^(1 / dimension)
end

"""
Update TR state - success and failure counters, base_length, lengths, tr_is_done.
"""
function update_TR!(tr::TurboTR, tr_xs, tr_ys, lengthscales, dimension)
    @assert length(tr_xs) == length(tr_ys)
    @assert length(tr_xs) != 0
    batch_max = maximum(tr_ys)
    # add epsilon to RHS like in https://botorch.org/tutorials/turbo_1
    if batch_max > tr.observed_maximum + 10^-3 * abs(tr.observed_maximum)
        # "success"
        tr.success_counter += 1
        tr.failure_counter = 0
        # TODO: set tr_center to max posterior mean in case of noisy observations?
        tr.center = tr.observed_maximizer = tr_xs[argmax(tr_ys)]
        tr.observed_maximum = batch_max
    else
        # "failure"
        tr.success_counter = 0
        tr.failure_counter += length(tr_xs)
    end
    # update trust region base_length
    if tr.success_counter == tr.success_tolerance
        # expand TR
        tr.base_length = min(2.0 * tr.base_length, tr.length_max)
        tr.success_counter = 0
    elseif tr.failure_counter >= tr.failure_tolerance
        # shrink TR
        tr.base_length /= 2.0
        tr.failure_counter = 0
    end
    # check for convergence, if we are done, we don't need to update lengths anymore
    if tr.base_length < tr.length_min
        tr.tr_is_done = true
    else
        # update lengths wrt updated lengthscales
        tr.lengths = compute_lengths(tr.base_length, lengthscales, dimension)
        tr.lb, tr.ub = compute_lb_ub(tr.center, tr.lengths)
    end
end
