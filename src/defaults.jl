function compute_θ_initial(dimension)
    # TODO: compute initial hyperparams from xs, ys, see https://infallible-thompson-49de36.netlify.app/
    #       for now, return each time the same constant inital θ
    l = ones(dimension)
    s_var = 1.0
    n_var = 0.09
    # lengthscale λ_i in [0.005,2.0], signal variance s^2 in [0.05,20.0], noise var. σ^2 in [0.0005,0.1]
    return (;
        lengthscales = bounded(l, 0.005, 2.0),
        signal_var = bounded(s_var, 0.05, 20.0),
        noise_var = bounded(n_var, 0.0005, 0.1),
    )
end

# noise_var is passed into AbstractGPs directly, not via a kernel
function kernel_creator(hyperparameters)
    return hyperparameters.signal_var *
           with_lengthscale(KernelFunctions.Matern52Kernel(), hyperparameters.lengthscales)
end

const DEFAULT_INIT_BASE_LENGTH = 0.8
function compute_tr_config(dimension, batch_size)
    # Original paper: failure_tolerance = Int(ceil(dimension / batch_size)),
    # here we set failure_tolerance as in https://botorch.org/tutorials/turbo_1
    return TurboTRConfig(
        2^(-7), #length_min
        1.6, # length_max
        Int(ceil(max(4.0 / batch_size, dimension / batch_size))), # failure_tolerance
        3, # success_tolerance
    )
end
