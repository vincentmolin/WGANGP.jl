# 8 2d Gaussians
# Point tensorboard to ./logs when running to monitor the training

using WGANGP
using Flux
using CUDA
using GLMakie
using Printf
using Dates
using Random
using ProgressMeter
using Base.Iterators: partition
using TensorBoardLogger
using Logging
using BSON: @save, @load
using Parameters: @with_kw

function sample_8gaussians(n; r = 10.0, sd = 1.0)
    centers = hcat(collect([r*cos(θ), r*sin(θ)] for θ in 0:π/4:7π/4)...)
    sd .* randn(Float32, 2, n) + centers[:,rand(1:8, n)]
end

# WARN: Actually doesn't mutate genr and crit if they live on CPU when
# this function is called, so make sure to |> gpu them or 
# catch them when they come home from the gym.
function train!(genr, crit, opt_genr, opt_crit, data, hps = Hyperparameters(); run_prefix = nothing, runid = nothing)

    if hps.gpu == true
        device = gpu
    else
        error("CPU training not implemented")
    end

    genr = genr |> device
    crit = crit |> device

    # Infer latent_dim from generator network architecture
    latent_dim = size(genr.layers[1].weight, 2)

    # Create some folders 
    runid = isnothing(runid) ? "mnist" : runid
    path = mkpath("output/$runid")
    imspath = mkpath("$path/ims")

    # Tensorboard logger
    tblog = TBLogger("logs/$runid")

    # For output images
    fixed_noise = randn(Float32, latent_dim, 25) |> device

    train_steps = 0
    loss_genr = 0.0
    p = Progress(hps.epochs * length(data), "Training...")

    data_batches = [data[:, r] |> device for
    r in partition(1:size(data, 2), hps.batch_size)]
    
    z = CUDA.zeros(latent_dim, hps.batch_size)

    with_logger(tblog) do
        for epoch in 1:hps.epochs
            for i in randperm(length(data_batches))
                z = CUDA.randn!(z)
                x_generated = genr(z)
                x_true = data_batches[i]

                loss_crit = step_critic!(opt_crit, crit, x_true, x_generated; λ = hps.λ_reg)

                if train_steps % hps.critic_train_factor == 1 || hps.critic_train_factor == 1
                    CUDA.@sync CUDA.randn!(z)
                    loss_genr = step_generator!(opt_genr, genr, crit, z)
                end

                if train_steps % hps.verbose_freq == 0
                    @info "Losses" critic = loss_crit generator = loss_genr
                    ProgressMeter.update!(p, train_steps;
                        showvalues = [
                            ("Epoch", epoch),
                            ("Step", train_steps),
                            ("Critic loss", loss_crit),
                            ("Generator loss", loss_genr)]
                    )
                end
                if train_steps % hps.sample_freq == 0
                    #CUDA.@sync zs = CUDA.randn(latent_dim, 1024)
                    #generated = cpu(genr(zs))
                    #scatter(generated[1,:], generated[2,:])
                end
                train_steps += 1
            end
        end
    end

    return genr, crit
end


function create_models(latent_dim = 2; device = gpu)
    genr = Chain(
        Dense(latent_dim => 8, relu),
        Dense(8 => 8, relu),
        Dense(8 => 2)
    ) |> device

    crit = Chain(
        Dense(2 => 8, relu),
        Dense(8 => 4, relu),
        Dense(4 => 1)
    ) |> device

    return (genr, crit)
end

@with_kw struct Hyperparameters
    batch_size::Int = 64
    critic_train_factor::Int = 5
    λ_reg::Float32 = 0.1f0
    epochs::Int = 10
    verbose_freq::Int = 20
    sample_freq::Int = 1000
    gpu::Bool = true
end

genr, crit = create_models()
opt_genr = Adam(1e-4, (0.5, 0.9))
opt_crit = Adam(1e-4, (0.5, 0.9))
hps = Hyperparameters()
data = sample_8gaussians(128 * 100)
train!(genr, crit, opt_genr, opt_crit, data, hps)

CUDA.@sync zs = CUDA.randn(2, 1024)
generated = cpu(genr(zs))
scatter(generated[1,:], generated[2,:])


latent_dim = 4

genr = Chain(
            Dense(latent_dim, 7 * 7 * 8, relu),
            x -> reshape(x, 7, 7, 8, :),
            ConvTranspose((5, 5), 8 => 4; stride = 1, pad = 2),
            BatchNorm(128, relu),
            ConvTranspose((4, 4), 4 => 2; stride = 2, pad = 1),
            BatchNorm(64, relu),
            ConvTranspose((4, 4), 2 => 1, sigmoid; stride = 2, pad = 1)
        )