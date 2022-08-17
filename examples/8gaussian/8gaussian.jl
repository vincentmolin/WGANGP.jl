# 2 1d Gaussians
# Point tensorboard to ./logs when running to monitor the training

using WGANGP
using Flux
using CUDA
using GLMakie
using Printf
using Dates
using ProgressMeter
using TensorBoardLogger
using Logging
using BSON: @save, @load
using Parameters: @with_kw

@with_kw struct Hyperparameters
    batch_size::Int = 64
    critic_train_factor::Int = 5
    batches_per_epoch = 16
    epochs::Int = 10
    verbose_freq::Int = 20
    sample_freq::Int = 1000
    gpu::Bool = true
end

function sample_8gaussians(n; r = 10.0, sd = 1.0)
    centers = hcat(collect([r*cos(θ), r*sin(θ)] for θ in 0:π/4:7π/4)...)
    sd .* randn(Float32, 2, n) + centers[:,rand(1:8, n)]
end

# WARN: Actually doesn't mutate genr and crit if they live on CPU when
# this function is called, so make sure to |> gpu them or 
# catch them when they come home from the gym.
function train!(genr, crit, opt_genr, opt_crit, hps = Hyperparameters(); run_prefix = nothing, runid = nothing)

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
    p = Progress(hps.epochs * hps.batches_per_epoch, "Training...")

    with_logger(tblog) do
        for epoch in 1:hps.epochs

            data = [ cu(sample_8gaussians(hps.batch_size)) for _ in hps.batches_per_epoch ]

            for i in eachindex(data)

                z = CUDA.randn(latent_dim, hps.batch_size)
                x_generated = genr(z)
                x_true = data[i]

                loss_crit = step_critic!(opt_crit, crit, x_true, x_generated)

                if i % hps.critic_train_factor == 1 || hps.critic_train_factor == 1
                    z = CUDA.randn(latent_dim, hps.batch_size)
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
                    # TODO Plot
                end
                train_steps += 1
            end
        end
    end

    return genr, crit
end


function create_models(latent_dim = 2)
    genr = Chain(
        Dense(latent_dim, 8, selu),
        Dense(8, 8, selu),
        Dense(8, 2, selu)
    ) |> gpu

    crit = Chain(
        Dense(2, 8, relu),
        Dense(8, 8, relu),
        Dense(8, 4, relu),
        Dense(4, 1)
    ) |> gpu

    return (genr, crit)
end

genr, crit = create_models()
opt_genr = Adam(1e-4, (0.9, 0.99))
opt_crit = Adam(1e-4, (0.9, 0.99))
hps = Hyperparameters()
train!(genr, crit, opt_genr, opt_crit, hps)