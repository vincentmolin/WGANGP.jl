# learn 1 gaussian
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
using Zygote

function sample_gaussian(n; mean = 4.0, sd = 0.5)
    mean .+ sd .* randn(Float32, 1, n)
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

    # Tensorboard logger
    tblog = TBLogger("logs/$runid")

    train_steps = 1
    loss_genr = 0.0

    
    data_batches = [data[:, r] |> device for
    r in partition(1:size(data, 2), hps.batch_size)]
    
    p = Progress(hps.epochs * length(data_batches), "Training...")

    z = CUDA.zeros(latent_dim, hps.batch_size)

    with_logger(tblog) do
        for epoch in 1:hps.epochs
            for i in randperm(length(data_batches))
                CUDA.@sync CUDA.randn!(z)
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


function create_models(latent_dim = 1; device = gpu)
    genr = Chain(
        Dense(latent_dim => 4, relu),
        Dense(4 => 1)
    ) |> device

    crit = Chain(
        Dense(1 => 4, relu),
        Dense(4 => 1)
    ) |> device

    return (genr, crit)
end

@with_kw struct Hyperparameters
    batch_size::Int = 64
    critic_train_factor::Int = 5
    λ_reg::Float32 = 0.1f0
    epochs::Int = 5
    verbose_freq::Int = 20
    sample_freq::Int = 1000
    gpu::Bool = true
end

genr, crit = create_models()
opt_genr = Adam(1e-4, (0.5, 0.9))
opt_crit = Adam(1e-4, (0.5, 0.9))
hps = Hyperparameters()
data = sample_gaussian(128 * 100)
train!(genr, crit, opt_genr, opt_crit, data, hps)


critic_train_factor = 5
batch_size = 64
λ_reg = 0.1f0

for _ in 1:10
    for _ in 1:critic_train_factor
        CUDA.@sync z = CUDA.randn(1, batch_size)
        x_generated = genr(z)
        x_true = cu(sample_gaussian(batch_size))
        loss_crit = step_critic!(opt_crit, crit, x_true, x_generated; λ = λ_reg)
    end
    CUDA.@sync z = CUDA.randn(1, batch_size)
    loss_genr = step_generator!(opt_genr, genr, crit, z)
end


scatter(generated[1,:], generated[2,:])