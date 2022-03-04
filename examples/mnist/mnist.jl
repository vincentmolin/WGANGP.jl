# Somewhat scrappy reference MNIST implementation.

using WGANGP
using Flux
using CUDA
using Base.Iterators: partition
using Plots, Images
using Printf
using Dates
using ProgressMeter
using TensorBoardLogger
using Logging
using BSON: @save, @load
using Parameters: @with_kw
using MLDatasets

@with_kw struct Hyperparameters
    batch_size::Int = 16
    critic_train_factor::Int = 5
    epochs::Int = 100
    verbose_freq::Int = 20
    sample_freq::Int = 1000
    gpu::Bool = true
end

function create_montage(genr, z, n, m; skip_batchnorm = true, kws...)
    skip_batchnorm && @eval Flux.istraining() = false
    out_tensor = cpu(genr(z))
    skip_batchnorm && @eval Flux.istraining() = true
    img_tensor = Gray.(clamp.(out_tensor, 0.0f0, 1.0f0))
    imgs = [img_tensor[:, :, 1, i] for i in 1:n*m]
    return Plots.plot(
        mosaicview(imgs; nrow = n, fillvalue = 1, npad = 5);
        title = "Generated images", size = (600, 600),
        xaxis = false, yaxis = false, axis = nothing,
        kws...
    )
end

# TODO/WARN: Actually doesn't mutate genr and crit if they live on CPU when
# this function is called, so make sure to catch them when they come home from the gym.
function train!(genr, crit, opt_genr, opt_crit, data_tensor, hps = Hyperparameters(); run_prefix = nothing, runid = nothing)

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
    data = [data_tensor[:, :, :, r] |> device for
            r in partition(1:size(data_tensor, 4), hps.batch_size)]
    p = Progress(hps.epochs * length(data), "Training...")

    with_logger(tblog) do
        for epoch in 1:hps.epochs
            for i = 1:length(data)

                z = CUDA.randn(latent_dim, hps.batch_size)
                x_generated = genr(z)
                x_true = data[i]

                loss_crit = step_critic!(opt_crit, crit, x_true, x_generated, hps.batch_size)

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
                    # Save generated images
                    static_output_montage = create_montage(genr, fixed_noise, 5, 5; size = (800, 800))
                    png(static_output_montage, @sprintf("%s/static_step_%06d.png", imspath, train_steps))
                    sample_output_montage = create_montage(genr, CUDA.randn(latent_dim, 25), 5, 5; size = (800, 800))
                    png(sample_output_montage, @sprintf("%s/sample_step_%06d.png", imspath, train_steps))
                    @info "Generated" static = static_output_montage sampled = sample_output_montage
                end
                train_steps += 1
            end
        end
    end

    return genr, crit
end


function create_mnist_models(latent_dim = 100)
    genr = Chain(
        Dense(latent_dim, 7 * 7 * 256, relu),
        x -> reshape(x, 7, 7, 256, :),
        ConvTranspose((5, 5), 256 => 128; stride = 1, pad = 2),
        BatchNorm(128, relu),
        ConvTranspose((4, 4), 128 => 64; stride = 2, pad = 1),
        BatchNorm(64, relu),
        ConvTranspose((4, 4), 64 => 1, sigmoid; stride = 2, pad = 1)
    ) |> gpu

    crit = Chain(
        Conv((5, 5), 1 => 64, selu; stride = 2, pad = 1),
        Conv((5, 5), 64 => 128, selu; stride = 2, pad = 1),
        Conv((5, 5), 128 => 256, selu; stride = 2, pad = 1),
        x -> reshape(x, 1024, :),
        Dense(1024, 1)
    ) |> gpu

    return (genr, crit)
end

function prepare_mnist_data()
    ims = MNIST.traintensor(Float32)
    return Flux.unsqueeze(ims, 3)
end

genr, crit = create_mnist_models()
opt_genr = ADAM(1e-4, (0.9, 0.99))
opt_crit = ADAM(1e-4, (0.9, 0.99))
data = prepare_mnist_data()
hps = Hyperparameters()
train!(genr, crit, opt_genr, opt_crit, data, hps)