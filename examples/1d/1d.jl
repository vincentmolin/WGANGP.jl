using WGANGP
using Flux

n_hidden = 4
n_critic = 5
n_batch = 64
n_steps = 10
λ = 0.1f0

next_batch(n) = 0.5f0 .* randn(Float32, 1, n) .+ 2.0f0
any_nan_weights(m) = any(ws -> any(isnan, ws), Flux.params(m))

function fit_wgan!(opt_generator, opt_critic, generator, critic, next_batch; n_steps = n_steps, n_critic = n_critic, callback = (_...) -> nothing)
    critic_ls = []
    generator_ls = []
    for i_step in 1:n_steps
        println("Iter ",i_step,"/",n_steps)
        for _ in 1:n_critic
            x_true = next_batch(n_batch)
            z = randn(Float32, 1, n_batch)
            x_generated = generator(z)
            l = step_critic!(opt_critic, critic, x_true, x_generated; λ = λ)
            if any_nan_weights(critic)
                println("Critic weights NaNed at step ", i_step)
                return i_step, (g, critic)
            end
            push!(critic_ls, l)
        end
        z = randn(Float32, 1, n_batch)
        l = step_generator!(opt_generator, generator, critic, z)
        push!(generator_ls, l)
        callback(i_step, generator, critic, generator_loss, critic_loss)
    end
    return true, generator_ls, critic_ls
end

generator = Chain( Dense(1 => n_hidden, swish), Dense(n_hidden => 1) )
critic = Chain( Dense(1 => n_hidden, swish), Dense(n_hidden => 1) )

opt_generator = Adam(5e-4,(0.5,0.9))
opt_critic = Adam(5e-4,(0.5,0.9))

r = fit_wgan!(opt_generator, opt_critic, generator, critic, next_batch; n_steps = 1000)


z = randn(Float32, 1, 10_000)
Flux.mean(generator(z))

using CairoMakie
CairoMakie.inline!(true)
density(vec(generator(randn(Float32,1,1000))))

ts = reshape(collect(LinRange(-10f0,10f0,1000)),1,:)
ys = critic(ts)
lines(vec(ts),vec(ys))