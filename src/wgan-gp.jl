# Define gradients for rand and ones to make the lipschitz loss autodiff'able
Zygote.@adjoint CUDA.ones(x...) = CUDA.ones(x...), _ -> map(_ -> nothing, x)
Zygote.@adjoint CUDA.rand(x...) = CUDA.rand(x...), _ -> map(_ -> nothing, x)

"""
    lipschitz1_gradient_loss(m, x_true, x_generated)
Estimates 𝐄ₓ(‖∇ₓD(x)‖₂ - 1)², where x is sampled uniformly on lines between
points from the data distribution and the generators distribution
"""
function lipschitz1_gradient_loss(m, x_true, x_generated)
    x_size = size(x_true)
    batch_size = x_size[end]
    data_dims = length(x_size) - 1
    ξ_size = tuple(ones(Int,data_dims)..., batch_size)
    ξ = CUDA.rand(ξ_size...)
    x_interpolated = ξ .* x_true + (1.0f0 .- ξ) .* x_generated
    _, b = pullback(() -> m(x_interpolated), params(x_interpolated))
    grads = b(CUDA.ones(1, batch_size))
    sqddx = grads[x_interpolated] .^ 2
    mean((sqrt.(sum(sqddx, dims = (1:data_dims))) .- 1.0f0) .^ 2)
end

"""
    critic_loss(m, x_true, x_generated, batch_size, λ)
WGAN-GP relaxed critic loss with lagrange multiplier λ
"""
function critic_loss(m, x_true, x_generated, λ)
    gp = lipschitz1_gradient_loss(m, x_true, x_generated)
    return mean(m(x_generated)) - mean(m(x_true)) + λ * gp
end

function generator_loss(m, crit, z)
    -mean(crit(m(z)))
end

"""
    step_critic!(opt, m, x_true, x_generated; λ = 10.0f0)
A single optimisation step for the critic, with λ gradient penalty factor.
"""
function step_critic!(opt, m, x_true, x_generated; λ = 10.0f0)
    ps = params(m)
    loss, back = pullback(ps) do
        critic_loss(m, x_true, x_generated, λ)
    end
    gs = back(1.0f0)
    Flux.update!(opt, ps, gs)
    return loss
end

"""
A single optimisation step for the generator
"""
function step_generator!(opt, m, crit, z)
    ps = params(m)
    loss, back = pullback(ps) do
        generator_loss(m, crit, z)
    end
    gs = back(1.0f0)
    Flux.update!(opt, ps, gs)
    return loss
end


# Alternatively, this could be done outside the pullback in step_critic! 
# function interpolate_x(x_true, x_generated, batch_size)
#    ξ = CUDA.rand(1, 1, 1, batch_size)
#    x_interpolated = ξ .* x_true + (1.0f0 .- ξ) .* x_generated
# end
# Zygote.@nograd interpolate_x