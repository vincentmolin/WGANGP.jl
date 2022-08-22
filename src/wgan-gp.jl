# Define CUDA gradients to make the lipschitz loss autodiff'able
Zygote.@adjoint CUDA.ones(x...) = CUDA.ones(x...), _ -> map(_ -> nothing, x)
Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)
Zygote.@adjoint CUDA.rand(x...) = CUDA.rand(x...), _ -> map(_ -> nothing, x)
Zygote.@adjoint CUDA.fill(x::Real, dims...) = fill(x, dims...), Δ->(sum(Δ), map(_->nothing, dims)...)

"""
    lipschitz1_gradient_loss(m, x_interpolated, batch_size, data_dims)

Estimates 𝐄ₓ(‖∇ₓD(x)‖₂ - 1)², where x is sampled uniformly on lines between
points from the data distribution and the generators distribution.
"""
function lipschitz1_gradient_loss(m, x_interpolated, batch_size, data_dims)
    # x_size = size(x_true)
    # data_dims = length(x_size) - 1
    # ξ_size = tuple(ones(Int,data_dims)..., batch_size)
    # CUDA.@sync ξ = CUDA.rand(ξ_size...)
    # x_interpolated = ξ .* x_true + (1.0f0 .- ξ) .* x_generated
    _, b = pullback(() -> m(x_interpolated), params(x_interpolated))
    grads = b(CUDA.ones(1, batch_size))
    sqddx = grads[x_interpolated] .^ 2
    mean((sqrt.(sum(sqddx, dims = 1:data_dims)) .- 1.0f0) .^ 2)
end

#function gpu_gradient_loss(m, x_interpolated)
#    mean((sqrt.(sum(abs2, gradient(sum∘m, x_interpolated)[1], dims = (1:ndims(x_interpolated)-1))) .- 1) .^ 2)
#end

"""
    critic_loss(m, x_true, x_generated, x_interpolated, λ, batch_size, data_dims)

WGAN-GP relaxed critic loss with Lagrange multiplier λ.
"""
function critic_loss(m, x_true, x_generated, x_interpolated, λ, batch_size, data_dims)
    gp = lipschitz1_gradient_loss(m, x_interpolated, batch_size, data_dims)
    return mean(m(x_generated)) - mean(m(x_true)) + λ * gp
end

function generator_loss(m, crit, z)
    -mean(crit(m(z)))
end

"""
    interpolate_x(x, y, batch_size, data_dims)

Each coordinate of the interpolation is a random convex combination of x and y.       
"""
function interpolate_x(x_true, x_generated, batch_size, data_dims)
    ξ = CUDA.rand(ones(Int64, data_dims)..., batch_size)
    x_interpolated = ξ .* x_true + (1.0f0 .- ξ) .* x_generated
    x_interpolated
end

Zygote.@nograd interpolate_x

"""
    step_critic!(opt, m, x_true, x_generated; λ = 10.0f0) = loss

A single optimisation step for the critic, with λ gradient penalty factor.
"""
function step_critic!(opt, m, x_true, x_generated; λ = 10.0f0)
    batch_size = size(x_true)[end]
    data_dims = ndims(x_true) - 1
    x_interpolated = interpolate_x(x_true, x_generated, batch_size, data_dims)
    ps = params(m)
    loss, back = pullback(ps) do
        critic_loss(m, x_true, x_generated, x_interpolated, λ, batch_size, data_dims)
    end
    gs = back(1.0f0)
    Flux.update!(opt, ps, gs)
    return loss
end

"""
    step_generator!(opt, m, crit, z) = loss

A single optimisation step for the generator.
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
