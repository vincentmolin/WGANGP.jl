module WGANGP

using CUDA
using Zygote
using Flux
using Flux: mean, pullback, params

include("wgan-gp.jl")
export lipschitz1_gradient_loss, critic_loss, 
       generator_loss, step_critic!, step_generator!

end
