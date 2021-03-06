using Flux
using CUDA

Random.seed!(123)

@testset "train steps" begin
    crit = Chain(
        # Input: (64, 64, 1, :)
        Conv((4, 4), 1 => 64, selu, stride = 2, pad = SamePad()),   # -> (32, 32, 64, :)
        Conv((4, 4), 64 => 128, selu, stride = 2, pad = SamePad()), # -> (16, 16, 128, :)
        Conv((5, 5), 128 => 256, selu, stride = 2, pad = SamePad()),   # -> (8, 8, 256, :)
        Flux.flatten, # -> (8*8*256, :)
        Dropout(0.2),
        Dense(8 * 8 * 256, 1)
    ) |> gpu
    id = 64
    bs = 32
    opt_crit = ADAM(0.0002, (0.5, 0.9))
    x_t = cu(randn(Float32, id, id, 1, bs))
    x_f = cu(randn(Float32, id, id, 1, bs))
    @test step_critic!(opt_crit, crit, x_t, x_f, bs) isa Float32
end