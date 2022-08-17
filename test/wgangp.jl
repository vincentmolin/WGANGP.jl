using Flux
using CUDA

if CUDA.functional()
    
    Random.seed!(123)
    
    @testset "train steps 3d data" begin
        crit = Chain(
            # Input: (64, 64, 1, :)
            Conv((4, 4), 1 => 2, selu, stride = 2, pad = SamePad()),   # -> (32, 32, 2, :)
            Conv((4, 4), 2 => 4, selu, stride = 2, pad = SamePad()), # -> (16, 16, 4, :)
            Conv((5, 5), 4 => 8, selu, stride = 2, pad = SamePad()),   # -> (8, 8, 8, :)
            Flux.flatten, # -> (8*8*8, :)
            Dropout(0.2),
            Dense(8 * 8 * 8, 1)
        ) |> gpu
        image_dim = 64
        batch_size = 32
        opt_crit = Adam(0.0002, (0.5, 0.9))
        x_true = cu(randn(Float32, image_dim, image_dim, 1, batch_size))
        x_generated = cu(randn(Float32, image_dim, image_dim, 1, batch_size))
        @test step_critic!(opt_crit, crit, x_true, x_generated) isa Float32
    end
    
    @testset "train steps 1d data" begin
        crit = Chain(
            Dense(4, 8, relu),
            Dense(8, 4, relu),
            Dropout(0.2),
            Dense(4, 1)
        ) |> gpu
        data_dim = 4
        batch_size = 8
        opt_crit = Adam(0.0002, (0.5, 0.9))
        x_true = cu(randn(Float32, data_dim, batch_size))
        x_generated = cu(randn(Float32, data_dim, batch_size))
        @test step_critic!(opt_crit, crit, x_true, x_generated) isa Float32
    end
    
else
    @info "No GPU detected, skipping test."
end