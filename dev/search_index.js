var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = WGANGP","category":"page"},{"location":"#WGANGP","page":"Home","title":"WGANGP","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for WGANGP.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [WGANGP]","category":"page"},{"location":"#WGANGP.critic_loss-NTuple{7, Any}","page":"Home","title":"WGANGP.critic_loss","text":"critic_loss(m, x_true, x_generated, x_interpolated, λ, batch_size, data_dims)\n\nWGAN-GP relaxed critic loss with Lagrange multiplier λ.\n\n\n\n\n\n","category":"method"},{"location":"#WGANGP.interpolate_x-NTuple{4, Any}","page":"Home","title":"WGANGP.interpolate_x","text":"interpolate_x(x, y, batch_size, data_dims)\n\nEach coordinate of the interpolation is a random convex combination of x and y.       \n\n\n\n\n\n","category":"method"},{"location":"#WGANGP.lipschitz1_gradient_loss-NTuple{4, Any}","page":"Home","title":"WGANGP.lipschitz1_gradient_loss","text":"lipschitz1_gradient_loss(m, x_interpolated, batch_size, data_dims)\n\nEstimates 𝐄ₓ(‖∇ₓD(x)‖₂ - 1)², where x is sampled uniformly on lines between points from the data distribution and the generators distribution.\n\n\n\n\n\n","category":"method"},{"location":"#WGANGP.step_critic!-NTuple{4, Any}","page":"Home","title":"WGANGP.step_critic!","text":"step_critic!(opt, m, x_true, x_generated; λ = 10.0f0) = loss\n\nA single optimisation step for the critic, with λ gradient penalty factor.\n\n\n\n\n\n","category":"method"},{"location":"#WGANGP.step_generator!-NTuple{4, Any}","page":"Home","title":"WGANGP.step_generator!","text":"step_generator!(opt, m, crit, z) = loss\n\nA single optimisation step for the generator.\n\n\n\n\n\n","category":"method"}]
}
