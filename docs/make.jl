using WGANGP
using Documenter

DocMeta.setdocmeta!(WGANGP, :DocTestSetup, :(using WGANGP); recursive=true)

makedocs(;
    modules=[WGANGP],
    authors="Vincent Molin <vincentmolin@gmail.com> and contributors",
    repo="https://github.com/vincentmolin/WGANGP.jl/blob/{commit}{path}#{line}",
    sitename="WGANGP.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://vincentmolin.github.io/WGANGP.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/vincentmolin/WGANGP.jl",
    devbranch="main",
)
