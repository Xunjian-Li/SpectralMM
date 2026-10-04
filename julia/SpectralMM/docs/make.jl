using Documenter, SpectralMM
makedocs(
    sitename="SpectralMM",
    modules=[SpectralMM],
    remotes=nothing,
    checkdocs=:none,
    format=Documenter.HTML(edit_link=nothing,
        repolink="https://github.com/Xunjian-Li/SpectralMM"),
    pages=["Home"=>"index.md", "Installation"=>"installation.md",
           "Julia examples"=>"examples.md", "Modeling interfaces"=>"api.md",
           "Solvers and diagnostics"=>"solvers.md", "Inference"=>"inference.md"],
)
