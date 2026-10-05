using Documenter, SpectralMM
makedocs(
    sitename="SpectralMM",
    modules=[SpectralMM],
    repo=Documenter.Remotes.GitHub("Xunjian-Li", "SpectralMM"),
    checkdocs=:exports,
    checkdocs_ignored_modules=[SpectralMM._CppBackend],
    format=Documenter.HTML(edit_link=nothing,
        repolink="https://github.com/Xunjian-Li/SpectralMM"),
    pages=["Home"=>"index.md", "Installation"=>"installation.md",
           "Julia examples"=>"examples.md", "Modeling interfaces"=>"api.md",
           "Julia API reference"=>"reference.md",
           "Solvers and diagnostics"=>"solvers.md", "Inference"=>"inference.md"],
)
