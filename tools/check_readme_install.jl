# Execute the README installation in a fresh project, then every fitting block.
# Only the repository URL is redirected to the checked-out commit under test.
using Test
root = normpath(joinpath(@__DIR__, ".."))
repository = isempty(ARGS) ? root : abspath(only(ARGS))
readme = read(joinpath(root, "README.md"), String)
blocks = collect(eachmatch(r"(?ms)^```julia-install\r?\n(.*?)^```", readme))
@test length(blocks) == 1
install = replace(only(blocks).captures[1],
    "\"https://github.com/Xunjian-Li/SpectralMM\"" => repr(repository))
mktempdir() do sandbox
    write(joinpath(sandbox, "Project.toml"), "[deps]\n")
    script = joinpath(sandbox, "install_and_examples.jl")
    write(script, install * "\nimport Pkg\nPkg.add(\"Distributions\")\nusing Test\ninclude(" *
          repr(joinpath(root, "test", "readme_examples.jl")) * ")\n")
    withenv("JULIA_LOAD_PATH" => join(["@", "@stdlib"], Sys.iswindows() ? ';' : ':')) do
        run(`$(Base.julia_cmd()) --startup-file=no --project=$sandbox $script`)
    end
end