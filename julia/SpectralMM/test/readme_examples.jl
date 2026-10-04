@testset "README Julia examples" begin
    text = read(joinpath(@__DIR__, "..", "README.md"), String)
    blocks = collect(eachmatch(r"(?ms)^```julia\r?\n(.*?)^```", text))
    @test length(blocks) >= 2
    for (i, block) in enumerate(blocks)
        mktempdir() do sandbox
            script = joinpath(sandbox, "example.jl")
            write(script, block.captures[1])
            scope = Module(Symbol("ReadmeExample", i))
            cd(sandbox) do
                Base.include(scope, script)
            end
            # Evaluate in the newly loaded module (including Julia 1.12 world ages).
            @test Core.eval(scope, :(length(coef(@isdefined(model) ? model : formula_model)) == 4))
            @test Core.eval(scope, :(length(y) == size(X, 1)))
            @test Core.eval(scope, :(all(isfinite, predict(@isdefined(model) ? model : formula_model, X))))
        end
    end
end
