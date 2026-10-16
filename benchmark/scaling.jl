# Compatibility entry point: retains the original CSV destination.
isdefined(@__MODULE__, :run_julia) || include("run_julia.jl")
results = run_julia(include(joinpath(@__DIR__, "julia_settings.jl")),
                   @__DIR__, "dense"; legacy=true)["dense"]
