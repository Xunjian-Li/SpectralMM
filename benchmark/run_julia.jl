# Public entry point for the pure-Julia benchmark.
include(joinpath(@__DIR__, "julia", "run_julia.jl"))
abspath(PROGRAM_FILE) == abspath(@__FILE__) && julia_main(ARGS)
