# Prepare an independent benchmark environment using the local Julia package.
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, "..", "julia"))
Pkg.instantiate()
