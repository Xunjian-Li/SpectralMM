# Documentation

The user manual starts at [src/index.md](src/index.md).

Build the Documenter site from the repository root:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path="julia"); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Serve `docs/build` with a local HTTP server to preview it. The LaTeX documents
are supplementary reports; the Documenter manual is the current API guide.
