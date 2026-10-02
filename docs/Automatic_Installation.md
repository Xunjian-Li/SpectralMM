# Automatic installation for Python and Julia

Current modeling interfaces are documented in [Statistical_API.md](Statistical_API.md).

The public names are `spectralmm` in Python and `SpectralMM` in Julia. Building
and locating the C++ library are handled internally. A C++17 toolchain is still
required when compiling the C++ backend from source; a compatible Python wheel does not require
local compilation. No published Python index release or Julia registry entry
is claimed. GitHub installation commands apply after these changes are pushed.

## Python

From a local checkout:

```sh
python -m pip install .
```

Or, after publication to GitHub:

```sh
python -m pip install git+https://github.com/Xunjian-Li/SpectralMM.git
```

```python
import numpy as np
import spectralmm

rng = np.random.default_rng(1)
X = rng.normal(size=(100, 3))
y = rng.binomial(1, 0.5, size=100)
fit = spectralmm.fit(X, y, family="bernoulli")
print(fit.summary())
```

The PEP 517 build backend installs CMake as a build dependency, downloads pinned
Eigen headers with SHA-256 verification, compiles the shared core and includes
it in the wheel. NumPy and SciPy are declared runtime dependencies. The package
loads its own bundled library after installation; neither the original checkout
nor `build/native` is required. There is no compilation during fitting.

On macOS, automatic BLAS discovery selects Accelerate. On other systems it uses
an available BLAS, or Eigen operations when BLAS is unavailable. An explicit
build override can require BLAS rather than falling back:

```sh
python -m pip install . -Ccmake.define.SPECTRALMM_USE_BLAS=ON
```

The locally built wheel is tagged for this host's macOS/arm64 platform and is
not a universal binary. Linux/Windows wheel publication needs separate builds
and platform testing; external BLAS dependencies must be handled when preparing
redistributable wheels for those platforms.

## Julia

Julia 1.12 or later is required by the current dependency bounds.

For a local checkout:

```julia
using Pkg
Pkg.develop(path="/path/to/SpectralMM")
```

Or, after publication to GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Xunjian-Li/SpectralMM.git")
```

```julia
using SpectralMM, Random
rng = MersenneTwister(1)
X = randn(rng, 100, 3)
y = Float64.(rand(rng, 100) .< 0.5)
model = SpectralMM.fit(X, y; family=:bernoulli) # Default: pure Julia
model_cpp = SpectralMM.fit(X, y; family=:bernoulli, backend=:cpp)
coef(model)
predict(model, X)
stderror(model)
```

The package manager resolves CMake_jll and Eigen_jll automatically. The first fit with
`backend=:cpp` compiles the core automatically. Pure Julia fitting and package
installation do not compile the C++ core. The result lives in Julia's scratch
cache, outside the source tree. The cache key includes core source/header
contents, build instructions, platform, Eigen artifact and relevant compiler
settings. Later fits and new sessions reuse the library. Failed builds are not
published as complete cache entries. There are no manually supplied include
paths or library paths in the normal workflow.

`SpectralMM.fit(X, y; family=:..., backend=:julia)` defaults to pure Julia;
`backend=:cpp` selects C++. Both support all fourteen families and eight solvers.
The read-only `model.backend` property and printed output identify the backend. Family parameters use `family_options`, for example
`family_options=(theta=4.0,)` or `(tau=0.25, smoothing=0.1)`.
`coef`, `predict`, `stderror`, `vcov` and `confint` are standard accessor methods.
The intercept, inference size cutoff and iteration table follow the other
interfaces.

Existing `SpectralMM.glm(X, y, distribution, link)` and positional-family Julia
methods retain the original Julia solvers and formula support. Packaging does
not silently change their numerical backend. The unified statistical interface accepts formulas and matrices. Case weights and offsets remain unsupported.

## Development compatibility and rebuilding

Legacy `spectralmm_native`, `SpectralMMNative` and R development-loader shims
have been removed. Benchmarks and examples use the installed public packages.
`SPECTRALMM_LIBRARY` remains an explicit developer override; normal users do not
need it. Julia additionally accepts `SPECTRALMM_BUILD_BLAS=ON`, `OFF` or `AUTO`
before building (default AUTO). Unsupported explicit settings raise an error.
Restart a session after changing a loaded library or updating package code.

The standard R package remains independent: it compiles the same canonical
sources under `src/native` using R's build system. See [R_Package.md](R_Package.md).

Build-system references:
[scikit-build-core](https://scikit-build-core.readthedocs.io/en/stable/guide/cmakelists.html)
and the package-managed CMake/Eigen artifacts declared in `Project.toml`.

## Local validation and installation artifacts

On macOS arm64, Python 3.9 and Julia 1.12.6 each passed 224 installed-interface
comparisons: fourteen families, eight solvers and dense/CSC matrices, using the
saved 100-by-3 examples. Coefficients and standard errors agreed with the saved
reference results within the specified tolerances; the maximum coefficient
difference was 2.66e-8. Convergence, inference and final iteration diagnostics
were checked. These are correctness checks, not new performance benchmarks.

The Python source distribution was used to build a wheel, which was installed
into a fresh virtual environment and checked outside the repository. A clean
Julia source checkout without a precompiled library built automatically into a
separate scratch cache; a new Julia session reused that library without changing
its modification time. The original Julia solver API was also checked.

Generated Python distributions are in `dist/Python/artifacts/`. Install a compatible
local wheel with:

```sh
python -m pip install dist/Python/artifacts/spectralmm-0.1.0-py3-none-macosx_26_0_arm64.whl
```

The source archive in the same directory is suitable for a source installation
on systems with the required compiler. Linux and Windows have not been tested
in this validation run. The R source distribution is in `dist/R/artifacts/`.

## Backend API compatibility

Earlier development versions routed two-argument keyword `fit` calls to C++.
Those calls now default to Julia; add `backend=:cpp` to preserve C++ execution.
Positional-family methods and `glm` continue to use Julia. The old development shims have been removed; use the installed public package. The public matrix defaults are documented
in the repository README; solver-specific advanced options may differ.

The two backends need not have identical convergence histories. The previously
reported Probit and sparse-CHO inference discrepancies on the saved small
examples were resolved by the 2026-10-01 numerical fixes. Inspect inference
status before requesting standard errors. Evidence and scope are recorded in
`benchmark/native/results/numerical-correctness-audit.md`.

Maintainers can run the retained public-interface regression suite with
`julia --project=. test/runtests.jl` or `using Pkg; Pkg.test()` after dependency
installation. The C++ tests require the C++ toolchain on their first run.

Independent package assembly and checks are documented in [Packaging.md](Packaging.md).
