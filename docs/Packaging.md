# Building independent SpectralMM packages

Maintain one repository and assemble three independently installable packages.
Edit the canonical sources, not the generated directories. R and Python use the
shared C++ core. Julia defaults to its Julia implementation and offers
`backend=:cpp` explicitly.

## Source ownership

| Component | Canonical source | Generated package |
|---|---|---|
| Shared C++ implementation | `src/native/` | Copied into all three packages |
| R interface and help | `R/`, `man/`, `DESCRIPTION`, `NAMESPACE`, R bridge in `src/` | `dist/R/SpectralMM/` |
| Python interface and build configuration | `bindings/python/spectralmm/`, `pyproject.toml`, `cpp/` | `dist/Python/SpectralMM/` |
| Julia implementation and interface | `src/*.jl`, `deps/`, `Project.toml`, `test/` | `dist/Julia/SpectralMM/` |

The Julia package also includes `examples/data.csv`, which its regression tests
require. Each generated package contains its own C++ sources and needs no files
from the original checkout. R does not require Python or Julia, and Python does
not require R or Julia.

## Assemble and build

Run from the repository root:

```sh
python3 tools/build_packages.py
```

This only assembles source trees; it does not compile or publish anything.
The script checks that the three package versions match, verifies each copied
C++ file against the canonical source, and writes `source-manifest.json` beside
each package directory with SHA-256 hashes. `dist/` is ignored by Git.
Reassembly replaces generated source trees but preserves sibling artifacts.
Unmarked existing language directories and symbolic-link destinations are
rejected. `--output /path/to/directory` selects a different output location.

Build distributions separately:

```sh
python3 tools/build_packages.py --language R --build
python3 -m pip install build
python3 tools/build_packages.py --language Python --build
python3 tools/build_packages.py --language Julia --build
```

Use the same Python interpreter to install `build` and run the Python build.
Artifacts appear under `dist/<language>/artifacts/`: an R source archive, a
Python source distribution and wheel, and a Julia source archive. Python builds
the wheel from its source distribution. R builds require Rcpp and RcppEigen;
source compilation requires a C++17 compiler. Python's isolated build obtains
CMake and pinned Eigen headers automatically and may require network access.

## Install and check R

Install dependencies once in R:

```r
install.packages(c("Rcpp", "RcppEigen"))
```

Then, from the repository root:

```sh
R CMD INSTALL dist/R/SpectralMM
R CMD check --as-cran dist/R/artifacts/SpectralMM_0.1.0.tar.gz
```

```r
library(SpectralMM)
model <- spectralmm_fit(X, y, family="bernoulli")
summary(model)
```

R compiles and loads the core through its own package installation system.
No manual CMake command or development loader is needed.

## Install and check Python

```sh
python3 -m venv /tmp/spectralmm-client
/tmp/spectralmm-client/bin/python -m pip install ./dist/Python/SpectralMM
/tmp/spectralmm-client/bin/python -m unittest discover -s dist/Python/SpectralMM/tests
```

A compatible wheel from `dist/Python/artifacts/` can replace the source directory
in the installation command and avoids local compilation. The package loads
its bundled library; fitting does not rebuild it.

```python
import spectralmm
model = spectralmm.fit(X, y, family="bernoulli")
print(model.summary())
```

## Install and check Julia

In a separate Julia environment, use an absolute package path:

```julia
using Pkg
Pkg.activate(mktempdir())
Pkg.develop(path="/absolute/path/to/SpectralMM/dist/Julia/SpectralMM")
Pkg.test("SpectralMM")

using SpectralMM
model = SpectralMM.fit(X, y; family=:bernoulli)
cpp_model = SpectralMM.fit(X, y; family=:bernoulli, backend=:cpp)
```

Julia 1.12 or later is currently required. The first explicit C++ fit compiles
and caches the core in Julia's scratch space. Pure Julia fitting does not
compile C++. The regression suite exercises both backends and therefore needs
a C++ toolchain. Package-managed build dependencies remain declared even when
only the Julia backend is used.

## Cleanup and retained evidence

Obsolete R and Julia loader shims and the Python `spectralmm_native.py` shim
have been removed. Benchmark scripts now import the public packages. The
explicit `SPECTRALMM_LIBRARY` override remains available to developers.
R's internal S3 class name is retained because method dispatch uses it.

Old installation archives, compiled objects, shared libraries in the source
directory, and build caches were removed. Benchmark results, LaTeX reports,
small-data examples and durable regression tests remain. Research notebooks,
`.RData`, `.Rhistory` and the development `Manifest.toml` are retained locally
but are not copied into the independent package trees.

## Validation of this assembly

On 2026-10-02, on macOS arm64:

- Julia 1.12.6: all 322 regression assertions passed using a separate client
  project and scratch cache. The staged package built its own C++ library.
  Dependency resolution used an existing dependency cache in offline mode.
- Python 3.9: a wheel built from the source distribution was installed with
  NumPy and SciPy into a fresh virtual environment. The Probit smoke test passed
  outside the repository, loading the library from the installed package.
- R 4.4.2: the standalone source archive passed installation, loading, examples,
  compiled-code checks, its smoke test and PDF manual checks. `R CMD check
  --as-cran` reported zero errors, zero warnings and three notes: missing pandoc
  for README conversion, an old system HTML validator, and Apple toolchain
  temporary file `xcrun_db`. System-clock and remote incoming checks were
  disabled for this local run.

Numerical audit evidence is retained in
[the correctness report](../benchmark/native/results/numerical-correctness-audit.md).
Packaging validation does not replace that audit or produce new timing results.

## Publication boundary

These artifacts have not been published to CRAN, PyPI or Julia General.
The locally built wheel targets macOS 26 arm64; Windows and Linux builds still
need platform testing. Resolve the R check environment notes and run complete
submission checks before a CRAN submission.

Use the R archive for R submission and the Python distributions for a Python
release after platform validation. Julia registry publication registers a Git
repository revision containing the Julia package; the canonical repository root
already has that layout. The generated Julia archive is for source delivery,
not a registry submission. Separate language releases do not require three
independently maintained copies of the numerical implementation.
