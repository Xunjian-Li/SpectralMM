# R package installation and use

Current modeling interfaces are documented in [Statistical_API.md](Statistical_API.md).

Install the current GitHub source:

```r
install.packages("remotes")
remotes::install_github("Xunjian-Li/SpectralMM")
library(SpectralMM)

set.seed(1)
X <- matrix(rnorm(300), 100, 3)
y <- rbinom(100, 1, plogis(0.2 + X %*% c(0.4, -0.3, 0.2)))
fit <- spectralmm_glm(X, y, family = binomial(link = "logit"))
summary(fit)
coef(fit)
predict(fit, X, type = "response")
vcov(fit)
confint(fit)
```

For a local checkout, install once with:

```r
remotes::install_local("/path/to/SpectralMM", upgrade = "never")
```

Installation automatically resolves R dependencies and compiles the shared C++
core using R's build system. Subsequent sessions only need `library(SpectralMM)`.
There is no CMake, manual shared-library path, `source()`, or `spectralmm_load()`
step in the package workflow. Source installation still requires a compatible
C++17 toolchain (for example Command Line Tools on macOS or the matching Rtools
on Windows). Binary distribution could remove that requirement for end users;
this repository currently provides a source package.

All fourteen families and eight solvers remain available. The matrix interface,
intercept convention, inference cutoff of 50 parameters, and iteration output
are unchanged. Use `?spectralmm_fit` and `?spectralmm_methods` for installed help.
Formula input is supported by the statistical API. Observation weights and offsets remain unsupported.

## Build layout

The repository root is both an R package and the existing Julia project. The
R source archive excludes Julia sources, notebooks, local binaries and benchmark
artifacts through `.Rbuildignore`.

- `DESCRIPTION` and `NAMESPACE`: dependencies, exports and native registration.
- `R/`: canonical R interface and generated Rcpp entry point.
- `src/r_bridge.cpp`: canonical Rcpp bridge.
- `src/native/`: the single shared numerical core and C API headers.
- `src/Makevars` and `src/Makevars.win`: C++17, RcppEigen headers, R BLAS linking.
- `cpp/CMakeLists.txt`: standalone build of the same core for Python/native Julia.
- Benchmarks now load the installed package with `library(SpectralMM)`; the old
  development loader has been removed.

Rcpp and RcppEigen are declared package dependencies; users do not need to find
Eigen headers manually. The installed R package links R's configured BLAS using
`BLAS_LIBS` and `FLIBS`. Its BLAS need not be the Apple Accelerate selected by the
standalone macOS CMake build. Sparse products still use Eigen. The R package
build does not depend on an existing `build/native` library.

The original Julia numerical implementation remains available. This packaging
change does not silently redirect its high-level API to the native wrapper.

## Maintainer workflow

After changing Rcpp-exported function signatures, regenerate registration:

```r
Rcpp::compileAttributes(".")
```

Build/check using the standard R commands:

```sh
R CMD build .
R CMD check SpectralMM_0.1.0.tar.gz
```

Do not commit temporary object files or platform binaries. A clean source
archive contains everything needed to compile on installation, apart from the
declared R dependencies and the user's compiler. Installation on the current
macOS host is validated locally; Linux/Windows and CRAN submission require their
own platform checks before a release claim.

The packaging follows the native-code, Makevars and registration mechanisms in
[Writing R Extensions](https://cran.r-project.org/doc/manuals/r-release/R-exts.html).

## Local validation

In the earlier packaging validation on macOS arm64 with R 4.4.2, a clean source archive installed successfully using
R CMD INSTALL, and R CMD check completed with Status: OK (zero errors, warnings
or notes), including native-code checks, examples and the PDF reference manual.
The archive excludes the standalone CMake build and Julia source files.

The installed package passed 224 combinations of fourteen families, eight
solvers and dense/CSC input. Coefficients and standard errors agreed with the
saved reference results within 1e-5; maximum coefficient difference was 2.66e-8.
Additional checks covered inference gating/override, registered native entry
points, standard S3 generics, explicit intercepts, rejected initial-vector
lengths, and unchanged caller inputs. The optional legacy loader and standalone
Python core also passed smoke checks after the shared source relocation.

The current independent source archive is retained at
[dist/R/artifacts/SpectralMM_0.1.0.tar.gz](../dist/R/artifacts/SpectralMM_0.1.0.tar.gz).
The 2026-10-02 standalone `--as-cran` check has zero errors, zero warnings and
three environment-related notes; see [Packaging.md](Packaging.md) for scope
and the remaining checks. The legacy loader mentioned above has been removed.
The original numerical benchmark files remain historical measurements of their
recorded builds; packaging does not replace their timing results.

The user-facing `remotes::install_local()` workflow also completed successfully.
The package and RcppEigen dependency were installed into the personal R library
`~/Library/R/arm64/4.4/library`, and a fresh R session outside the checkout
loaded SpectralMM and fitted a Logistic model without custom library paths.
