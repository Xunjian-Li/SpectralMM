# SpectralMM 0.1.0 submission preparation

Status: prepared locally; no registry submission or upload has been made.

## Required before submission

1. Commit and push the reviewed canonical sources to the public GitHub repository.
   Do not commit local research session files or credentials.
2. Run `.github/workflows/package-checks.yml`. It checks independent source
   packages on Linux, macOS and Windows. This workflow has not yet run remotely;
   resolve failures and review notes before publishing.
3. Rebuild final distributions from the exact release revision. Retain the
   source manifests and artifact checksums with the release evidence.
4. Verify name availability and account ownership directly on each platform.
   Automated web lookups in this session could not establish availability.

## R / CRAN

Upload `dist/R/artifacts/SpectralMM_0.1.0.tar.gz` using the
[CRAN submission form](https://cran.r-project.org/submit.html).
Use Xun-Jian Li and xunjianli@ucla.edu as the maintainer. Confirm the submission
through the email sent to that address. Review and accept the
[CRAN policy](https://cran.r-project.org/web/packages/policies.html).

Draft submission comments (replace the validation paragraph after remote checks):

> This is the first submission of SpectralMM. It provides spectral majorization
> solvers for generalized linear, asymmetric and robust regression models, with
> dense and sparse designs and optional Wald inference. The R source package
> compiles its included C++ core with R's build system and does not require
> Julia, Python or a manual CMake build.
>
> Local checks used R 4.4.2 on macOS arm64. There were no errors or warnings.
> Three notes concerned missing pandoc, an outdated system HTML validator,
> and an Apple toolchain temporary file. System-clock and remote incoming checks
> were disabled in that local run. Cross-platform validation is pending.

Do not describe this draft as a clean cross-platform check. CRAN normally
requires operation on at least two major R platforms.

## Python / PyPI

Build with `python tools/build_packages.py --language Python --build` after
installing `build` and `twine` into the same environment. Validate:

```sh
python -m twine check dist/Python/artifacts/*
```

First test the release workflow against TestPyPI if desired. TestPyPI and PyPI
accounts and credentials are separate. Once final files and account access are
ready, the actual production upload command is:

```sh
python -m twine upload dist/Python/artifacts/*
```

This command publishes files. Never put an API token in repository files or chat;
use a secure credential store or trusted publishing. The current local wheel is
macOS 26 arm64 only. Source distributions need a compiler; CI-generated wheels
need platform dependency/portability review before redistribution. In particular,
a plain Linux build is not automatically a portable manylinux wheel.
See the [official packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

## Julia / General

The repository root is the Julia package. Push the release sources and ensure
`Project.toml` version and dependency bounds are correct. Install/enable
[Registrator](https://github.com/JuliaRegistries/Registrator.jl) for the repository.
Trigger registration with `@JuliaRegistrator register` on the exact release
commit, then review the resulting General registry pull request and address its
checks. Do not upload the Julia tarball as a registry submission.

Julia remains pure Julia by default; `backend=:cpp` is explicit. Both backends
are included in this single Julia package. Check the current
[General automatic merge guidelines](https://juliaregistries.github.io/RegistryCI.jl/stable/guidelines/)
before registration. A successful local package test does not establish
registry acceptance, dependency availability or name approval.
