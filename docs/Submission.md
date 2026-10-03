# SpectralMM 0.1.0 release procedure

The canonical sources are on `main`. Release the exact commit that passes the
independent package checks; do not move a published version tag.

## Validation and artifacts

- Julia, R and Python versions must all be 0.1.0; the package license is GPL-3.
- Run `.github/workflows/package-checks.yml`: all nine platform jobs and the
  cross-language comparison must pass for the release commit.
- Build independent packages with `tools/build_packages.py`; retain source
  manifests and artifact SHA-256 checksums with the GitHub release.
- Verify clean installation, public and legacy interfaces, formula/matrix
  equivalence and the shared PseudoHuber default delta=1.
- The version tag is `v0.1.0`. Registry upload/submission and acceptance are
  separate events; record actual outcomes on the release, not anticipated ones.

## Python / PyPI

Run the manual `publish-pypi.yml` workflow at the tested tag. It checks that the
same commit passed package checks, builds a source distribution, validates it
with twine, and uploads through the configured PyPI Trusted Publisher.
No API token belongs in this repository. The initial release uploads the source
distribution only; source installation requires a C++17 compiler. Locally built
wheels require platform portability review before public redistribution.

## Julia / General

Trigger the installed Registrator with `@JuliaRegistrator register` on the exact
release commit. Review the resulting General registry pull request and its
checks. The root Julia package includes both backends; Julia remains the default.
Registry submission is not acceptance. Do not upload a Julia source tarball as
an alternative to registration.

## R / CRAN

Submit the R CMD build artifact `dist/R/artifacts/SpectralMM_0.1.0.tar.gz` through
https://cran.r-project.org/submit.html. Maintainer: Xun-Jian Li,
xunjianli@ucla.edu. Report actual R CMD check results and relevant notes.
The maintainer must confirm the submission through CRAN's confirmation email.
A submitted or confirmed package is not yet a CRAN-accepted package.

The R package compiles its bundled C++ core using R's build system and requires
neither Julia nor Python nor a manual CMake build. Describe it as a first
submission and provide the cross-platform checks for the tagged commit.
