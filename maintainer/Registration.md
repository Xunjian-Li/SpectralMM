# Julia registration and documentation maintenance

The registered package lives in `julia/SpectralMM`. It is a self-contained
snapshot of the canonical Julia and shared C++ sources, generated with:

```sh
python3 tools/sync_julia_package.py
python3 tools/sync_julia_package.py --check
```

Edit canonical sources first. Commit the regenerated tree in the same commit.
CI rejects drift. No symlinks or parent-directory includes are used, so the
registered subtree installs without the rest of this multilingual repository.
Internal maintainer notes are excluded from staged R/Python/Julia distributions.

After testing and author review, trigger on the intended commit:

```text
@JuliaRegistrator register subdir=julia/SpectralMM
```

The existing registration must be updated to this subdirectory; generating it
locally does not update General. Do not rewrite the published v0.1.0 tag or PyPI
artifact. TagBot uses `SpectralMM-vVERSION` tags for subsequent subpackage
registrations, avoiding collisions with multilingual release tags.

Before publishing documentation, select GitHub Actions as the repository's
Pages source. Enable the repository in Codecov for coverage uploads using
GitHub Actions OIDC. A badge is not evidence of coverage until an upload succeeds.
The documentation workflow builds on PRs and deploys only from main.

The README is an assisted draft. The maintainer must personally review and
rewrite it as needed based on their understanding before responding to the
registry reviewer. Do not describe automated writing as handwritten work.
JLL distribution is a future improvement: build SpectralMM_jll through Yggdrasil
to avoid requiring a local compiler and unconditional CMake/Eigen artifacts.
No JLL migration is part of this review.
