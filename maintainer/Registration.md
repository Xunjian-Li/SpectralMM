# Package maintenance and Julia registration

Maintain each language directly in `R/`, `julia/` and `python/`. Julia sources
are no longer generated from another source tree. The canonical shared C++
implementation lives in `cpp/src/native/` and its build configuration in `cpp/`.

After editing shared C++ sources, run:

```sh
python3 tools/sync_package_sources.py
python3 tools/sync_package_sources.py --check
```

Commit the synchronized copies together. The script only synchronizes shared
C++ files, licenses and test fixtures; it never overwrites language source code.
Each package is independently installable without parent-directory includes.
`tools/build_packages.py` stages these directories and builds distribution files.

After checks and author review, request registration on the intended commit:

```text
@JuliaRegistrator register subdir=julia
```

Moving the directory locally does not change the General registry. Registrator
must use the new `julia` subdirectory for the next registration. TagBot uses the
same subdirectory. Leave existing published tags and artifacts untouched.

Documentation builds from root `docs/` against `julia/`. CI checks the shared
source snapshots, tests each package and uploads Julia coverage. The maintainer
must review README changes and their examples before responding to reviewers.
A future SpectralMM_jll package could remove Julia's local C++ compiler requirement.
