# Registration review changes (2026-10-04)

## Scope

Documentation, runnable examples, CI configuration and a self-contained Julia
registration subtree. No numerical implementation changed. Julia and standard-library compatibility
bounds now admit Julia 1.10 after a successful full-suite run. The published v0.1.0 tag and PyPI distribution are untouched.

## Local checks

- Existing Julia suite: 517 assertions passed under Julia 1.12.6, including both
  backends. The first combined run exposed a world-age issue in the new README
  harness; it was corrected without changing package code.
- Corrected README harness: 7 assertions passed. Each Julia fitting block is
  extracted dynamically and executed in its own module and temporary directory.
  The separate clean-client test also extracts and executes the installation
  block, redirecting only its repository URL to the checkout under test.
- Documenter: six pages built; logistic, sparse and Pseudo-Huber examples execute
  during the build. These examples construct their own data.
- Coverage collection produced 14 Julia source coverage files in the package
  subtree. Remote Codecov reporting has not yet run for these changes.
- R, Python and Julia staging succeeded; internal release documents are excluded.
- Snapshot synchronization, workflow YAML parsing and local Markdown links checked.

## Before resubmission

The author must review and finalize the assisted README draft. Push the reviewed
changes and run the workflows on GitHub. Pages is configured to use GitHub Actions. Verify successful documentation
deployment and Codecov/OIDC uploads. Update the registration with `subdir=julia/SpectralMM`; the current General
PR is not changed by these local edits. See Registration.md for the commands.

Julia 1.10.12: the complete suite passed all 524 assertions with coverage
collection and both backends. README installation and its 7 fitting assertions
also passed in a fresh client project with `JULIA_LOAD_PATH=@:@stdlib`.
The CI matrix covers Julia 1.10, 1.11, 1.12 and latest stable on all three OSes.

Prebuilt JLL distribution remains a future improvement.
