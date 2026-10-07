# SpectralMM for R

Fit generalized linear models and robust or asymmetric regression with the shared
SpectralMM C++ core. Requires R 4.0 or later and a C++17 compiler for source installation.

```r
install.packages("remotes")
remotes::install_github("Xunjian-Li/SpectralMM", subdir="R")
library(SpectralMM)
model <- spectralmm_glm(mpg ~ wt + hp, data=mtcars, family=gaussian())
summary(model)
predict(model, newdata=mtcars[1:5, ])
```

For a local checkout, use `remotes::install_local("R")` from the repository root,
or install Rcpp and RcppEigen and run `R CMD INSTALL R` in a terminal.
Edit R code in `R/` and help files in `man/`. Shared C++ files in `src/native/`
are synchronized from the repository's root `cpp/src/native/` directory.

See `help(package="SpectralMM")` and the
[full documentation](https://xunjian-li.github.io/SpectralMM/).
