# Julia API reference

Entries below are generated from the public API docstrings. Use the `source`
links to inspect each implementation. Start with [modeling interfaces](api.md)
for examples in Julia, Python and R.

```@meta
CurrentModule = SpectralMM
```

```@index
Pages = ["reference.md"]
```

## Model fitting

```@docs
SpectralMM.glm
SpectralMM.fit
SpectralMM.SpectralMMControl
```

## Results and prediction

```@docs
SpectralMM.StatisticalResult
SpectralMM.coef
SpectralMM.predict
SpectralMM.fitted
SpectralMM.residuals
SpectralMM.nobs
SpectralMM.diagnostics
```

## Inference and fit statistics

```@docs
SpectralMM.vcov
SpectralMM.stderror
SpectralMM.confint
SpectralMM.coeftable
SpectralMM.deviance
SpectralMM.loglikelihood
```

## Residual-loss specifications

```@docs
SpectralMM.PseudoHuber
SpectralMM.Expectile
SpectralMM.SmoothQuantile
SpectralMM.StudentT
```

## Advanced Julia solver interface

```@docs
SpectralMM.spectral_mm
SpectralMM.SpectralOptions
SpectralMM.InnerOptions
SpectralMM.OuterOptions
SpectralMM.MMResult
```

## Backend result types

```@docs
SpectralMM.SpectralGLM
SpectralMM.SpectralModel
SpectralMM.FittedModel
```

