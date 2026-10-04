# Julia examples

Each example constructs its own data. The documentation build executes these
examples, and the test suite executes the README's Julia fitting examples.

## Logistic regression

```@example logistic
using SpectralMM, Distributions, GLM, Random
rng = MersenneTwister(71)
X = randn(rng, 100, 3)
y = Float64.(rand(rng, 100) .< 0.5)
model = SpectralMM.glm(X, y, Bernoulli(), LogitLink())
@assert length(coef(model)) == 4
@assert length(predict(model, X)) == 100
coef(model)
```

## Sparse inputs

The response is generated from the same design, so its length always matches
the number of observations. Sparse storage is retained during fitting;
optional covariance computation uses dense coefficient matrices.

```@example sparse
using SpectralMM, Distributions, GLM, SparseArrays, Random
rng = MersenneTwister(72)
X = sprandn(rng, 200, 8, 0.3)
mu = 0.4 .+ X * fill(0.15, size(X, 2))
y = mu + randn(rng, size(X, 1))
model = SpectralMM.glm(X, y, Normal(), IdentityLink(); inference=false)
@assert length(y) == size(X, 1)
@assert length(coef(model)) == size(X, 2) + 1
@assert all(isfinite, predict(model, X))
coef(model)
```

## Robust regression

```@example robust
using SpectralMM, Random
rng = MersenneTwister(73)
X = randn(rng, 100, 3)
y = 0.5 .+ X * [0.2, -0.3, 0.1] + randn(rng, 100)
model = SpectralMM.fit(X, y, PseudoHuber(); inference=false)
@assert length(coef(model)) == 4
coef(model)
```
