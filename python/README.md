# SpectralMM for Python

Fit generalized linear models and robust or asymmetric regression with the shared
SpectralMM C++ core. Requires Python 3.9 or later.

From the repository root, install this package with:

```sh
python -m pip install ./python
```

Source installation automatically builds C++ and requires a C++17 compiler.

```python
import numpy as np
import spectralmm
rng = np.random.default_rng(42)
X = rng.normal(size=(100, 3))
y = rng.binomial(1, 0.5, size=100)
model = spectralmm.glm(X, y, family="binomial", link="logit", verbose=True)
print(model.summary())
print(model.predict(X[:5]))
```

Edit Python code in `spectralmm/` and run `python -m unittest discover -s python/tests`
from the repository root after installation. Shared C++ files under `cpp/` are
synchronized from the repository's root `cpp/` directory.

See the [full documentation](https://xunjian-li.github.io/SpectralMM/).
