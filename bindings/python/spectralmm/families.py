"""Immutable frontend specifications for non-GLM losses."""
from dataclasses import dataclass
from numbers import Real
import math


def _validate(name, value, *, probability=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0
            or (probability and value >= 1)):
        constraint = 'strictly between 0 and 1' if probability else 'positive'
        raise ValueError(f'{name} must be a finite real scalar, {constraint}')


@dataclass(frozen=True)
class PseudoHuber:
    """Pseudo-Huber loss with positive delta (default 1)."""
    delta: float = 1.0

    def __post_init__(self):
        _validate('delta', self.delta)


@dataclass(frozen=True)
class Expectile:
    """Expectile loss with 0 < q < 1."""
    q: float = 0.5

    def __post_init__(self):
        _validate('q', self.q, probability=True)


@dataclass(frozen=True)
class SmoothQuantile:
    """Smoothed quantile loss with 0 < q < 1 and positive epsilon."""
    q: float = 0.5
    epsilon: float = 0.1

    def __post_init__(self):
        _validate('q', self.q, probability=True)
        _validate('epsilon', self.epsilon)


@dataclass(frozen=True)
class StudentT:
    """Student-t residual loss with positive nu and existing unit scale."""
    nu: float = 4.0

    def __post_init__(self):
        _validate('nu', self.nu)


def _normalize_family(family, options):
    """Convert frontend objects once, preserving the legacy string path."""
    if isinstance(family, PseudoHuber):
        name, parameters = 'pseudo_huber', {'delta': family.delta}
    elif isinstance(family, Expectile):
        name, parameters = 'expectile', {'tau': family.q}
    elif isinstance(family, SmoothQuantile):
        name, parameters = 'smooth_quantile', {'tau': family.q, 'smoothing': family.epsilon}
    elif isinstance(family, StudentT):
        name, parameters = 'student_t', {'nu': family.nu}
    else:
        return family, options
    if 'family_options' in options:
        raise TypeError('do not supply family_options with a family object; use constructor parameters')
    return name, dict(options, family_options=parameters)
