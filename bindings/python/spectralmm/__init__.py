"""SpectralMM statistical modeling interfaces."""
from ._api import Model
from .model import Control, Results, fit, glm
__version__ = "0.1.0"
__all__ = ["Model", "Results", "Control", "fit", "glm"]
