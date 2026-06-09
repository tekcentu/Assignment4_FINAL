"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.30.5"
__what_is_new__ = (
    "Moment diagram drawn below member (sagging-down convention) on canvas · "
    "V and M diagrams coloured by sign (blue +, red −) with zero-crossing split · "
    "Display-only: V(x), M(x), and solver math unchanged"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
