"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.34.2"
__what_is_new__ = (
    "Fix: one-axis generated grids keep the reference grid on the other axis · "
    "Reference grid is per-axis (minor where lines exist, major otherwise) · "
    "Section thumbnail guards against an empty outline"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
