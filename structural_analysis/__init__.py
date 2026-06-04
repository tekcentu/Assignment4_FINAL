"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.24.0"
__what_is_new__ = (
    "Show local axes overlay (View menu) · "
    "Renumber elements… preview + undo (Edit menu) · "
    "Merge adjacent unloaded collinear elements (node right-click)"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
