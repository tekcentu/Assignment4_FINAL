"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.30.6"
__what_is_new__ = (
    "Canvas coordinate numbers stay readable at any zoom (adaptive 1-2-5 ticks) · "
    "Zoom-out no longer piles axis labels into an unreadable blur · "
    "Reference grid coarsens with the labels instead of smearing solid"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
