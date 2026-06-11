"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.30.7"
__what_is_new__ = (
    "Canvas coordinate numbers stay readable at any zoom (adaptive 1-2-5 ticks) · "
    "Axis labels show plain metre values — no '+1e3' offset notation · "
    "Hover snap-marker updates skip the full scene rebuild (smoother on big models)"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
