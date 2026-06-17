"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.41.0"
__what_is_new__ = (
    "Canvas member-load glyphs now follow the active view: single case "
    "shows that case only · combination shows the factored net load "
    "(e.g. 1.2D+1.6L → -20 kN/m) · labels stay in kN / kN/m / kN·m"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
