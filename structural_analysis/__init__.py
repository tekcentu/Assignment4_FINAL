"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.36.1"
__what_is_new__ = (
    "Precast sheet: mouse wheel scrolls the page instead of nudging spin "
    "boxes · Shows support spacing per stage"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
