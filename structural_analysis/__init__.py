"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.36.0"
__what_is_new__ = (
    "Precast Handling Stages V2: all three stages on one scrolling sheet · "
    "Always 2 supports; per-stage Auto-space button · "
    "Global DAF / weight / orientation drive all stages at once"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
