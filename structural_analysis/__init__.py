"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.40.3"
__what_is_new__ = (
    "Fix: N/V/M diagrams + station export now use case-filtered / "
    "factored member loads (matches the displayed combination) · "
    "Point-load shear jumps render as true vertical steps"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
