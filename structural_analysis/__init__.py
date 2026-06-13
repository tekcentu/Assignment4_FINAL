"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.30.10"
__what_is_new__ = (
    "Coordinate numbers mirrored to top/right spines (no fit/aspect impact) · "
    "Generated-grid letter labels stick to the spine — no more drift on zoom · "
    "Optional: show grid letter next to coord on the axes, e.g. '3 (A)'"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
