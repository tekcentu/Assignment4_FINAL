"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.11.0"
__what_is_new__ = (
    "split element on insert (grouped under member draw) · "
    "coincident-node warning"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
