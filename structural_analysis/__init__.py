"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.30.3"
__what_is_new__ = (
    "Physical Members overlay shows section-aware body around centerlines · "
    "Joint overlap shaded at shared nodes · "
    "Analysis still uses centerline elements (visual only)"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
