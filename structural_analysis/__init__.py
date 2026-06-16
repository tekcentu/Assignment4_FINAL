"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.38.1"
__what_is_new__ = (
    "Precast: V / M x-axis pinned to member span · "
    "Switching to a shorter element no longer leaves blank space on the right · "
    "Per-stage DAF and enable/disable still in place"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
