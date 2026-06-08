"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.30.0"
__what_is_new__ = (
    "Component-aware modal analysis: disconnected structures solved separately · "
    "Multi-component results grouped by component in modal table · "
    "Canvas skips red overlay for inactive-component elements"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
