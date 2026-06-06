"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.26.0"
__what_is_new__ = (
    "Batch assign member loads to selected elements (Edit menu · context menu) · "
    "Batch assign nodal loads to selected nodes · "
    "Relative point-load position for batch · one undo per batch"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
