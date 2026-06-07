"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.28.0"
__what_is_new__ = (
    "Edit menu shows 'Undo <action>' / 'Redo <action>' · "
    "Tooltips and status bar describe the next undo/redo · "
    "Plain disabled labels when stacks are empty"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
