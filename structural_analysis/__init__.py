"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.23.1"
__what_is_new__ = (
    "Load cases auto-register from assignments; unused cases labelled · "
    "Load Cases / Combinations moved to a Model menu · "
    "spreadsheet-style Ctrl+C / right-click Copy on all tables"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
