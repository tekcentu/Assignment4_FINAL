"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.30.1"
__what_is_new__ = (
    "Toolbar: Subplots/Customize removed, Fit button added · "
    "ESC exits pan/zoom mode first, then cancels active tool · "
    "Canvas ESC now reliable regardless of keyboard focus"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
