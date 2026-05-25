"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.8.0"
__what_is_new__ = (
    "element material override · per-element grade trials · effective material"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
