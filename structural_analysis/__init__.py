"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.32.0"
__what_is_new__ = (
    "Three new combined examples in File → Open example… · "
    "Show load cases + combinations, rigid offsets, diagonal bracing and modal together · "
    "example_09 braced frame · example_10 rigid offsets · example_11 kitchen-sink"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
