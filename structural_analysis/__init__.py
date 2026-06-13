"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.34.1"
__what_is_new__ = (
    "Polish: faint reference grid is uniform (minor ticks only, no double-draw) · "
    "Robustness: section thumbnail guards against an empty outline · "
    "Follow-ups to the section + grid review notes"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
