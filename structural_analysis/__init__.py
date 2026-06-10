"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.31.2"
__what_is_new__ = (
    "Rigid-offset review fixes: face-displacement deformed shapes · "
    "minimum flexible-span guard · add-command offset propagation · "
    "capped right-rigid-zone diagram load terms"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
