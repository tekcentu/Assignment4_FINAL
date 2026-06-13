"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.31.0"
__what_is_new__ = (
    "Element Details section preview now shows measured dimensions (b / h, tf / tw) · "
    "Manual sections label the √A equivalent-square side instead of an unlabelled shape · "
    "Matches the Add-Section live-preview annotations"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
