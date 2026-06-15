"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.37.0"
__what_is_new__ = (
    "Precast: elastic σ = M·y/I cracking check per stage · "
    "Tensile stress, controlling x, ratio vs. allowable, OK / CRACKING · "
    "Manual y_top / y_bottom override when section depth is missing"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
