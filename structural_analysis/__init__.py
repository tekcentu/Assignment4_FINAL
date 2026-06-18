"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.40.3"
__what_is_new__ = (
    "Active load-case member-load N/V/M reconstruction · "
    "Case/combination diagrams use effective member loads · "
    "Station export and hover share active result path"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
