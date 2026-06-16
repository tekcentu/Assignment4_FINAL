"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.40.0"
__what_is_new__ = (
    "Global Units V1 (display-only): kN/m, N/mm, kgf, tf, lbf, kip presets · "
    "View → Units… and status-bar selector · "
    "Internal solver and project files remain kN-m"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
