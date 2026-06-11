"""2D/3D Structural Analysis Package — CE 4011 Assignment 4."""

__version__ = "0.33.0"
__what_is_new__ = (
    "OpenGL 3D viewport (beta): orbit, pick, draw in space · "
    "3D load dialogs + projected arrows, Tab cycles stacked nodes · "
    "3D rigid offsets, Section Iy/roll, storeys, all-view diagrams"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .element3d import Element3D, FrameElement3D, TrussElement3D
from .main import run_analysis, run_from_file
