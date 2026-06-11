"""2D/3D Structural Analysis Package — CE 4011 Assignment 4."""

__version__ = "0.32.0"
__what_is_new__ = (
    "3D FEM: 6-DOF space frames/trusses with auto-promotion · "
    "Work planes XY/XZ/ZY + isometric, working depth, node Z · "
    "6-DOF supports, SUPPORTS3D/LOADS3D file format"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .element3d import Element3D, FrameElement3D, TrussElement3D
from .main import run_analysis, run_from_file
