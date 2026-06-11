"""2D/3D Structural Analysis Package — CE 4011 Assignment 4."""

__version__ = "0.33.1"
__what_is_new__ = (
    "Fast canvas: cursor repaints via blitting (no per-move rebuild) · "
    "OpenGL 3D viewport (beta) with deps now installed by default · "
    "3D loads/offsets/Iy/roll, Tab cycles stacks, storeys"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .element3d import Element3D, FrameElement3D, TrussElement3D
from .main import run_analysis, run_from_file
