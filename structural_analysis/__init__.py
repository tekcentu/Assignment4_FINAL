"""2D/3D Structural Analysis Package — CE 4011 Assignment 4."""

__version__ = "0.33.2"
__what_is_new__ = (
    "3D viewport renders on every GPU: light bg, compat GL profile, "
    "pyqtgraph>=0.14 · Fast canvas via blitted cursor overlay · "
    "3D loads/offsets/Iy/roll, Tab cycles stacks, storeys"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .element3d import Element3D, FrameElement3D, TrussElement3D
from .main import run_analysis, run_from_file
