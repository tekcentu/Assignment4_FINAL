"""
3D element-level computations (space frame + space truss).

Class hierarchy
---------------
    Element3D  (abstract base)
    ├── FrameElement3D   — 12-DOF beam-column: axial + torsion + biaxial bending
    └── TrussElement3D   — axial-only space truss (6 translational DOFs)

Local axis convention (single source of truth: :func:`local_axes`)
------------------------------------------------------------------
``x̂`` runs from node i to node j. ``ŷ`` and ``ẑ`` complete a
right-handed triad chosen so that **a member lying in the global XY
plane reproduces the 2D element axes exactly**:

    ŷ = normalize(ẑ_global × x̂)        (reference vector = global Z)
    ẑ = x̂ × ŷ

For members parallel to global Z the reference degenerates and global
Y is used instead. An optional ``roll`` angle (rad) rotates ŷ/ẑ about
x̂ for sections that are not aligned with the default triad.

With this convention an XY-plane member has ẑ = global +Z, so its
local (N, Vy, Mz) components match the 2D (N, V, M) sign convention
bit-for-bit — the planar-equivalence regression tests rely on it.

Local DOF order
---------------
Frame:  [ux_i, uy_i, uz_i, rx_i, ry_i, rz_i,  ux_j, …, rz_j]  (12)
Truss:  [ux_i, uy_i, uz_i,  ux_j, uy_j, uz_j]                  (6)

Sign convention for the consistent load vector ``p`` matches
:mod:`structural_analysis.element`: ``F[dof] += p[dof]`` at assembly,
``q = k·d − p`` at recovery.

Gravity stays **global −Y** (the program-wide convention inherited
from the 2D solver and the GUI canvas, where Y is the vertical axis).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import (
    MemberLoad, PointLoad, UniformDistributedLoad,
    TrussTemperatureLoad, FrameTemperatureLoad,
    STANDARD_GRAVITY,
)

_PARALLEL_TOL = 1e-8


def length_and_xhat(ni, nj) -> tuple[float, np.ndarray]:
    """Element length and unit direction vector in 3D.

    Args:
        ni: Start node (needs .x, .y and optional .z).
        nj: End node.

    Returns:
        Tuple (L, x̂) — length and the 3-component unit vector i→j.

    Raises:
        ValueError: If the element has zero length.
    """
    d = np.array([
        nj.x - ni.x,
        nj.y - ni.y,
        getattr(nj, "z", 0.0) - getattr(ni, "z", 0.0),
    ], dtype=float)
    L = float(np.linalg.norm(d))
    if L < 1e-12:
        raise ValueError(
            f"Zero-length element between ({ni.x},{ni.y},"
            f"{getattr(ni, 'z', 0.0)}) and ({nj.x},{nj.y},"
            f"{getattr(nj, 'z', 0.0)})."
        )
    return L, d / L


def local_axes(ni, nj, roll: float = 0.0) -> tuple[float, np.ndarray]:
    """Build the local triad for a 3D member.

    Args:
        ni: Start node.
        nj: End node.
        roll: Rotation (rad) of ŷ/ẑ about x̂, applied after the
            default triad is built. Positive per the right-hand rule
            about x̂.

    Returns:
        Tuple (L, Λ) where Λ is the 3×3 matrix whose ROWS are
        x̂, ŷ, ẑ expressed in global coordinates (Λ maps global →
        local: ``v_local = Λ @ v_global``).
    """
    L, xhat = length_and_xhat(ni, nj)
    ref = np.array([0.0, 0.0, 1.0])
    if np.linalg.norm(np.cross(ref, xhat)) < _PARALLEL_TOL:
        # Member parallel to global Z — fall back to global Y.
        ref = np.array([0.0, 1.0, 0.0])
    yhat = np.cross(ref, xhat)
    yhat /= np.linalg.norm(yhat)
    zhat = np.cross(xhat, yhat)
    if roll != 0.0:
        cr, sr = np.cos(roll), np.sin(roll)
        yhat, zhat = cr * yhat + sr * zhat, -sr * yhat + cr * zhat
    return L, np.vstack([xhat, yhat, zhat])


def _block_rotation(lam: np.ndarray, n_blocks: int) -> np.ndarray:
    """Block-diagonal rotation: diag(Λ, Λ, …) with ``n_blocks`` blocks."""
    n = 3 * n_blocks
    R = np.zeros((n, n))
    for b in range(n_blocks):
        R[3 * b:3 * b + 3, 3 * b:3 * b + 3] = lam
    return R


@dataclass
class Element3D:
    """Abstract base class for 3D structural elements.

    Mirrors :class:`structural_analysis.element.Element2D` but for the
    6-DOF-per-node space formulation. Intentionally NOT a subclass of
    Element2D — large parts of the 2D pipeline dispatch on
    ``isinstance(elem, FrameElement2D / TrussElement2D)`` and 3D
    elements must never satisfy those checks.
    """

    id: int
    node_i: int
    node_j: int
    E: float
    A: float
    alpha: float = 0.0
    depth: float = 0.0
    rho: float = 0.0
    section_id: int | None = None
    material_id_override: int | None = None
    member_loads: list[MemberLoad] = field(default_factory=list)

    @property
    def kind(self) -> str:
        raise NotImplementedError

    @property
    def n_local_dofs(self) -> int:
        raise NotImplementedError

    def dof_keys(self) -> list[tuple[int, str]]:
        """(node_id, dof_name) per local DOF, in local DOF order."""
        raise NotImplementedError

    def length_cos_sin(self, nodes: dict) -> tuple[float, float, float]:
        """3D-aware (L, c, s) — kept for signature compatibility.

        ``c``/``s`` are the direction cosines of the member's
        projection onto the XY plane (matching the 2D meaning for
        in-plane members); they are only used for display/reporting,
        never for 3D stiffness math.
        """
        L, xhat = length_and_xhat(nodes[self.node_i], nodes[self.node_j])
        return L, float(xhat[0]), float(xhat[1])

    def local_triad(self, nodes: dict) -> tuple[float, np.ndarray]:
        return local_axes(
            nodes[self.node_i], nodes[self.node_j],
            roll=getattr(self, "roll", 0.0),
        )

    def raw_local_stiffness(self, nodes: dict) -> np.ndarray:
        raise NotImplementedError

    def transformation_matrix(self, nodes: dict) -> np.ndarray:
        """Rotation R (global → local) sized to the element's DOFs."""
        _, lam = self.local_triad(nodes)
        return _block_rotation(lam, self.n_local_dofs // 3)

    def local_consistent_load(self, nodes: dict) -> np.ndarray:
        return np.zeros(self.n_local_dofs)

    def assembly_local_indices(self) -> list[int | None]:
        return list(range(self.n_local_dofs))

    def assembled_local_stiffness_and_load(
        self, nodes: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (self.raw_local_stiffness(nodes),
                self.local_consistent_load(nodes))

    def global_stiffness_and_load(
        self, nodes: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        k_local, p_local = self.assembled_local_stiffness_and_load(nodes)
        R = self.transformation_matrix(nodes)
        return R.T @ k_local @ R, R.T @ p_local

    def local_displacement_and_end_forces(
        self,
        nodes: dict,
        u_global_elem: np.ndarray,
        p_extra_local: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``q_local = k·d − p`` recovery (same convention as 2D)."""
        R = self.transformation_matrix(nodes)
        d_local = R @ u_global_elem
        p_full = self.local_consistent_load(nodes)
        if p_extra_local is not None:
            p_full = p_full + np.asarray(p_extra_local, dtype=float)
        q_local = self.raw_local_stiffness(nodes) @ d_local - p_full
        return d_local, q_local


@dataclass
class FrameElement3D(Element3D):
    """3D frame (space beam-column) element.

    Attributes:
        Iy: Second moment of area about local y (out-of-plane bending
            for an XY-plane member), m⁴.
        Iz: Second moment about local z (the 2D ``I``), m⁴.
        J: St-Venant torsion constant, m⁴.
        G: Shear modulus (kN/m²).
        roll: Rotation of the section about the member axis (rad).
        release_i / release_j: Moment hinges about LOCAL Z at the
            respective end — the exact 3D counterpart of the 2D
            moment release (condensed out via the same Schur
            reduction). Torsion and local-y bending stay continuous.
    """

    Iy: float = 0.0
    Iz: float = 0.0
    J: float = 0.0
    G: float = 0.0
    roll: float = 0.0
    release_i: bool = False
    release_j: bool = False

    @property
    def kind(self) -> str:
        return "frame3d"

    @property
    def n_local_dofs(self) -> int:
        return 12

    def dof_keys(self) -> list[tuple[int, str]]:
        names = ("ux", "uy", "uz", "rx", "ry", "rz")
        return ([(self.node_i, n) for n in names]
                + [(self.node_j, n) for n in names])

    def raw_local_stiffness(self, nodes: dict) -> np.ndarray:
        """Textbook 12×12 space-frame stiffness (no releases)."""
        L, _ = self.local_triad(nodes)
        a = self.E * self.A / L
        t = self.G * self.J / L
        L2, L3 = L * L, L * L * L
        bz1 = 12 * self.E * self.Iz / L3
        bz2 = 6 * self.E * self.Iz / L2
        bz3 = 4 * self.E * self.Iz / L
        bz4 = 2 * self.E * self.Iz / L
        by1 = 12 * self.E * self.Iy / L3
        by2 = 6 * self.E * self.Iy / L2
        by3 = 4 * self.E * self.Iy / L
        by4 = 2 * self.E * self.Iy / L

        k = np.zeros((12, 12))
        # Axial
        k[0, 0] = k[6, 6] = a
        k[0, 6] = k[6, 0] = -a
        # Torsion
        k[3, 3] = k[9, 9] = t
        k[3, 9] = k[9, 3] = -t
        # Bending about local z (displacement in local y)
        k[1, 1] = k[7, 7] = bz1
        k[1, 7] = k[7, 1] = -bz1
        k[1, 5] = k[5, 1] = k[1, 11] = k[11, 1] = bz2
        k[5, 7] = k[7, 5] = k[7, 11] = k[11, 7] = -bz2
        k[5, 5] = k[11, 11] = bz3
        k[5, 11] = k[11, 5] = bz4
        # Bending about local y (displacement in local z). Note the
        # sign flip on the shear/rotation coupling terms relative to
        # the z-bending block: θy = −dw/dx for a right-handed triad.
        k[2, 2] = k[8, 8] = by1
        k[2, 8] = k[8, 2] = -by1
        k[2, 4] = k[4, 2] = k[2, 10] = k[10, 2] = -by2
        k[4, 8] = k[8, 4] = k[8, 10] = k[10, 8] = by2
        k[4, 4] = k[10, 10] = by3
        k[4, 10] = k[10, 4] = by4
        return k

    # ── member loads ──

    def _project_to_local(
        self, lam: np.ndarray, cx: float, cy: float, cz: float,
        coord_system: str,
    ) -> np.ndarray:
        """Project a mechanical load's components into the local triad."""
        if coord_system == "local":
            return np.array([cx, cy, cz], dtype=float)
        if coord_system == "global":
            return lam @ np.array([cx, cy, cz], dtype=float)
        if coord_system == "gravity":
            # Magnitude in cy, direction global −Y (program convention).
            return lam @ np.array([0.0, -cy, 0.0])
        raise ValueError(
            f"Unknown coord_system {coord_system!r}; "
            f"expected 'local', 'global', or 'gravity'."
        )

    @staticmethod
    def _udl_fixed_end(L: float, w: np.ndarray) -> np.ndarray:
        """Consistent nodal vector for a full-length local UDL (wx,wy,wz)."""
        wx, wy, wz = w
        p = np.zeros(12)
        p[0] += wx * L / 2
        p[6] += wx * L / 2
        p[1] += wy * L / 2
        p[7] += wy * L / 2
        p[5] += wy * L * L / 12
        p[11] += -wy * L * L / 12
        p[2] += wz * L / 2
        p[8] += wz * L / 2
        p[4] += -wz * L * L / 12
        p[10] += +wz * L * L / 12
        return p

    @staticmethod
    def _point_fixed_end(L: float, a: float, P: np.ndarray) -> np.ndarray:
        """Consistent nodal vector for a local point load (px,py,pz) at a."""
        px, py, pz = P
        xi = a / L
        n1 = 1 - 3 * xi**2 + 2 * xi**3
        n2 = L * (xi - 2 * xi**2 + xi**3)
        n3 = 3 * xi**2 - 2 * xi**3
        n4 = L * (-xi**2 + xi**3)
        p = np.zeros(12)
        p[0] += px * (1 - xi)
        p[6] += px * xi
        p[1] += py * n1
        p[5] += py * n2
        p[7] += py * n3
        p[11] += py * n4
        # θy = −dw/dx ⇒ the rotation shape functions flip sign.
        p[2] += pz * n1
        p[4] += -pz * n2
        p[8] += pz * n3
        p[10] += -pz * n4
        return p

    def local_consistent_load(self, nodes: dict) -> np.ndarray:
        """Fixed-fixed equivalent nodal loads for all member loads."""
        L, lam = self.local_triad(nodes)
        p = np.zeros(12)
        for load in self.member_loads:
            if isinstance(load, UniformDistributedLoad):
                w = self._project_to_local(
                    lam, load.wx, load.wy, getattr(load, "wz", 0.0),
                    load.coord_system,
                )
                p += self._udl_fixed_end(L, w)
            elif isinstance(load, PointLoad):
                a = float(load.a)
                if not (0 <= a <= L + 1e-10):
                    raise ValueError(
                        f"Element {self.id}: point load a={a:.3f} "
                        f"outside L={L:.3f}."
                    )
                P = self._project_to_local(
                    lam, load.px, load.py, getattr(load, "pz", 0.0),
                    load.coord_system,
                )
                p += self._point_fixed_end(L, a, P)
            elif isinstance(load, FrameTemperatureLoad):
                dT_mean = 0.5 * (load.t_top + load.t_bottom)
                if dT_mean != 0.0:
                    N_T = self.E * self.A * self.alpha * dT_mean
                    p[0] += +N_T
                    p[6] += -N_T
                dT_diff = load.t_bottom - load.t_top
                if dT_diff != 0.0:
                    if self.depth <= 0.0:
                        raise ValueError(
                            f"Frame element {self.id}: thermal gradient "
                            f"requires a positive depth (got {self.depth})."
                        )
                    # Gradient across the section depth (local y) bends
                    # about local z — same convention as the 2D element.
                    M_T = (self.E * self.Iz * self.alpha
                           * dT_diff / self.depth)
                    p[5] += -M_T
                    p[11] += +M_T
            elif isinstance(load, TrussTemperatureLoad):
                raise TypeError(
                    f"Frame element {self.id} cannot carry a "
                    f"TrussTemperatureLoad. Use FrameTemperatureLoad"
                    f"(t_top=ΔT, t_bottom=ΔT) for uniform heating."
                )
            else:
                raise TypeError(
                    f"Unsupported load on element {self.id}: {type(load)}"
                )
        return p

    def self_weight_fixed_end_local(self, nodes: dict) -> np.ndarray:
        """RAW fixed-end vector for self-weight (global −Y gravity)."""
        rho = float(self.rho)
        if rho == 0.0 or self.A == 0.0:
            return np.zeros(12)
        L, lam = self.local_triad(nodes)
        w_mag = rho * self.A * STANDARD_GRAVITY / 1000.0  # kN/m
        w_local = lam @ np.array([0.0, -w_mag, 0.0])
        return self._udl_fixed_end(L, w_local)

    # ── moment releases (local-z hinges, mirroring 2D semantics) ──

    def _released_dofs(self) -> list[int]:
        r: list[int] = []
        if self.release_i:
            r.append(5)
        if self.release_j:
            r.append(11)
        return r

    def assembly_local_indices(self) -> list[int | None]:
        m: list[int | None] = list(range(12))
        if self.release_i:
            m[5] = None
        if self.release_j:
            m[11] = None
        return m

    def condense_local_load_for_releases(
        self, p_local: np.ndarray, nodes: dict,
    ) -> np.ndarray:
        """Schur-reduce a fixed-fixed local vector for the hinged DOFs."""
        released = self._released_dofs()
        p_arr = np.asarray(p_local, dtype=float)
        if not released:
            return p_arr.copy()
        retained = [i for i in range(12) if i not in released]
        k = self.raw_local_stiffness(nodes)
        kab = k[np.ix_(retained, released)]
        kbb = k[np.ix_(released, released)]
        p_out = np.zeros(12, dtype=float)
        p_out[retained] = (
            p_arr[retained]
            - kab @ np.linalg.solve(kbb, p_arr[released])
        )
        return p_out

    def assembled_local_stiffness_and_load(
        self, nodes: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        k = self.raw_local_stiffness(nodes)
        p = self.local_consistent_load(nodes)
        released = self._released_dofs()
        if not released:
            return k, p
        retained = [i for i in range(12) if i not in released]
        kaa = k[np.ix_(retained, retained)]
        kab = k[np.ix_(retained, released)]
        kba = k[np.ix_(released, retained)]
        kbb = k[np.ix_(released, released)]
        kbb_inv = np.linalg.inv(kbb)
        k_out = np.zeros_like(k)
        k_out[np.ix_(retained, retained)] = kaa - kab @ kbb_inv @ kba
        return k_out, self.condense_local_load_for_releases(p, nodes)

    def local_displacement_and_end_forces(
        self,
        nodes: dict,
        u_global_elem: np.ndarray,
        p_extra_local: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recovery with back-substitution of released rotations."""
        R = self.transformation_matrix(nodes)
        d_from_global = R @ u_global_elem
        k_full = self.raw_local_stiffness(nodes)
        p_full = self.local_consistent_load(nodes)
        if p_extra_local is not None:
            p_full = p_full + np.asarray(p_extra_local, dtype=float)
        released = self._released_dofs()
        if not released:
            return d_from_global, k_full @ d_from_global - p_full
        retained = [i for i in range(12) if i not in released]
        kba = k_full[np.ix_(released, retained)]
        kbb = k_full[np.ix_(released, released)]
        db = np.linalg.solve(
            kbb, p_full[released] - kba @ d_from_global[retained],
        )
        d_local = np.array(d_from_global, copy=True)
        d_local[released] = db
        return d_local, k_full @ d_local - p_full


@dataclass
class TrussElement3D(Element3D):
    """3D truss element — axial force only, 6 translational DOFs."""

    @property
    def kind(self) -> str:
        return "truss3d"

    @property
    def n_local_dofs(self) -> int:
        return 6

    def dof_keys(self) -> list[tuple[int, str]]:
        names = ("ux", "uy", "uz")
        return ([(self.node_i, n) for n in names]
                + [(self.node_j, n) for n in names])

    def raw_local_stiffness(self, nodes: dict) -> np.ndarray:
        L, _ = self.local_triad(nodes)
        EA_L = self.E * self.A / L
        k = np.zeros((6, 6))
        k[0, 0] = k[3, 3] = EA_L
        k[0, 3] = k[3, 0] = -EA_L
        return k

    def local_consistent_load(self, nodes: dict) -> np.ndarray:
        p = np.zeros(6)
        for load in self.member_loads:
            if isinstance(load, TrussTemperatureLoad):
                N_T = self.E * self.A * self.alpha * load.delta_T
                p[0] += +N_T
                p[3] += -N_T
            elif isinstance(load, FrameTemperatureLoad):
                raise TypeError(
                    f"Truss element {self.id} cannot carry a "
                    f"FrameTemperatureLoad. Use TrussTemperatureLoad"
                    f"(delta_T=ΔT) for uniform heating."
                )
            elif isinstance(load, (UniformDistributedLoad, PointLoad)):
                raise TypeError(
                    f"TrussElement {self.id} cannot carry transverse "
                    f"{type(load).__name__} — use FrameElement3D instead."
                )
        return p


# ═══════════════════════════════════════════════════════════════
#  2D → 3D promotion
# ═══════════════════════════════════════════════════════════════


def promote_element_to_3d(elem, model) -> Element3D:
    """Build the 3D solve-time equivalent of a 2D model element.

    Used by the assembler when :func:`structural_analysis.assembler.
    model_is_3d` decides the model needs the 6-DOF-per-node pipeline:
    the user keeps drawing plain frame/truss members in the GUI, and
    the solve silently lifts them into space elements.

    Property mapping for frames (documented V1 defaults):
        Iz = elem.I              (the 2D bending inertia)
        Iy = section.Iy if set, else elem.I
        J  = section.J if > 0, else Iy + Iz (polar approximation)
        G  = effective material G (E/2 when ν or the material is
             unavailable — the ν = 0 isotropic identity)

    Raises:
        ValueError: For 2D features without a 3D counterpart yet
            (rigid end offsets).
    """
    from .element import FrameElement2D, TrussElement2D
    from .model import effective_material

    if isinstance(elem, Element3D):
        return elem
    if isinstance(elem, TrussElement2D):
        return TrussElement3D(
            id=elem.id, node_i=elem.node_i, node_j=elem.node_j,
            E=elem.E, A=elem.A, alpha=elem.alpha, depth=elem.depth,
            rho=elem.rho, section_id=elem.section_id,
            material_id_override=elem.material_id_override,
            member_loads=elem.member_loads,
        )
    if isinstance(elem, FrameElement2D):
        if elem.has_offsets:
            raise ValueError(
                f"Element {elem.id}: rigid end offsets are not yet "
                "supported in 3D analysis. Remove the offsets or keep "
                "the model planar (all z = 0)."
            )
        G = elem.E / 2.0
        try:
            G = effective_material(model, elem).G
        except (KeyError, TypeError, ValueError):
            pass
        sec = model.sections.get(elem.section_id) if model else None
        Iy = elem.I
        J = sec.J if (sec is not None and sec.J > 0.0) else Iy + elem.I
        return FrameElement3D(
            id=elem.id, node_i=elem.node_i, node_j=elem.node_j,
            E=elem.E, A=elem.A, alpha=elem.alpha, depth=elem.depth,
            rho=elem.rho, section_id=elem.section_id,
            material_id_override=elem.material_id_override,
            member_loads=elem.member_loads,
            Iy=Iy, Iz=elem.I, J=J, G=G,
            release_i=elem.release_i, release_j=elem.release_j,
        )
    raise TypeError(
        f"Cannot promote element {elem!r} to 3D — unknown element type."
    )
