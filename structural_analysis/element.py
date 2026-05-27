"""
Element-level computations using OOP inheritance.

Class hierarchy
---------------
    Element2D  (abstract base)
    ├── FrameElement2D   — 6-DOF beam-column with axial + flexural stiffness
    └── TrussElement2D   — axial-only (zero flexural stiffness)

Sign convention for consistent load vector *p*
-----------------------------------------------
``p`` represents the **equivalent nodal forces that do the same work as
the distributed/point member load** (energy-consistent convention).
During assembly: ``F[dof] += p[dof]``.
During force recovery: ``q = k · d − p``.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .model import (
    MemberLoad, PointLoad, UniformDistributedLoad,
    TrussTemperatureLoad, FrameTemperatureLoad,
)


def _length_cos_sin(ni, nj) -> tuple[float, float, float]:
    """Compute element length and direction cosines.

    Args:
        ni: Start node (must have .x, .y attributes).
        nj: End node (must have .x, .y attributes).

    Returns:
        Tuple (L, c, s) where L is length, c = cos θ, s = sin θ.

    Raises:
        ValueError: If the element has zero length.
    """
    dx = nj.x - ni.x
    dy = nj.y - ni.y
    L = float(np.hypot(dx, dy))
    if L < 1e-12:
        raise ValueError(f"Zero-length element between ({ni.x},{ni.y}) and ({nj.x},{nj.y}).")
    return L, dx / L, dy / L


def _rotation_matrix_6x6(c: float, s: float) -> np.ndarray:
    """Build the 6×6 block-diagonal rotation matrix R = diag(T, T).

    Args:
        c: Cosine of the element inclination angle (dx/L).
        s: Sine of the element inclination angle (dy/L).

    Returns:
        6×6 numpy array — the rotation matrix R.
    """
    R = np.zeros((6, 6))
    T = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    R[0:3, 0:3] = T
    R[3:6, 3:6] = T
    return R


@dataclass
class Element2D:
    """Abstract base class for 2-D structural elements.

    All element types share the same 6-DOF local convention:
    [u_i, v_i, θ_i, u_j, v_j, θ_j].

    Attributes:
        id: Unique element identifier.
        node_i: Start node ID.
        node_j: End node ID.
        E: Modulus of elasticity (kN/m²).
        A: Cross-sectional area (m²).
        alpha: Coefficient of thermal expansion (1/°C). Default 0 (inert).
        depth: Section depth (m), used for frame thermal gradient. Default 0.
        section_id: id of the :class:`Section` this element was assigned to
            (None for elements built outside the model layer, e.g. raw unit
            tests). Stiffness math ignores this — it's a back-reference used
            by the writer and the GUI command propagation logic to find
            elements that belong to a given Section/Material.
        material_id_override: optional id of a :class:`Material` that takes
            precedence over the section's default material for this element.
            ``None`` means "use the section default". E / α / ρ on this
            element are always populated from the *effective* material
            (override if set, otherwise section default) — see
            :func:`structural_analysis.model.effective_material`. Stiffness
            math doesn't read this attribute directly; it reads ``self.E``
            etc., which are written at construction / propagation time.
        member_loads: List of MemberLoad objects (UDL, PointLoad, thermal).
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
        """Return the element type as a string.

        Returns:
            Element type identifier ("frame" or "truss").
        """
        raise NotImplementedError

    def raw_local_stiffness(self, nodes: dict) -> np.ndarray:
        """Compute the 6×6 local stiffness matrix (no releases).

        Args:
            nodes: Dict mapping node IDs to Node objects.

        Returns:
            6×6 numpy array — unreleased local stiffness k'.
        """
        raise NotImplementedError

    def length_cos_sin(self, nodes: dict) -> tuple[float, float, float]:
        """Compute element length and direction cosines.

        Args:
            nodes: Dict mapping node IDs to Node objects.

        Returns:
            Tuple (L, c, s) — length, cos θ, sin θ.
        """
        return _length_cos_sin(nodes[self.node_i], nodes[self.node_j])

    def transformation_matrix(self, nodes: dict) -> np.ndarray:
        """Build the 6×6 rotation matrix R (local → global).

        Args:
            nodes: Dict mapping node IDs to Node objects.

        Returns:
            6×6 numpy array — the rotation matrix R.
        """
        _, c, s = self.length_cos_sin(nodes)
        return _rotation_matrix_6x6(c, s)

    def local_consistent_load(self, nodes: dict) -> np.ndarray:
        """Compute the consistent load vector p in local coordinates.

        Default: zero vector. Overridden by FrameElement2D.

        Args:
            nodes: Dict mapping node IDs to Node objects.

        Returns:
            6-element numpy array — consistent load vector p.
        """
        return np.zeros(6)

    def assembly_local_indices(self) -> list[int | None]:
        """Map from element local DOFs to active DOF slots.

        None means the DOF is not assembled (e.g. truss θ).

        Returns:
            List of 6 entries: int index or None per local DOF.
        """
        return [0, 1, 2, 3, 4, 5]

    def consistent_mass_local(self, nodes: dict) -> np.ndarray:
        """Return the 6×6 consistent mass matrix in the local frame.

        Implementations live on the concrete element subclasses.
        Mass entries are emitted in the kN-m-s consistent system
        (mass in Mg = kN·s²/m); the unit conversion from the
        user-facing kg/m³ density is done here.

        Args:
            nodes: Dict mapping node IDs to Node objects.

        Returns:
            6×6 numpy array — element consistent mass matrix.
        """
        raise NotImplementedError

    def assembled_local_stiffness_and_load(self, nodes: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return stiffness and load after any release condensation.

        Args:
            nodes: Dict mapping node IDs to Node objects.

        Returns:
            Tuple (k_condensed, p_condensed) — 6×6 and 6×1 arrays.
        """
        return self.raw_local_stiffness(nodes), self.local_consistent_load(nodes)

    def global_stiffness_and_load(self, nodes: dict) -> tuple[np.ndarray, np.ndarray]:
        """Transform condensed k and p to global coordinates.

        Args:
            nodes: Dict mapping node IDs to Node objects.

        Returns:
            Tuple (k_global, p_global) — 6×6 and 6×1 arrays in global coords.
        """
        k_local, p_local = self.assembled_local_stiffness_and_load(nodes)
        R = self.transformation_matrix(nodes)
        return R.T @ k_local @ R, R.T @ p_local

    def local_displacement_and_end_forces(
        self,
        nodes: dict,
        u_global_elem: np.ndarray,
        p_extra_local: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute local displacements and member end forces.

        Args:
            nodes: Dict mapping node IDs to Node objects.
            u_global_elem: 6-element array of element global displacements.
            p_extra_local: Optional extra fixed-end vector to include in
                the recovery alongside ``local_consistent_load``. Used to
                feed back transient loads (e.g. self-weight) that were
                added to global F but are not stored on the element.

        Returns:
            Tuple (d_local, q_local) where d_local is the 6-element local
            displacement vector and q_local is [N_i, V_i, M_i, N_j, V_j, M_j].

        Sign convention: ``q_local = K·d − p_local`` is the action of the
        nodes on the element in the element's local frame. ``+N_i`` is
        tension at the i-end; ``+V_i`` is in the +y_local direction at the
        i-end; ``+M_i`` is in the +z_local (out-of-plane CCW) direction at
        the i-end. ``q_global = Rᵀ·q_local`` satisfies ``Σ q_global = applied``
        at every free node (see :func:`postprocessor.equilibrium_check`),
        i.e. q is the force the node applies to the element.
        """
        R = self.transformation_matrix(nodes)
        d_local = R @ u_global_elem
        p_full = self.local_consistent_load(nodes)
        if p_extra_local is not None:
            p_full = p_full + np.asarray(p_extra_local, dtype=float)
        q_local = self.raw_local_stiffness(nodes) @ d_local - p_full
        return d_local, q_local


@dataclass
class FrameElement2D(Element2D):
    """2-D frame (beam-column) element with optional moment releases.

    Uses Schur-complement static condensation for releases, applied
    to both k and p simultaneously (exact for member loads + releases).

    Attributes:
        I: Moment of inertia (m⁴).
        release_i: If True, hinge at node i (local DOF 2).
        release_j: If True, hinge at node j (local DOF 5).
    """

    I: float = 0.0
    release_i: bool = False
    release_j: bool = False

    @property
    def kind(self) -> str:
        """Return the element type identifier.

        Returns:
            The string "frame".
        """
        return "frame"

    def raw_local_stiffness(self, nodes: dict) -> np.ndarray:
        """Compute the 6×6 local stiffness for an unreleased frame element.

        Args:
            nodes: Dict mapping node IDs to Node objects.

        Returns:
            6×6 numpy array — unreleased local stiffness using EA/L
            and 12EI/L³, 6EI/L², 4EI/L, 2EI/L terms.
        """
        L, _, _ = self.length_cos_sin(nodes)
        EA_L = self.E * self.A / L
        EI = self.E * self.I
        L2, L3 = L * L, L * L * L
        return np.array([
            [ EA_L,       0,         0,    -EA_L,        0,         0],
            [    0, 12*EI/L3,  6*EI/L2,        0, -12*EI/L3,  6*EI/L2],
            [    0,  6*EI/L2,  4*EI/L,         0,  -6*EI/L2,  2*EI/L ],
            [-EA_L,       0,         0,     EA_L,        0,         0],
            [    0,-12*EI/L3, -6*EI/L2,        0,  12*EI/L3, -6*EI/L2],
            [    0,  6*EI/L2,  2*EI/L,         0,  -6*EI/L2,  4*EI/L ],
        ])

    def consistent_mass_local(self, nodes: dict) -> np.ndarray:
        """Hermitian (energy-consistent) mass matrix for a 2D beam-column.

        Local DOFs: [u_i, v_i, θ_i, u_j, v_j, θ_j].

        Args:
            nodes: Dict mapping node IDs to Node objects.

        Returns:
            6×6 numpy array — element consistent mass matrix in
            kN·s²/m (i.e. Mg) units.

        Notes:
            Density on the element is stored as ``self.rho`` in kg/m³;
            we convert to Mg/m³ (divide by 1000) so the resulting mass
            entries are consistent with the kN-m static stiffness.
            The full Hermitian form is used on both translational and
            rotational DOFs — moment-release condensation is not applied
            to mass (release DOFs are then simply unassembled by
            :meth:`assembly_local_indices`).
        """
        L, _, _ = self.length_cos_sin(nodes)
        rho_consistent = self.rho / 1000.0  # kg/m³ → Mg/m³
        m_bar = rho_consistent * self.A      # Mg/m
        if m_bar <= 0.0:
            return np.zeros((6, 6))
        coef = m_bar * L / 420.0
        L2 = L * L
        # Axial (Hermitian linear, 1/6·[2,1;1,2]·m̄L → 1/420·[140,70;70,140]·m̄L)
        # Bending block uses Hermite cubic shape functions.
        return coef * np.array([
            [140.0,    0.0,     0.0,    70.0,    0.0,     0.0],
            [  0.0,  156.0,   22.0*L,   0.0,    54.0,  -13.0*L],
            [  0.0,  22.0*L,   4.0*L2,  0.0,    13.0*L, -3.0*L2],
            [ 70.0,    0.0,     0.0,   140.0,    0.0,     0.0],
            [  0.0,   54.0,    13.0*L,  0.0,   156.0,  -22.0*L],
            [  0.0, -13.0*L,  -3.0*L2,  0.0,  -22.0*L,   4.0*L2],
        ])

    def lumped_mass_local(self, nodes: dict) -> np.ndarray:
        """Translational-only lumped mass matrix for a 2D beam-column.

        Half the total bar mass m = ρ·A·L is placed at each end node
        on the ux and uy DOFs; rotational θ DOFs receive zero mass.
        The translational block is isotropic (m/2 · I₂), so the global
        matrix is rotation-invariant — R.T M_loc R returns the same
        matrix. The modal solver detects the zero rz diagonals and
        condenses them out (Guyan reduction), see solve_modal.
        """
        L, _, _ = self.length_cos_sin(nodes)
        rho_consistent = self.rho / 1000.0
        m_bar = rho_consistent * self.A
        if m_bar <= 0.0:
            return np.zeros((6, 6))
        half = 0.5 * m_bar * L
        M = np.zeros((6, 6))
        # ux_i, uy_i at node i ; ux_j, uy_j at node j ; θ slots stay 0.
        M[0, 0] = half
        M[1, 1] = half
        M[3, 3] = half
        M[4, 4] = half
        return M

    def local_consistent_load(self, nodes: dict) -> np.ndarray:
        """Compute equivalent nodal loads from member loads (fixed-fixed).

        Supports four load types:
          - UniformDistributedLoad: wL/2 and wL²/12 terms
          - PointLoad: Hermitian shape-function equivalents
          - FrameTemperatureLoad: uses t_top and t_bottom
              * Axial:    N_T = E·A·α·ΔT_mean  where ΔT_mean = (t_top+t_bottom)/2
              * Bending:  M_T = E·I·α·(t_bottom − t_top)/depth
          - TrussTemperatureLoad: REJECTED on frame elements with TypeError
              (use FrameTemperatureLoad(t_top=X, t_bottom=X) for uniform
               heating of a frame element).

        Sign convention: p is ADDED to F during assembly.
        For a positive mean-temperature rise (heating):
            p_axial = [+N_T, 0, 0, −N_T, 0, 0]
        For a gradient with bottom warmer than top:
            p_bending = [0, 0, −M_T, 0, 0, +M_T]

        Args:
            nodes: Dict mapping node IDs to Node objects.

        Returns:
            6-element numpy array — consistent load vector p.

        Raises:
            ValueError: If a point load a is outside [0, L], or if a thermal
                gradient is applied but the element's depth is not positive.
            TypeError: If a TrussTemperatureLoad is applied to a frame element
                (use FrameTemperatureLoad with equal t_top/t_bottom instead).
        """
        L, _, _ = self.length_cos_sin(nodes)
        p = np.zeros(6)
        for load in self.member_loads:
            if isinstance(load, UniformDistributedLoad):
                w = load.wy
                p += np.array([0, w*L/2, w*L**2/12, 0, w*L/2, -w*L**2/12])
            elif isinstance(load, PointLoad):
                a = float(load.a)
                if not (0 <= a <= L + 1e-10):
                    raise ValueError(f"Element {self.id}: point load a={a:.3f} outside L={L:.3f}.")
                xi = a / L if L > 0 else 0
                n1 = 1 - 3*xi**2 + 2*xi**3
                n2 = L*(xi - 2*xi**2 + xi**3)
                n3 = 3*xi**2 - 2*xi**3
                n4 = L*(-xi**2 + xi**3)
                p += np.array([0, load.py*n1, load.py*n2, 0, load.py*n3, load.py*n4])
            elif isinstance(load, FrameTemperatureLoad):
                # Mean temperature → axial effect
                dT_mean = 0.5 * (load.t_top + load.t_bottom)
                if dT_mean != 0.0:
                    N_T = self.E * self.A * self.alpha * dT_mean
                    p += np.array([+N_T, 0, 0, -N_T, 0, 0])
                # Top/bottom difference → bending effect
                dT_diff = load.t_bottom - load.t_top
                if dT_diff != 0.0:
                    if self.depth <= 0.0:
                        raise ValueError(
                            f"Frame element {self.id}: thermal gradient "
                            f"requires a positive depth (got {self.depth})."
                        )
                    # I is available on FrameElement2D only
                    M_T = self.E * self.I * self.alpha * dT_diff / self.depth
                    p += np.array([0, 0, -M_T, 0, 0, +M_T])
            elif isinstance(load, TrussTemperatureLoad):
                raise TypeError(
                    f"Frame element {self.id} cannot carry a TrussTemperatureLoad. "
                    f"Use FrameTemperatureLoad(t_top=ΔT, t_bottom=ΔT) for uniform heating."
                )
            else:
                raise TypeError(f"Unsupported load on element {self.id}: {type(load)}")
        return p

    def _released_dofs(self) -> list[int]:
        """Return indices of released (condensed-out) local DOFs.

        Returns:
            List of int — indices 2 and/or 5, or empty if no releases.
        """
        r: list[int] = []
        if self.release_i: r.append(2)
        if self.release_j: r.append(5)
        return r

    def assembly_local_indices(self) -> list[int | None]:
        """Mark released rotational DOFs as None (not assembled).

        Returns:
            List of 6 entries where released DOFs are None.
        """
        m: list[int | None] = [0, 1, 2, 3, 4, 5]
        if self.release_i: m[2] = None
        if self.release_j: m[5] = None
        return m

    def condense_local_load_for_releases(
        self, p_local: np.ndarray, nodes: dict,
    ) -> np.ndarray:
        """Apply moment-release condensation to an arbitrary local load.

        Same Schur-complement reduction as
        ``assembled_local_stiffness_and_load``, but acting on any
        fixed-fixed local 6-vector (member loads, self-weight, …):
            p_c = p_a  − k_ab · k_bb⁻¹ · p_b   (released entries → 0)

        Callers that build a self-weight or other equivalent nodal-load
        vector outside ``local_consistent_load`` MUST route it through
        this helper before transforming to global, otherwise released
        rotational terms get silently dropped at assembly time.
        """
        released = self._released_dofs()
        p_arr = np.asarray(p_local, dtype=float)
        if not released:
            return p_arr.copy()
        retained = [i for i in range(6) if i not in released]
        k = self.raw_local_stiffness(nodes)
        kab = k[np.ix_(retained, released)]
        kbb = k[np.ix_(released, released)]
        p_out = np.zeros(6, dtype=float)
        p_out[retained] = p_arr[retained] - kab @ np.linalg.solve(kbb, p_arr[released])
        return p_out

    def assembled_local_stiffness_and_load(self, nodes: dict) -> tuple[np.ndarray, np.ndarray]:
        """Apply Schur-complement static condensation for moment releases.

        Condenses both stiffness and load simultaneously:
            k_c = k_aa − k_ab · k_bb⁻¹ · k_ba
            p_c = p_a  − k_ab · k_bb⁻¹ · p_b

        Args:
            nodes: Dict mapping node IDs to Node objects.

        Returns:
            Tuple (k_condensed, p_condensed) — 6×6 and 6-element arrays
            with zeros at released DOF positions.
        """
        k = self.raw_local_stiffness(nodes)
        p = self.local_consistent_load(nodes)
        released = self._released_dofs()
        if not released:
            return k, p
        retained = [i for i in range(6) if i not in released]
        kaa = k[np.ix_(retained, retained)]
        kab = k[np.ix_(retained, released)]
        kba = k[np.ix_(released, retained)]
        kbb = k[np.ix_(released, released)]
        kbb_inv = np.linalg.inv(kbb)
        k_out = np.zeros_like(k)
        k_out[np.ix_(retained, retained)] = kaa - kab @ kbb_inv @ kba
        p_out = self.condense_local_load_for_releases(p, nodes)
        return k_out, p_out

    def local_displacement_and_end_forces(
        self,
        nodes: dict,
        u_global_elem: np.ndarray,
        p_extra_local: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recover local displacements and end forces (with back-substitution).

        For released elements, condensed-out rotational DOFs are recovered:
            d_b = k_bb⁻¹ · (p_b − k_ba · d_a)
        Then: q = k_full · d_full − p_full.

        Args:
            nodes: Dict mapping node IDs to Node objects.
            u_global_elem: 6-element array of element global displacements.
            p_extra_local: Optional extra raw (unreleased) fixed-end
                vector folded into ``p_full``. Used to include self-weight
                (or any other non-persisted member load) in both the
                released-DOF back-substitution and the final ``q = K·d − p``.

        Returns:
            Tuple (d_local, q_local) where d_local includes recovered
            released DOFs and q_local = [N_i, V_i, M_i, N_j, V_j, M_j].
        """
        R = self.transformation_matrix(nodes)
        d_from_global = R @ u_global_elem
        k_full = self.raw_local_stiffness(nodes)
        p_full = self.local_consistent_load(nodes)
        if p_extra_local is not None:
            p_full = p_full + np.asarray(p_extra_local, dtype=float)
        released = self._released_dofs()
        if not released:
            return d_from_global, k_full @ d_from_global - p_full
        retained = [i for i in range(6) if i not in released]
        kba = k_full[np.ix_(released, retained)]
        kbb = k_full[np.ix_(released, released)]
        db = np.linalg.solve(kbb, p_full[released] - kba @ d_from_global[retained])
        d_local = np.array(d_from_global, copy=True)
        d_local[released] = db
        return d_local, k_full @ d_local - p_full


@dataclass
class TrussElement2D(Element2D):
    """2-D truss element — axial force only.

    Rotational DOFs are marked None in assembly_local_indices()
    so the DofManager auto-suppresses Rz at pure-truss nodes.
    """

    @property
    def kind(self) -> str:
        """Return the element type identifier.

        Returns:
            The string "truss".
        """
        return "truss"

    def raw_local_stiffness(self, nodes: dict) -> np.ndarray:
        """Compute 6×6 local stiffness with only axial terms.

        Args:
            nodes: Dict mapping node IDs to Node objects.

        Returns:
            6×6 numpy array — axial-only stiffness (DOFs 0, 3 non-zero).
        """
        L, _, _ = self.length_cos_sin(nodes)
        EA_L = self.E * self.A / L
        k = np.zeros((6, 6))
        k[0,0] = EA_L; k[0,3] = -EA_L
        k[3,0] = -EA_L; k[3,3] = EA_L
        return k

    def assembly_local_indices(self) -> list[int | None]:
        """Mark rotational DOFs as inactive for truss elements.

        Returns:
            [0, 1, None, 3, 4, None] — DOFs 2 and 5 suppressed.
        """
        return [0, 1, None, 3, 4, None]

    def consistent_mass_local(self, nodes: dict) -> np.ndarray:
        """Consistent translational mass for a 2D truss bar.

        Local DOFs: [u_i, v_i, _, u_j, v_j, _] (the rotational slots are
        kept for shape compatibility; their mass entries are zero and
        :meth:`assembly_local_indices` already skips them at assembly).

        Args:
            nodes: Dict mapping node IDs to Node objects.

        Returns:
            6×6 numpy array — translational consistent mass matrix
            (m̄·L/6 · diag([2,2,*,1,1,*]) block form), in Mg units.
        """
        L, _, _ = self.length_cos_sin(nodes)
        rho_consistent = self.rho / 1000.0
        m_bar = rho_consistent * self.A
        if m_bar <= 0.0:
            return np.zeros((6, 6))
        c = m_bar * L / 6.0
        return c * np.array([
            [2.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])

    def lumped_mass_local(self, nodes: dict) -> np.ndarray:
        """Translational-only lumped mass matrix for a 2D truss bar.

        Same diagonal pattern as the frame case: half the total bar
        mass at each end on ux and uy. The rotational slots remain
        zero (truss already has no rotational DOF at assembly time).
        """
        L, _, _ = self.length_cos_sin(nodes)
        rho_consistent = self.rho / 1000.0
        m_bar = rho_consistent * self.A
        if m_bar <= 0.0:
            return np.zeros((6, 6))
        half = 0.5 * m_bar * L
        M = np.zeros((6, 6))
        M[0, 0] = half
        M[1, 1] = half
        M[3, 3] = half
        M[4, 4] = half
        return M

    def local_consistent_load(self, nodes: dict) -> np.ndarray:
        """Compute thermal fixed-end forces for truss elements.

        Truss elements carry axial force only. The only accepted thermal
        load type is TrussTemperatureLoad (uniform ΔT). The α value is
        read from self.alpha (populated from the element's Material).

        For uniform ΔT: N_T = E · A · α · ΔT
            p = [+N_T, 0, 0, −N_T, 0, 0]

        FrameTemperatureLoad is explicitly rejected with TypeError — a
        through-depth gradient is physically meaningless on a truss.
        UDL and PointLoad (transverse loads) are also rejected.

        Args:
            nodes: Dict mapping node IDs to Node objects.

        Returns:
            6-element numpy array — thermal consistent load vector.

        Raises:
            TypeError: If a FrameTemperatureLoad, UDL, or PointLoad is
                applied to a truss element.
        """
        p = np.zeros(6)
        for load in self.member_loads:
            if isinstance(load, TrussTemperatureLoad):
                N_T = self.E * self.A * self.alpha * load.delta_T
                p += np.array([+N_T, 0.0, 0.0, -N_T, 0.0, 0.0])
            elif isinstance(load, FrameTemperatureLoad):
                raise TypeError(
                    f"Truss element {self.id} cannot carry a FrameTemperatureLoad. "
                    f"Use TrussTemperatureLoad(delta_T=ΔT) for uniform heating."
                )
            elif isinstance(load, (UniformDistributedLoad, PointLoad)):
                raise TypeError(
                    f"TrussElement {self.id} cannot carry transverse "
                    f"{type(load).__name__} — use FrameElement2D instead."
                )
        return p
