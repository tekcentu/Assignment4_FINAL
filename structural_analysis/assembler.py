"""
Assembler: DOF management, connectivity validation, global K and F.

Design decisions
----------------
- DofManager dynamically decides which nodes get a rotational DOF (Rz).
  A node only receives Rz if at least one FrameElement2D with an
  unreleased end connects to it, or if the support/load explicitly
  involves Rz.  This prevents singular K for pure-truss models without
  requiring the user to manually restrain rotations.

- Graph-based DFS detects disconnected components *before* assembly.
  Each component is checked for at least one support — floating
  components are reported with the offending node set.

- The E matrix uses the course notation:
    E_map[node_id] = {"ux": eq_number_or_None, "uy": ..., "rz": ...}
  where ``None`` means the DOF does not exist (truss node with no
  rotation) and ``0`` means restrained.  Positive integers are
  free equation numbers.

  For backward-compatibility the printed E matrix shows 0 for both
  restrained and inactive DOFs (same visual as Assignment 2).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .element import Element2D, FrameElement2D, TrussElement2D
from .element3d import (
    Element3D, FrameElement3D, TrussElement3D, promote_element_to_3d,
)
from .model import NODE_COINCIDENCE_TOL, STANDARD_GRAVITY, StructuralModel


# Cap the coincident-pair list shown in the warning to keep the
# results panel readable on pathological imports. The full set is
# always available by re-running the check.
_MAX_COINCIDENT_PAIRS_IN_WARNING: int = 10


# Per-node DOF name tuples for the two pipelines. Order matters: it is
# the column order of the displayed E matrix and the per-node DOF
# numbering order.
DOF_NAMES_2D: tuple[str, ...] = ("ux", "uy", "rz")
DOF_NAMES_3D: tuple[str, ...] = ("ux", "uy", "uz", "rx", "ry", "rz")


def model_is_3d(model: StructuralModel) -> bool:
    """Decide whether ``model`` needs the 6-DOF-per-node 3D pipeline.

    True when the model carries any out-of-plane content:
    a node with z ≠ 0, a native :class:`Element3D`, a support using a
    3D-only DOF/settlement, a nodal load with fz/mx/my, a member load
    with a z component — or when ``model.force_3d`` is set.

    A model where all of those are absent solves through the legacy 2D
    pipeline, bit-identical to pre-3D versions.
    """
    if getattr(model, "force_3d", False):
        return True
    if any(getattr(n, "z", 0.0) != 0.0 for n in model.nodes.values()):
        return True
    if any(isinstance(e, Element3D) for e in model.elements):
        return True
    if any(getattr(s, "has_3d_content", False)
           for s in model.supports.values()):
        return True
    if any(getattr(ld, "has_3d_content", False)
           for ld in model.nodal_loads):
        return True
    for elem in model.elements:
        for ld in getattr(elem, "member_loads", []) or []:
            if getattr(ld, "wz", 0.0) or getattr(ld, "pz", 0.0):
                return True
    return False


def prepare_solve_elements(model: StructuralModel) -> tuple[bool, list]:
    """Return ``(is_3d, solve_elements)`` for the assembly pass.

    In 2D mode the model's own element list is returned unchanged. In
    3D mode every 2D element is promoted to its space equivalent (see
    :func:`structural_analysis.element3d.promote_element_to_3d`);
    native 3D elements pass through. The model itself is never
    mutated — promotion is a per-solve view.
    """
    is_3d = model_is_3d(model)
    if not is_3d:
        return False, list(model.elements)
    return True, [promote_element_to_3d(e, model) for e in model.elements]


# ═══════════════════════════════════════════════════════════════
#  DOF Manager
# ═══════════════════════════════════════════════════════════════


@dataclass
class DofManager:
    """Manages equation numbering with dynamic rotational-DOF omission.

    Attributes
    ----------
    active_map : dict[int, dict[str, int | None]]
        For each node, maps "ux"/"uy"/"rz" to a global DOF index
        (0-based), or None if the DOF does not exist.
    free_indices : list[int]
        Indices of free (unconstrained) DOFs.
    restrained_indices : list[int]
        Indices of restrained DOFs.
    labels : dict[int, str]
        Human-readable label for each DOF index.
    n_total : int
        Total number of active DOFs (free + restrained).
    """

    active_map: dict[int, dict[str, int | None]]
    free_indices: list[int]
    restrained_indices: list[int]
    labels: dict[int, str]
    n_total: int
    # Per-node DOF name order for this model — DOF_NAMES_2D for the
    # legacy planar pipeline, DOF_NAMES_3D for the 6-DOF space solve.
    dof_names: tuple[str, ...] = DOF_NAMES_2D

    @property
    def is_3d(self) -> bool:
        """True when this numbering uses the 6-DOF-per-node space set."""
        return len(self.dof_names) == 6

    @classmethod
    def from_model(
        cls,
        model: StructuralModel,
        elements: list | None = None,
        is_3d: bool | None = None,
    ) -> DofManager:
        """Build DOF numbering from the model.

        Rotational DOFs are only created at a node if:
        1. A frame element with an unreleased end connects there, OR
        2. A support restrains the rotation there, OR
        3. A nodal load has a non-zero moment there.

        Args:
            model: The structural model to number DOFs for.
            elements: Solve-time element list (promoted in 3D mode).
                Defaults to ``prepare_solve_elements(model)``.
            is_3d: Pipeline override; auto-detected when None.

        Returns:
            A new DofManager instance with all DOF mappings computed.
        """
        if elements is None or is_3d is None:
            is_3d, elements = prepare_solve_elements(model)
        if is_3d:
            return cls._from_model_3d(model, elements)
        rotation_active: dict[int, bool] = defaultdict(bool)

        for elem in elements:
            if isinstance(elem, FrameElement2D):
                if not elem.release_i:
                    rotation_active[elem.node_i] = True
                if not elem.release_j:
                    rotation_active[elem.node_j] = True

        for nid, sup in model.supports.items():
            if sup.rz:
                rotation_active[nid] = True

        for load in model.nodal_loads:
            if abs(load.mz) > 0:
                rotation_active[load.node_id] = True

        active_map: dict[int, dict[str, int | None]] = {}
        labels: dict[int, str] = {}
        idx = 0

        for nid in model.node_ids:
            node_map: dict[str, int | None] = {}
            for dof in ("ux", "uy"):
                node_map[dof] = idx
                labels[idx] = f"Node {nid} {dof.upper()}"
                idx += 1
            if rotation_active[nid]:
                node_map["rz"] = idx
                labels[idx] = f"Node {nid} RZ"
                idx += 1
            else:
                node_map["rz"] = None
            active_map[nid] = node_map

        restrained: list[int] = []
        for nid in model.node_ids:
            sup = model.support_for(nid)
            for dof, flag in [("ux", sup.ux), ("uy", sup.uy), ("rz", sup.rz)]:
                i = active_map[nid].get(dof)
                if i is not None and flag:
                    restrained.append(i)
        restrained = sorted(set(restrained))
        free = [i for i in range(idx) if i not in set(restrained)]

        return cls(
            active_map=active_map,
            free_indices=free,
            restrained_indices=restrained,
            labels=labels,
            n_total=idx,
        )

    @classmethod
    def _from_model_3d(
        cls, model: StructuralModel, elements: list,
    ) -> DofManager:
        """6-DOF-per-node numbering for the space pipeline.

        Translations (ux, uy, uz) are always active. Rotations follow
        the same dynamic-omission rule as the 2D pipeline, per axis:

        * rx / ry activate where any :class:`FrameElement3D` connects
          (torsion and out-of-plane bending always couple there);
        * rz activates where a frame connects with an UNRELEASED end
          (a hinge about local z mirrors the 2D moment release);
        * a support restraining a rotation, or a nodal moment about an
          axis, activates that axis explicitly.

        Pure space-truss nodes therefore carry only translations —
        no singular rotational equations, same UX as the 2D solver.
        """
        rot_active: dict[int, set[str]] = defaultdict(set)

        for elem in elements:
            if isinstance(elem, FrameElement3D):
                for nid in (elem.node_i, elem.node_j):
                    rot_active[nid].update(("rx", "ry"))
                if not elem.release_i:
                    rot_active[elem.node_i].add("rz")
                if not elem.release_j:
                    rot_active[elem.node_j].add("rz")

        for nid, sup in model.supports.items():
            for dof in ("rx", "ry", "rz"):
                if getattr(sup, dof):
                    rot_active[nid].add(dof)

        for load in model.nodal_loads:
            for dof, val in (("rx", load.mx), ("ry", load.my),
                             ("rz", load.mz)):
                if abs(val) > 0:
                    rot_active[load.node_id].add(dof)

        active_map: dict[int, dict[str, int | None]] = {}
        labels: dict[int, str] = {}
        idx = 0
        for nid in model.node_ids:
            node_map: dict[str, int | None] = {}
            for dof in DOF_NAMES_3D:
                if dof.startswith("r") and dof not in rot_active[nid]:
                    node_map[dof] = None
                    continue
                node_map[dof] = idx
                labels[idx] = f"Node {nid} {dof.upper()}"
                idx += 1
            active_map[nid] = node_map

        restrained: list[int] = []
        for nid in model.node_ids:
            sup = model.support_for(nid)
            for dof in DOF_NAMES_3D:
                i = active_map[nid].get(dof)
                if i is not None and getattr(sup, dof):
                    restrained.append(i)
        restrained = sorted(set(restrained))
        restrained_set = set(restrained)
        free = [i for i in range(idx) if i not in restrained_set]

        return cls(
            active_map=active_map,
            free_indices=free,
            restrained_indices=restrained,
            labels=labels,
            n_total=idx,
            dof_names=DOF_NAMES_3D,
        )

    # ── helpers ──

    def index(self, node_id: int, dof: str) -> int | None:
        """Return the global DOF index for a given node and DOF name.

        Args:
            node_id: The node identifier.
            dof: DOF name — "ux", "uy", or "rz".

        Returns:
            Global DOF index (0-based int), or None if the DOF is inactive.
        """
        return self.active_map[node_id][dof]

    def element_dof_map(self, elem) -> list[int | None]:
        """Build the DOF address vector for an element.

        Args:
            elem: The element to build the DOF map for (2D or 3D).

        Returns:
            List of global DOF indices (or None for inactive DOFs),
            one per local DOF — 6 entries for 2D elements, 12 for a
            space frame, 6 for a space truss.
        """
        local_mask = elem.assembly_local_indices()
        keys = elem.dof_keys()
        mapping: list[int | None] = []
        for pos, key in enumerate(keys):
            if local_mask[pos] is None:
                mapping.append(None)
            else:
                mapping.append(self.index(*key))
        return mapping

    def e_matrix_for_display(self, model: StructuralModel) -> dict[int, list[int]]:
        """Build E matrix in course notation (1-based, 0 = restrained/inactive).

        Args:
            model: The structural model (used for node iteration order).

        Returns:
            Dict mapping node_id to one equation number per entry of
            ``self.dof_names`` ([Tx, Ty, Rz] in 2D; six entries in 3D)
            where 0 means restrained or inactive and positive integers
            are free equation numbers.
        """
        # Build reverse map: global_index -> equation_number (1-based among free)
        free_set = set(self.free_indices)
        eq_number: dict[int, int] = {}
        eq = 1
        for idx in sorted(self.free_indices):
            eq_number[idx] = eq
            eq += 1

        E: dict[int, list[int]] = {}
        for nid in model.node_ids:
            row = []
            for dof in self.dof_names:
                gi = self.active_map[nid].get(dof)
                if gi is None or gi not in free_set:
                    row.append(0)
                else:
                    row.append(eq_number[gi])
            E[nid] = row
        return E

    def g_vector_for_display(self, elem: Element2D) -> list[int]:
        """G vector in course notation (1-based, 0 = restrained/inactive).

        Args:
            elem: The element to build the G vector for.

        Returns:
            List of 6 integers — equation numbers (1-based) or 0.
        """
        dof_map = self.element_dof_map(elem)
        free_set = set(self.free_indices)
        eq_number = {}
        eq = 1
        for idx in sorted(self.free_indices):
            eq_number[idx] = eq
            eq += 1

        G: list[int] = []
        for gi in dof_map:
            if gi is None or gi not in free_set:
                G.append(0)
            else:
                G.append(eq_number[gi])
        return G


# ═══════════════════════════════════════════════════════════════
#  Connectivity check (DFS)
# ═══════════════════════════════════════════════════════════════


def _connectivity_components(model: StructuralModel) -> list[set[int]]:
    """Find connected components via DFS on the node-element graph.

    Args:
        model: The structural model to check connectivity for.

    Returns:
        List of sets, each containing the node IDs in one connected component.
    """
    graph: dict[int, set[int]] = {nid: set() for nid in model.node_ids}
    for elem in model.elements:
        graph[elem.node_i].add(elem.node_j)
        graph[elem.node_j].add(elem.node_i)

    seen: set[int] = set()
    components: list[set[int]] = []
    for start in model.node_ids:
        if start in seen:
            continue
        stack = [start]
        comp: set[int] = set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            comp.add(n)
            stack.extend(graph[n] - seen)
        components.append(comp)
    return components


def _find_coincident_node_pairs(
    model: StructuralModel,
    *,
    tol: float = NODE_COINCIDENCE_TOL,
) -> list[tuple[int, int, float]]:
    """Return pairs of nodes within ``tol`` of each other (id_a < id_b).

    Returns a list of ``(id_a, id_b, distance)`` tuples. The
    O(n²) pairwise comparison is fine for typical model sizes
    (< few hundred nodes); for larger imports a spatial index would
    help but is out of scope for the Stage B-lite minimal audit.

    Used by :func:`validate_model` to emit a non-fatal warning when
    the model has duplicate nodes that the add-time block in
    ``AddNodeCmd`` didn't catch (most commonly: nodes that drifted
    into coincidence through file import or manual moves).
    """
    nodes = list(model.nodes.values())
    pairs: list[tuple[int, int, float]] = []
    for i, ni in enumerate(nodes):
        for nj in nodes[i + 1:]:
            dx = ni.x - nj.x
            dy = ni.y - nj.y
            dz = getattr(ni, "z", 0.0) - getattr(nj, "z", 0.0)
            if abs(dz) >= tol:
                continue
            # Cheap bounding-box pre-filter rules out the obviously-
            # distant majority before the hypot call. The strict
            # Euclidean check below is what actually decides whether
            # the pair counts as coincident — using only the box
            # check could flag pairs whose Euclidean distance reaches
            # √2·tol, contradicting the warning message's "Δ ≤ tol"
            # claim (per gemini PR-21 review).
            if abs(dx) >= tol or abs(dy) >= tol:
                continue
            dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            if dist >= tol:
                continue
            a, b = (ni.id, nj.id) if ni.id < nj.id else (nj.id, ni.id)
            pairs.append((a, b, dist))
    return pairs


# ═══════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════


def validate_model(
    model: StructuralModel,
    dofs: DofManager,
    elements: list | None = None,
) -> list[str]:
    """Pre-assembly validation of the structural model.

    Checks for: undefined nodes, non-positive properties, zero-length
    elements, isolated nodes, disconnected unsupported components,
    and free rotational DOFs with no bending stiffness.

    Args:
        model: The structural model to validate.
        dofs: The DOF manager (used for free-DOF checks).
        elements: Solve-time element list — in 3D mode the promoted
            space elements (their length math is z-aware). Defaults to
            ``model.elements`` (the legacy 2D behaviour).

    Returns:
        List of warning strings. Non-fatal issues are returned as warnings.

    Raises:
        ValueError: On fatal errors (isolated nodes, floating components,
            non-positive properties, undefined nodes).
    """
    warnings: list[str] = []
    if elements is None:
        elements = model.elements

    if not model.nodes:
        raise ValueError("No nodes defined.")
    if not elements:
        raise ValueError("No elements defined.")

    # Element checks
    incidence: dict[int, int] = defaultdict(int)
    for elem in elements:
        if elem.node_i not in model.nodes or elem.node_j not in model.nodes:
            raise ValueError(f"Element {elem.id} references undefined node.")
        if elem.A <= 0 or elem.E <= 0:
            raise ValueError(f"Element {elem.id} has non-positive A or E.")
        if isinstance(elem, FrameElement2D) and elem.I <= 0:
            raise ValueError(f"Frame element {elem.id} has non-positive I.")
        if isinstance(elem, FrameElement3D):
            for prop in ("Iy", "Iz", "J", "G"):
                if getattr(elem, prop) <= 0:
                    raise ValueError(
                        f"3D frame element {elem.id} has non-positive "
                        f"{prop}."
                    )
        elem.length_cos_sin(model.nodes)  # raises if zero-length
        incidence[elem.node_i] += 1
        incidence[elem.node_j] += 1

    # Isolated nodes
    isolated = [nid for nid in model.node_ids if incidence[nid] == 0]
    if isolated:
        raise ValueError(
            f"Isolated nodes with no elements: {isolated}. "
            f"Remove them or connect them."
        )

    # v0.11.0 Stage B-lite: coincident-node audit. The add-time block
    # in `AddNodeCmd` prevents new duplicates, but nothing today
    # catches duplicates introduced by file import or by manual node
    # moves. Surfaced as a non-fatal warning so existing fatal checks
    # (orphan / disconnected-unsupported) stay fatal — only the
    # missing-from-today's-validation case is added.
    pairs = _find_coincident_node_pairs(model, tol=NODE_COINCIDENCE_TOL)
    if pairs:
        shown = pairs[:_MAX_COINCIDENT_PAIRS_IN_WARNING]
        suffix = (
            f" …(+{len(pairs) - _MAX_COINCIDENT_PAIRS_IN_WARNING} more)"
            if len(pairs) > _MAX_COINCIDENT_PAIRS_IN_WARNING
            else ""
        )
        pair_text = ", ".join(f"({a}, {b})" for a, b, _ in shown) + suffix
        warnings.append(
            f"Coincident nodes detected "
            f"(Δ ≤ {NODE_COINCIDENCE_TOL:.0e} m): pairs {pair_text}."
        )

    # Connectivity — each component must have at least one support
    components = _connectivity_components(model)
    for comp in components:
        has_support = any(
            model.support_for(nid).any_restrained for nid in comp
        )
        if not has_support:
            raise ValueError(
                f"Disconnected component {sorted(comp)} has no supports — "
                f"rigid-body motion, stiffness matrix is singular."
            )

    if len(components) > 1:
        warnings.append(
            f"Model has {len(components)} disconnected but supported components. "
            f"K is block-diagonal."
        )

    # 3D pipeline: a model promoted from a planar drawing often still
    # carries 2D-only supports — nothing restrains the out-of-plane
    # rigid-body modes and K_ff is guaranteed singular. Surface the
    # actionable hint here instead of letting the SVD mechanism report
    # be the first clue.
    if dofs.is_3d and not any(
        s.uz for s in model.supports.values()
    ):
        warnings.append(
            "3D analysis: no support restrains UZ — the structure can "
            "translate freely out of plane (singular K). Add UZ (and "
            "where needed RX/RY) restraints to the supports."
        )

    # Free rotational DOFs with no bending stiffness
    for idx in dofs.free_indices:
        label = dofs.labels[idx]
        parts = label.split()
        dof_name = parts[-1]  # "UX" … "RZ"
        if not dof_name.startswith("R"):
            continue
        nid = int(parts[1])
        if dof_name == "RZ":
            has_stiffness = any(
                isinstance(el, (FrameElement2D, FrameElement3D)) and (
                    (el.node_i == nid and not el.release_i) or
                    (el.node_j == nid and not el.release_j)
                )
                for el in elements
            )
        else:  # RX / RY — any connected space frame is stiff there
            has_stiffness = any(
                isinstance(el, FrameElement3D)
                and nid in (el.node_i, el.node_j)
                for el in elements
            )
        if not has_stiffness:
            warnings.append(
                f"{label} is free but has no bending stiffness — "
                f"should be restrained or will cause singular K."
            )

    return warnings


# ═══════════════════════════════════════════════════════════════
#  Assembly
# ═══════════════════════════════════════════════════════════════


def assemble_global_system(
    model: StructuralModel,
) -> tuple[np.ndarray, np.ndarray, DofManager, list[str], dict]:
    """Assemble global stiffness matrix K and load vector F.

    Args:
        model: The structural model containing nodes, elements,
            supports, and loads.

    Returns:
        Tuple of (K, F, dofs, warnings, elem_data) where:
            K: ndarray (n_total × n_total) — global stiffness matrix.
            F: ndarray (n_total,) — global load vector.
            dofs: DofManager — DOF numbering information.
            warnings: list[str] — validation warnings.
            elem_data: dict[int, dict] — per-element data for postprocessing.
    """
    is_3d, solve_elements = prepare_solve_elements(model)
    dofs = DofManager.from_model(model, solve_elements, is_3d)
    warnings = validate_model(model, dofs, solve_elements)

    n = dofs.n_total
    K = np.zeros((n, n))
    F = np.zeros(n)
    elem_data: dict[int, dict] = {}

    # Nodal loads
    load_components = [("ux", "fx"), ("uy", "fy"), ("rz", "mz")]
    if is_3d:
        load_components += [("uz", "fz"), ("rx", "mx"), ("ry", "my")]
    for load in model.nodal_loads:
        for dof, attr in load_components:
            idx = dofs.index(load.node_id, dof)
            if idx is not None:
                F[idx] += getattr(load, attr)

    # Element contributions. In 3D mode ``solve_elements`` holds the
    # promoted space elements; ``elem_data["element"]`` stores the
    # SOLVE element (the one whose stiffness was assembled) so the
    # postprocessor recovers forces with the same matrices.
    for elem in solve_elements:
        k_global, p_global = elem.global_stiffness_and_load(model.nodes)
        mapping = dofs.element_dof_map(elem)

        for a, I in enumerate(mapping):
            if I is None:
                continue
            F[I] += p_global[a]
            for b, J in enumerate(mapping):
                if J is None:
                    continue
                K[I, J] += k_global[a, b]

        # Store per-element data
        L, c, s = elem.length_cos_sin(model.nodes)
        elem_data[elem.id] = {
            "element": elem,
            "mapping": mapping,
            "L": L, "c": c, "s": s,
        }

    # Self-weight pass — applied directly to F, never persisted to the model.
    # Per-element raw fixed-end vectors are stashed in ``elem_data`` so the
    # postprocessor can feed them into ``q = K·d − p`` recovery without
    # the loads ever being attached as member_loads.
    if model.include_self_weight:
        _apply_self_weight(model, dofs, F, elem_data,
                           elements=solve_elements)

    return K, F, dofs, warnings, elem_data


def _apply_self_weight(
    model: StructuralModel,
    dofs: DofManager,
    F: np.ndarray,
    elem_data: dict[int, dict],
    elements: list | None = None,
) -> None:
    """Inject gravity loads on every element into the global F vector.

    Gravity is hard-coded to global -Y at g = STANDARD_GRAVITY in v0.9.0.

    Frame elements get a full local 6-DOF fixed-end force vector built
    from the same equivalent-load sign convention as the UDL path in
    ``FrameElement2D.assembled_local_stiffness_and_load`` (see
    element.py UDL block) — i.e. ``p`` is the equivalent nodal load
    that is *added* to F, not the fixed-end reaction. The local
    components of gravity are derived from the element's orientation:
    a body-force of magnitude ``w = ρ·A·g/1000`` (kN/m) in global -Y
    projects to ``w_local_x = -w·sin θ`` (axial) and
    ``w_local_y = -w·cos θ`` (transverse), then transformed back to
    global via ``Rᵀ``.

    Truss elements take half the bar weight lumped at each endpoint
    in global -Y. This bypasses the existing TrussElement2D invariant
    that rejects member loads (only TrussTemperatureLoad is allowed),
    because the contribution is applied directly to F at the uy DOF
    of each endpoint — no member-load object is ever attached.

    Elements with ``ρ = 0`` or ``A = 0`` contribute nothing.

    In 3D mode ``elements`` carries the promoted space elements:
    frames build their fixed-end vector through
    ``FrameElement3D.self_weight_fixed_end_local`` (gravity stays
    global −Y); trusses lump half the bar weight at each endpoint's
    uy DOF exactly as in 2D.
    """
    g = STANDARD_GRAVITY
    if elements is None:
        elements = model.elements
    for elem in elements:
        rho = float(getattr(elem, "rho", 0.0))
        A = float(getattr(elem, "A", 0.0))
        if rho == 0.0 or A == 0.0:
            continue

        L, c, s = elem.length_cos_sin(model.nodes)
        if L <= 0.0:
            continue

        w = rho * A * g / 1000.0  # kN/m, magnitude in global -Y

        if isinstance(elem, FrameElement3D):
            p_local_raw = elem.self_weight_fixed_end_local(model.nodes)
            elem_data[elem.id]["self_weight_p_local"] = p_local_raw
            p_local = elem.condense_local_load_for_releases(
                p_local_raw, model.nodes,
            )
            R = elem.transformation_matrix(model.nodes)
            p_global = R.T @ p_local
            mapping = dofs.element_dof_map(elem)
            for a, I in enumerate(mapping):
                if I is None:
                    continue
                F[I] += p_global[a]
        elif isinstance(elem, TrussElement3D):
            half_kN = w * L / 2.0
            for nid in (elem.node_i, elem.node_j):
                idx = dofs.index(nid, "uy")
                if idx is not None:
                    F[idx] -= half_kN
        elif isinstance(elem, FrameElement2D):
            # Built by the element so rigid end offsets are honoured
            # (flexible-span weight, mapped to joint coordinates via
            # Tᵀ); identical to the legacy inline wL/2, ±wL²/12 vector
            # when offsets are zero.
            p_local_raw = elem.self_weight_fixed_end_local(model.nodes)
            # Stash the RAW (uncondensed) fixed-end vector so the
            # postprocessor can include it in q = K·d − p recovery. The
            # back-substitution path needs p_b at released DOFs, so the
            # condensed (zero-at-released) version is the wrong thing to
            # hand off — see Element2D.local_displacement_and_end_forces.
            elem_data[elem.id]["self_weight_p_local"] = p_local_raw
            # Released rotational DOFs are unassembled (mapping[r]=None);
            # without this Schur reduction the released-end moment FEF
            # would be silently dropped instead of redistributed to the
            # retained DOFs.
            p_local = elem.condense_local_load_for_releases(p_local_raw, model.nodes)
            R = elem.transformation_matrix(model.nodes)
            p_global = R.T @ p_local
            mapping = dofs.element_dof_map(elem)
            for a, I in enumerate(mapping):
                if I is None:
                    continue
                F[I] += p_global[a]
        elif isinstance(elem, TrussElement2D):
            half_kN = w * L / 2.0
            for nid in (elem.node_i, elem.node_j):
                idx = dofs.index(nid, "uy")
                if idx is not None:
                    F[idx] -= half_kN
