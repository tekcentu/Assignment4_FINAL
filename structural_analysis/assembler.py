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
from .model import NODE_COINCIDENCE_TOL, STANDARD_GRAVITY, StructuralModel


# Cap the coincident-pair list shown in the warning to keep the
# results panel readable on pathological imports. The full set is
# always available by re-running the check.
_MAX_COINCIDENT_PAIRS_IN_WARNING: int = 10


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

    @classmethod
    def from_model(cls, model: StructuralModel) -> DofManager:
        """Build DOF numbering from the model.

        Rotational DOFs are only created at a node if:
        1. A FrameElement2D with an unreleased end connects there, OR
        2. A support restrains Rz there, OR
        3. A nodal load has a non-zero Mz there.

        Args:
            model: The structural model to number DOFs for.

        Returns:
            A new DofManager instance with all DOF mappings computed.
        """
        rotation_active: dict[int, bool] = defaultdict(bool)

        for elem in model.elements:
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

    def element_dof_map(self, elem: Element2D) -> list[int | None]:
        """Build the 6-entry DOF address vector for an element.

        Args:
            elem: The element to build the DOF map for.

        Returns:
            List of 6 global DOF indices (or None for inactive DOFs).
        """
        local_mask = elem.assembly_local_indices()
        keys = [
            (elem.node_i, "ux"), (elem.node_i, "uy"), (elem.node_i, "rz"),
            (elem.node_j, "ux"), (elem.node_j, "uy"), (elem.node_j, "rz"),
        ]
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
            Dict mapping node_id to [eq_tx, eq_ty, eq_rz] where 0 means
            restrained or inactive and positive integers are free equation numbers.
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
            for dof in ("ux", "uy", "rz"):
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
            # Cheap bounding-box pre-filter rules out the obviously-
            # distant majority before the hypot call. The strict
            # Euclidean check below is what actually decides whether
            # the pair counts as coincident — using only the box
            # check could flag pairs whose Euclidean distance reaches
            # √2·tol, contradicting the warning message's "Δ ≤ tol"
            # claim (per gemini PR-21 review).
            if abs(dx) >= tol or abs(dy) >= tol:
                continue
            dist = float(np.hypot(dx, dy))
            if dist >= tol:
                continue
            a, b = (ni.id, nj.id) if ni.id < nj.id else (nj.id, ni.id)
            pairs.append((a, b, dist))
    return pairs


# ═══════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════


def validate_model(model: StructuralModel, dofs: DofManager) -> list[str]:
    """Pre-assembly validation of the structural model.

    Checks for: undefined nodes, non-positive properties, zero-length
    elements, isolated nodes, disconnected unsupported components,
    and free rotational DOFs with no bending stiffness.

    Args:
        model: The structural model to validate.
        dofs: The DOF manager (used for free-DOF checks).

    Returns:
        List of warning strings. Non-fatal issues are returned as warnings.

    Raises:
        ValueError: On fatal errors (isolated nodes, floating components,
            non-positive properties, undefined nodes).
    """
    warnings: list[str] = []

    if not model.nodes:
        raise ValueError("No nodes defined.")
    if not model.elements:
        raise ValueError("No elements defined.")

    # Element checks
    incidence: dict[int, int] = defaultdict(int)
    for elem in model.elements:
        if elem.node_i not in model.nodes or elem.node_j not in model.nodes:
            raise ValueError(f"Element {elem.id} references undefined node.")
        if elem.A <= 0 or elem.E <= 0:
            raise ValueError(f"Element {elem.id} has non-positive A or E.")
        if isinstance(elem, FrameElement2D) and elem.I <= 0:
            raise ValueError(f"Frame element {elem.id} has non-positive I.")
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
            model.support_for(nid).ux or model.support_for(nid).uy or model.support_for(nid).rz
            for nid in comp
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

    # Free rotational DOFs with no bending stiffness
    for idx in dofs.free_indices:
        label = dofs.labels[idx]
        if "RZ" not in label:
            continue
        nid = int(label.split()[1])
        has_stiffness = any(
            isinstance(el, FrameElement2D) and (
                (el.node_i == nid and not el.release_i) or
                (el.node_j == nid and not el.release_j)
            )
            for el in model.elements
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
    dofs = DofManager.from_model(model)
    warnings = validate_model(model, dofs)

    n = dofs.n_total
    K = np.zeros((n, n))
    F = np.zeros(n)
    elem_data: dict[int, dict] = {}

    # Nodal loads
    for load in model.nodal_loads:
        for dof, val in [("ux", load.fx), ("uy", load.fy), ("rz", load.mz)]:
            idx = dofs.index(load.node_id, dof)
            if idx is not None:
                F[idx] += val

    # Element contributions
    for elem in model.elements:
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
        _apply_self_weight(model, dofs, F, elem_data)

    return K, F, dofs, warnings, elem_data


def _apply_self_weight(
    model: StructuralModel,
    dofs: DofManager,
    F: np.ndarray,
    elem_data: dict[int, dict],
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
    """
    g = STANDARD_GRAVITY
    for elem in model.elements:
        rho = float(getattr(elem, "rho", 0.0))
        A = float(getattr(elem, "A", 0.0))
        if rho == 0.0 or A == 0.0:
            continue

        L, c, s = elem.length_cos_sin(model.nodes)
        if L <= 0.0:
            continue

        w = rho * A * g / 1000.0  # kN/m, magnitude in global -Y

        if isinstance(elem, FrameElement2D):
            w_local_x = -w * s
            w_local_y = -w * c
            p_local_raw = np.array([
                w_local_x * L / 2.0,
                w_local_y * L / 2.0,
                w_local_y * L ** 2 / 12.0,
                w_local_x * L / 2.0,
                w_local_y * L / 2.0,
                -w_local_y * L ** 2 / 12.0,
            ])
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
