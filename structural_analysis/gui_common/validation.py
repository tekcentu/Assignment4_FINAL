"""Model validation for the pre-solve safety pass (PR #31).

Pure-Python helpers that examine a :class:`StructuralModel` and report
problems that would either (a) make the stiffness matrix singular or
(b) leave the user with a model that solves but isn't what they meant.

The module has no Qt / matplotlib dependencies so it can be unit-tested
in isolation; the GUI consumes :func:`validate_model` and pushes the
resulting :class:`ModelValidationResult` to the canvas highlight layer
and the result-text panel.

Detections shipped here:

* **Orphan node** — a node with no incident element (warning).
* **Disconnected unsupported component** — a connected component of
  elements whose nodes carry no restraint (error: rigid-body motion).
* **Axial-only free-end mechanism** — an unsupported node whose every
  incident element provides *only axial stiffness* at that node, and
  those elements' directions don't span 2-D (error: unconstrained
  transverse DOF).  "Axial-only" covers two cases:

  - **Truss elements** — no bending or shear stiffness by definition.
  - **Double-pinned frame elements** (``release_i=True`` *and*
    ``release_j=True``) — both moment releases cause Schur condensation
    to reduce the frame to a truss equivalent: axial force only.

  A frame element with a release at only *one* end still carries shear
  at the query node and therefore stabilises it.  Two non-collinear
  axial-only members at the same free node span 2-D and are not flagged.

* **Single-release rigid-body rotation** — a free node N connected by
  exactly one frame element whose OPPOSITE (far) end carries a moment
  release (pin) at a directly-supported node M.  The pin in the element
  at M allows the element to rotate as a rigid body about M regardless
  of any rotational restraint at M (the pin decouples element rotation
  from node rotation).  Stiffness matrix for N's free DOFs is provably
  singular.  Caught only for the obvious single-element topology; the
  multi-element generalisation is deferred.

Active-load-case filtering for "Solve All Cases":

* :func:`cases_with_loads` returns the sorted list of *enabled* cases
  that actually carry at least one source of load (nodal, member, or
  self-weight if self-weight is enabled and assigned to that case).
  Used to skip empty WIND / THERMAL placeholders during a full solve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import StructuralModel


# Tolerance for two truss bars being considered collinear.  Two unit
# direction vectors d1 and d2 are collinear iff |d1 × d2| < eps.  At
# 1e-6 this catches floating-point dust but still flags bars that
# differ by ~0.06° as non-collinear — fine for engineering geometry.
_COLLINEAR_EPS = 1e-6


@dataclass
class ValidationIssue:
    """A single problem the validator found.

    ``node_ids`` / ``element_ids`` carry the model objects the GUI
    should paint as the problem location — the canvas highlight layer
    aggregates these from every issue.
    """

    severity: str  # "error" | "warning"
    message: str
    node_ids: list[int] = field(default_factory=list)
    element_ids: list[int] = field(default_factory=list)


@dataclass
class ModelValidationResult:
    """Aggregate of every issue :func:`validate_model` produced."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)

    @property
    def error_node_ids(self) -> set[int]:
        out: set[int] = set()
        for i in self.issues:
            if i.severity == "error":
                out.update(i.node_ids)
        return out

    @property
    def warning_node_ids(self) -> set[int]:
        out: set[int] = set()
        for i in self.issues:
            if i.severity == "warning":
                out.update(i.node_ids)
        return out

    @property
    def error_element_ids(self) -> set[int]:
        out: set[int] = set()
        for i in self.issues:
            if i.severity == "error":
                out.update(i.element_ids)
        return out

    @property
    def warning_element_ids(self) -> set[int]:
        out: set[int] = set()
        for i in self.issues:
            if i.severity == "warning":
                out.update(i.element_ids)
        return out

    def format_report(self) -> str:
        """Render a human-readable block for the result-text panel."""
        if not self.issues:
            return ""
        lines: list[str] = []
        warns = [i for i in self.issues if i.severity == "warning"]
        errs = [i for i in self.issues if i.severity == "error"]
        if warns:
            lines.append("Validation warnings:")
            for w in warns:
                lines.append(f"  • {w.message}")
        if errs:
            if lines:
                lines.append("")
            lines.append("Validation errors:")
            for e in errs:
                lines.append(f"  • {e.message}")
            lines.append("")
            lines.append("Analysis blocked because the model is unstable.")
        return "\n".join(lines)


# ── helpers ───────────────────────────────────────────────────────────


def _node_is_supported(model: "StructuralModel", node_id: int) -> bool:
    """True iff at least one DOF at the node is restrained."""
    s = model.supports.get(node_id)
    if s is None:
        return False
    return bool(s.ux or s.uy or s.rz)


def _element_is_truss(elem) -> bool:
    """True iff the element behaves as a truss bar (no transverse / rotational stiffness)."""
    kind = getattr(elem, "kind", None)
    return kind == "truss"


def _is_axial_only_at_node(elem, _node_id: int) -> bool:
    """True iff *elem* contributes only axial stiffness at the query node.

    Two element classes qualify:

    * **Truss elements** — always axial-only (their stiffness matrix has
      no shear terms regardless of end conditions).
    * **Frame elements with releases at both ends** (``release_i=True``
      *and* ``release_j=True``) — the Schur-complement condensation of
      both rotational DOFs reduces the 6×6 frame stiffness to a 4×4
      matrix that is mathematically identical to a truss: no off-axis
      stiffness survives.

    A frame with a release at only one end still carries transverse shear
    at both nodes — its condensed stiffness has non-zero transverse terms
    — so it is *not* axial-only.
    """
    if _element_is_truss(elem):
        return True
    return bool(
        getattr(elem, "release_i", False) and getattr(elem, "release_j", False)
    )


def _adjacency(model: "StructuralModel") -> dict[int, set[int]]:
    """Node-id → set of neighbour node ids via elements."""
    adj: dict[int, set[int]] = {nid: set() for nid in model.nodes}
    for elem in model.elements:
        i, j = elem.node_i, elem.node_j
        if i in adj and j in adj:
            adj[i].add(j)
            adj[j].add(i)
    return adj


def _connected_components(model: "StructuralModel") -> list[set[int]]:
    """BFS components over the node-graph induced by elements.

    Returns a list of node-id sets, one per component.  Orphan nodes
    (those with no incident element) form a singleton component each
    so the caller can decide what to do with them (currently: they're
    reported separately by :func:`_find_orphan_nodes`, and isolated
    singletons skip the supported-component check below).
    """
    adj = _adjacency(model)
    seen: set[int] = set()
    comps: list[set[int]] = []
    for start in model.nodes:
        if start in seen:
            continue
        # BFS
        comp: set[int] = set()
        stack = [start]
        while stack:
            n = stack.pop()
            if n in comp:
                continue
            comp.add(n)
            for nb in adj.get(n, ()):
                if nb not in comp:
                    stack.append(nb)
        seen |= comp
        comps.append(comp)
    return comps


def _elements_in_component(
    model: "StructuralModel", component: set[int],
) -> list[int]:
    return [
        e.id for e in model.elements
        if e.node_i in component and e.node_j in component
    ]


def _incident_elements(model: "StructuralModel", node_id: int) -> list:
    return [
        e for e in model.elements
        if e.node_i == node_id or e.node_j == node_id
    ]


def _bar_direction_from_node(elem, node_id: int, nodes: dict) -> tuple[float, float]:
    """Unit vector along the element, *away from* ``node_id``."""
    i_node = nodes[elem.node_i]
    j_node = nodes[elem.node_j]
    if elem.node_i == node_id:
        dx, dy = j_node.x - i_node.x, j_node.y - i_node.y
    else:
        dx, dy = i_node.x - j_node.x, i_node.y - j_node.y
    L = math.hypot(dx, dy)
    if L == 0.0:
        return (0.0, 0.0)
    return (dx / L, dy / L)


def _directions_span_2d(dirs: list[tuple[float, float]]) -> bool:
    """True iff at least two of the unit vectors are non-collinear.

    Collinearity test: |d_i × d_j| > eps for some pair.  Zero-length
    direction vectors (degenerate elements) are ignored.
    """
    nz = [d for d in dirs if d != (0.0, 0.0)]
    if len(nz) < 2:
        return False
    base = nz[0]
    for d in nz[1:]:
        cross = base[0] * d[1] - base[1] * d[0]
        if abs(cross) > _COLLINEAR_EPS:
            return True
    return False


# ── individual checks ────────────────────────────────────────────────


def _find_orphan_nodes(model: "StructuralModel") -> list[ValidationIssue]:
    """Nodes with no incident element.

    These are warnings, not errors: an orphan node carries no DOFs in
    the active system because no element references it, so the
    stiffness matrix isn't enlarged by it.  Still worth reporting
    because the user almost certainly didn't mean to leave it floating.
    """
    used: set[int] = set()
    for elem in model.elements:
        used.add(elem.node_i)
        used.add(elem.node_j)
    orphans = sorted(set(model.nodes) - used)
    return [
        ValidationIssue(
            severity="warning",
            message=f"Node {nid} is not connected to any element.",
            node_ids=[nid],
        )
        for nid in orphans
    ]


def _find_unsupported_components(
    model: "StructuralModel",
) -> list[ValidationIssue]:
    """Connected components of elements with no restrained node.

    A component whose every node has no support entry (or all-False
    support DOFs) admits rigid-body motion and makes the stiffness
    matrix singular.  Reported as an error so the solve is blocked.

    Singleton components (nodes with no incident element) are skipped
    here — they're reported by :func:`_find_orphan_nodes`.
    """
    issues: list[ValidationIssue] = []
    for comp in _connected_components(model):
        # Skip orphan singletons — handled by the orphan check.  A
        # component of one node with no elements has no DOFs in the
        # active system.
        elem_ids = _elements_in_component(model, comp)
        if not elem_ids:
            continue
        if any(_node_is_supported(model, n) for n in comp):
            continue
        sorted_nodes = sorted(comp)
        issues.append(ValidationIssue(
            severity="error",
            message=(
                f"Component {sorted_nodes} has no supports — "
                f"rigid-body motion; stiffness matrix will be singular."
            ),
            node_ids=sorted_nodes,
            element_ids=sorted(elem_ids),
        ))
    return issues


def _find_single_release_mechanisms(
    model: "StructuralModel",
) -> list[ValidationIssue]:
    """Single-element, single-release rigid-body-rotation mechanism.

    A free node N connected by exactly one frame element E is a mechanism
    when E has a moment release (pin) at its OTHER end M and M is a
    directly-supported node.  The element pin at M decouples E's rotation
    from the node's rotation — rz support at M provides NO help.  The
    3×3 free-DOF stiffness for N (UX, UY, RZ) has a provable zero
    eigenvalue (rigid-body rotation of E about M).

    Only the single-element topology is checked (multi-element cases are
    deferred).  The double-pin case (both ends released) is already caught
    by :func:`_find_truss_mechanisms`.  The double-flag with the
    unsupported-component check is avoided by requiring that M itself be
    a directly-supported node.
    """
    issues: list[ValidationIssue] = []
    for nid in model.nodes:
        if _node_is_supported(model, nid):
            continue
        incident = _incident_elements(model, nid)
        if len(incident) != 1:
            continue  # multi-element topology: mechanism may not exist
        elem = incident[0]
        if _element_is_truss(elem):
            continue  # trusses (and double-pin frames) handled elsewhere
        if _is_axial_only_at_node(elem, nid):
            continue  # double-pin already caught by _find_truss_mechanisms
        # Identify the far end of this element.
        if elem.node_i == nid:
            far_nid = elem.node_j
            release_at_far = getattr(elem, "release_j", False)
        else:
            far_nid = elem.node_i
            release_at_far = getattr(elem, "release_i", False)
        if not release_at_far:
            continue  # no pin at far end → element provides full moment coupling → stable
        # Only flag when the far node is directly supported; if it is not
        # supported, the unsupported-component check already raises an error.
        if not _node_is_supported(model, far_nid):
            continue
        issues.append(ValidationIssue(
            severity="error",
            message=(
                f"Node {nid} is connected only to element {elem.id}, "
                f"which has a moment release (pin) at its far end "
                f"(node {far_nid}). The element can rotate as a rigid "
                f"body about that pin, leaving node {nid} with an "
                f"unconstrained transverse DOF. Add a support at "
                f"node {nid}, connect another stabilising member, or "
                f"remove the release at node {far_nid}."
            ),
            node_ids=[nid],
            element_ids=[elem.id],
        ))
    return issues


def _find_truss_mechanisms(model: "StructuralModel") -> list[ValidationIssue]:
    """Unsupported nodes stabilised only by axial-force-only members
    whose directions don't span 2-D.

    Axial-force-only members are truss elements and double-pinned frame
    elements (both ``release_i`` and ``release_j`` True).  A frame
    element with a release at only one end still carries shear stiffness
    at the query node — its presence prevents this error for that node.

    Two non-collinear axial-only members at a free node span 2-D and are
    NOT flagged (classic triangulated truss joint).
    """
    issues: list[ValidationIssue] = []
    for nid in model.nodes:
        if _node_is_supported(model, nid):
            continue
        incident = _incident_elements(model, nid)
        if not incident:
            continue  # orphan — handled elsewhere
        # If any incident element provides shear/bending stiffness at
        # this node, the node's transverse DOF is stabilised — skip it.
        if any(not _is_axial_only_at_node(e, nid) for e in incident):
            continue
        # All incident elements are axial-only.  Need ≥ 2 non-collinear
        # directions for translational stability.
        dirs = [
            _bar_direction_from_node(e, nid, model.nodes) for e in incident
        ]
        if _directions_span_2d(dirs):
            continue
        # Produce a message that names the specific member type(s).
        has_double_pin = any(not _element_is_truss(e) for e in incident)
        if has_double_pin:
            member_desc = (
                "elements with no transverse stiffness at this node "
                "(truss or double-pinned frame)"
            )
        else:
            member_desc = "truss elements"
        issues.append(ValidationIssue(
            severity="error",
            message=(
                f"Node {nid} is connected only by {member_desc} and has "
                f"an unconstrained transverse DOF. Add a support, connect "
                f"another stabilising member, or remove the double-pin releases."
            ),
            node_ids=[nid],
            element_ids=sorted(e.id for e in incident),
        ))
    return issues


# ── basic structural sanity (replacement for the prior tuple-based pass) ───


def _find_structural_basics(model: "StructuralModel") -> list[ValidationIssue]:
    """Material / section / element-reference sanity.

    These are the checks the v0.21 :func:`_validate_model_for_solve`
    returned in its ``fatal`` list, ported into the new
    :class:`ValidationIssue` shape so the GUI gets uniform handling.
    """
    issues: list[ValidationIssue] = []
    if not model.materials:
        issues.append(ValidationIssue(
            severity="error", message="No materials defined.",
        ))
    if not model.sections:
        issues.append(ValidationIssue(
            severity="error", message="No sections defined.",
        ))
    for sec in model.sections.values():
        if sec.material_id not in model.materials:
            issues.append(ValidationIssue(
                severity="error",
                message=(
                    f"Section {sec.id} references missing material "
                    f"{sec.material_id}."
                ),
            ))
    if not model.elements:
        issues.append(ValidationIssue(
            severity="error", message="Model has no elements.",
        ))
    for elem in model.elements:
        if elem.node_i not in model.nodes:
            issues.append(ValidationIssue(
                severity="error",
                message=(
                    f"Element {elem.id} references missing start "
                    f"node {elem.node_i}."
                ),
                element_ids=[elem.id],
            ))
        if elem.node_j not in model.nodes:
            issues.append(ValidationIssue(
                severity="error",
                message=(
                    f"Element {elem.id} references missing end "
                    f"node {elem.node_j}."
                ),
                element_ids=[elem.id],
            ))
    for ld in model.nodal_loads:
        if ld.node_id not in model.nodes:
            issues.append(ValidationIssue(
                severity="error",
                message=(
                    f"Nodal load references missing node {ld.node_id}."
                ),
            ))
    if not model.supports:
        issues.append(ValidationIssue(
            severity="warning",
            message=(
                "No supports defined — the stiffness matrix will be singular."
            ),
        ))
    return issues


# ── public entrypoint ────────────────────────────────────────────────


def validate_model(model: "StructuralModel") -> ModelValidationResult:
    """Run every validator and bundle the results.

    Basic structural sanity runs first.  If it finds any errors —
    e.g. an element referencing a missing node — the geometry-based
    checks (orphan / component / truss mechanism) would dereference
    invalid ids and crash with ``KeyError``, so we return early.  The
    basic-sanity report alone is enough to tell the user what to fix
    before geometry-level validation can meaningfully run.
    """
    issues: list[ValidationIssue] = []
    basics = _find_structural_basics(model)
    issues.extend(basics)
    if any(i.severity == "error" for i in basics):
        return ModelValidationResult(issues=issues)
    issues.extend(_find_orphan_nodes(model))
    issues.extend(_find_unsupported_components(model))
    issues.extend(_find_truss_mechanisms(model))
    issues.extend(_find_single_release_mechanisms(model))
    return ModelValidationResult(issues=issues)


# ── active-load-case filtering ───────────────────────────────────────


def _has_support_settlement(model: "StructuralModel") -> bool:
    """True iff any support carries a non-zero prescribed displacement.

    Settlements are not case-tagged — they're an intrinsic property of
    the support — but they drive the solver just like applied loads.
    A model whose only "load" is a settlement (e.g. the Q2(a) example
    fixture) still needs to solve.
    """
    for s in model.supports.values():
        for attr in ("settle_ux", "settle_uy", "settle_rz"):
            v = getattr(s, attr, None)
            if v is not None and v != 0.0:
                return True
    return False


def cases_with_loads(model: "StructuralModel") -> list[str]:
    """Enabled cases that actually carry at least one load source.

    Used by the GUI to skip empty cases during "Solve All Cases" so
    the solver doesn't waste a stiffness factorisation on a WIND case
    that has no wind loads.

    A case counts as having loads if any of these is true:

    * a :class:`NodalLoad` references it via ``load_case``;
    * a member load (UDL / point / thermal) references it via
      ``load_case``;
    * self-weight is enabled and assigned to it
      (``model.include_self_weight and case == model.self_weight_case``);
    * a support has a non-zero prescribed displacement (settlement).
      Settlements aren't case-tagged, so they make *every* enabled
      case active — the settlement response shows up under whichever
      case the user selects.

    Cases not present in :attr:`StructuralModel.load_cases` are
    ignored — defensive against legacy fixtures that tagged loads with
    a case name the model doesn't know about.

    Returns:
        Alphabetically-sorted list of case names.
    """
    has_loads: set[str] = set()
    for ld in model.nodal_loads:
        has_loads.add(getattr(ld, "load_case", "DEFAULT"))
    for elem in model.elements:
        for ml in getattr(elem, "member_loads", []):
            has_loads.add(getattr(ml, "load_case", "DEFAULT"))
    if getattr(model, "include_self_weight", False):
        has_loads.add(getattr(model, "self_weight_case", "DEFAULT"))
    enabled = {
        name for name, lc in model.load_cases.items() if lc.enabled
    }
    if _has_support_settlement(model):
        # Settlement is case-independent — every enabled case gets it.
        has_loads.update(enabled)
    return sorted(enabled & has_loads)
