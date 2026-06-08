"""Stage B-lite (v0.11.0): coincident-node pre-solve warning.

Validates the new ``_find_coincident_node_pairs`` helper and its
surfacing as a non-fatal warning inside ``validate_model``. Does not
test the existing fatal checks (orphan / disconnected-unsupported /
zero-length) — those live in tests/test_all.py.
"""

from __future__ import annotations

import pytest

from structural_analysis.assembler import (
    DofManager,
    _find_coincident_node_pairs,
    validate_model,
)
from structural_analysis.element import FrameElement2D
from structural_analysis.model import (
    NODE_COINCIDENCE_TOL,
    Material,
    Node,
    Section,
    StructuralModel,
    Support,
)


def _supported_three_node_frame(
    *,
    coincident: bool = False,
) -> StructuralModel:
    """Return a connected, fully-supported small model.

    With ``coincident=True``, node 3 is placed within
    :data:`NODE_COINCIDENCE_TOL` of node 2 — but the elements still
    point at distinct ids so the model passes the isolated-node and
    disconnected-component fatal checks. This is the scenario the new
    warning is meant to catch.
    """
    m = StructuralModel(title="audit-test")
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 5.0, 0.0)
    if coincident:
        m.nodes[3] = Node(3, 5.0 + 1e-11, 0.0)  # within tol of node 2
    else:
        m.nodes[3] = Node(3, 5.0, 4.0)
    m.materials[1] = Material(id=1, name="S", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=0.01, I=1e-4, rho=7850.0, depth=0.3, section_id=1,
    ))
    m.elements.append(FrameElement2D(
        id=2, node_i=1, node_j=3,
        E=2.1e8, A=0.01, I=1e-4, rho=7850.0, depth=0.3, section_id=1,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[2] = Support(node_id=2, uy=True)
    m.supports[3] = Support(node_id=3, uy=True)
    return m


def test_find_coincident_node_pairs_returns_pair_within_tolerance():
    m = _supported_three_node_frame(coincident=True)
    pairs = _find_coincident_node_pairs(m)
    assert len(pairs) == 1
    a, b, dist = pairs[0]
    assert (a, b) == (2, 3)  # sorted by id
    assert dist < NODE_COINCIDENCE_TOL


def test_find_coincident_node_pairs_empty_for_distinct_nodes():
    m = _supported_three_node_frame(coincident=False)
    assert _find_coincident_node_pairs(m) == []


def test_validate_model_warns_on_coincident_nodes():
    """The fatal-vs-warning split: coincident nodes are a warning,
    not a raise. The connected-but-coincident model passes
    validation and the warning is in the returned list."""
    m = _supported_three_node_frame(coincident=True)
    dofs = DofManager.from_model(m)
    warnings = validate_model(m, dofs)
    coincident_warnings = [w for w in warnings if "Coincident nodes" in w]
    assert len(coincident_warnings) == 1
    msg = coincident_warnings[0]
    assert "(2, 3)" in msg


def test_validate_model_no_coincident_warning_when_distinct():
    m = _supported_three_node_frame(coincident=False)
    dofs = DofManager.from_model(m)
    warnings = validate_model(m, dofs)
    assert not any("Coincident nodes" in w for w in warnings)


def test_node_coincidence_tol_is_shared_with_commands_module():
    """The constant lives in model.py and is re-exported from
    commands.py for back-compat. Both must reference the same float
    so the add-time block and the audit warning agree."""
    from structural_analysis.gui_common.commands import (
        NODE_COINCIDENCE_TOL as CMD_TOL,
    )
    assert NODE_COINCIDENCE_TOL == CMD_TOL


def test_pairs_at_bounding_box_corner_are_not_coincident():
    """PR #21 review fix: two nodes with ``|dx| < tol AND |dy| < tol``
    can still have Euclidean distance up to ``sqrt(2)·tol``. The
    helper must enforce the strict Euclidean check so the warning
    message's stated threshold is honest.

    Place two nodes such that ``|dx| = |dy| = 0.8·tol``; ``√(2)·0.8 =
    1.13·tol > tol``, so they must NOT be flagged.
    """
    m = StructuralModel(title="bbox-corner")
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 0.8 * NODE_COINCIDENCE_TOL, 0.8 * NODE_COINCIDENCE_TOL)
    pairs = _find_coincident_node_pairs(m)
    assert pairs == [], (
        "expected no coincident pair: |dx|, |dy| < tol but Euclidean "
        f"distance is √2·0.8·tol > tol; got {pairs}"
    )


def test_pairs_just_inside_euclidean_are_coincident():
    """Symmetric: place two nodes at half-tolerance on each axis so
    Euclidean distance √2·0.5·tol ≈ 0.71·tol stays under the
    threshold. Must be flagged."""
    m = StructuralModel(title="bbox-inside")
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 0.5 * NODE_COINCIDENCE_TOL, 0.5 * NODE_COINCIDENCE_TOL)
    pairs = _find_coincident_node_pairs(m)
    assert len(pairs) == 1
    assert pairs[0][2] < NODE_COINCIDENCE_TOL


def test_coincident_warning_caps_pair_list_for_large_groups():
    """Pathological imports may have many coincident pairs; the
    warning lists at most _MAX_COINCIDENT_PAIRS_IN_WARNING (10) and
    appends '…(+N more)'. Build a model with 13 coincident pairs and
    verify the truncation."""
    m = StructuralModel(title="many-dups")
    # 12 nodes around (5, 0), each within tol of the others. That's
    # C(12, 2) = 66 pairs. With the cap at 10, expect the suffix.
    for i in range(1, 13):
        m.nodes[i] = Node(i, 5.0 + i * 1e-12, 0.0)
    pairs = _find_coincident_node_pairs(m)
    assert len(pairs) > 10


def test_validate_model_warning_truncates_pair_list_with_suffix():
    """Cover the truncation *formatting* in validate_model itself (PR
    #21 review): with more than _MAX_COINCIDENT_PAIRS_IN_WARNING pairs,
    the emitted warning must list exactly the cap and end with the
    '…(+N more)' suffix. The previous test only exercised the helper,
    leaving the suffix assembly in validate_model unverified.

    Model shape: one fixed anchor at the origin plus a fan of 6
    near-coincident nodes around (5, 0), each tied to the anchor by a
    frame element. Every fan node has incidence >= 1 (not isolated)
    and the whole thing is one supported connected component, so
    validate_model returns (warnings) rather than raising. The fan of
    6 yields C(6, 2) = 15 coincident pairs > cap of 10."""
    from structural_analysis.assembler import _MAX_COINCIDENT_PAIRS_IN_WARNING

    m = StructuralModel(title="fan-of-dups")
    m.materials[1] = Material(id=1, name="S", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    m.nodes[0] = Node(0, 0.0, 0.0)  # anchor
    n_fan = 6
    for i in range(1, n_fan + 1):
        m.nodes[i] = Node(i, 5.0 + i * 1e-12, 0.0)  # all within tol
        m.elements.append(FrameElement2D(
            id=i, node_i=0, node_j=i,
            E=2.1e8, A=0.01, I=1e-4, rho=7850.0, depth=0.3, section_id=1,
        ))
    m.supports[0] = Support(node_id=0, ux=True, uy=True, rz=True)

    dofs = DofManager.from_model(m)
    warnings = validate_model(m, dofs)
    coincident = [w for w in warnings if "Coincident nodes" in w]
    assert len(coincident) == 1
    msg = coincident[0]
    total_pairs = _find_coincident_node_pairs(m)
    extra = len(total_pairs) - _MAX_COINCIDENT_PAIRS_IN_WARNING
    assert extra > 0
    assert f"…(+{extra} more)" in msg
    # Exactly the cap number of "(a, b)" pairs is listed (the regex
    # ignores the "(Δ ≤ …)" prefix and the "(+N more)" suffix).
    import re
    shown_pairs = re.findall(r"\(\d+, \d+\)", msg)
    assert len(shown_pairs) == _MAX_COINCIDENT_PAIRS_IN_WARNING
