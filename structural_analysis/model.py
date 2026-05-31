"""
Data model classes for 2D structural analysis.

Supports frame elements, truss elements, moment releases,
and member loads (point loads and UDL).

Design decisions
----------------
- Node, Support, NodalLoad are frozen (immutable) dataclasses — once created
  they should not be mutated.
- Element classes live in element.py and use inheritance (Element2D base).
- MemberLoad is a union type: UniformDistributedLoad | PointLoad.
- AnalysisResult is a structured container for all outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


# ── Physical constants ─────────────────────────────────────────

STANDARD_GRAVITY = 9.81  # m/s²
"""Single source of truth for gravitational acceleration.

Used by the assembler's self-weight pass, the mass / self-weight summary
window, and the tests that verify them. In v0.9.0 this constant is the
only supported value; future versions may let the user pick from a
fixed list (9.81 / 9.80665 / 10.0 / custom) on an Advanced analysis
settings panel.
"""


NODE_COINCIDENCE_TOL: float = 1e-9
"""World-unit tolerance for "are these two coordinates the same node?".

Single source of truth shared across the model, GUI commands
(``gui_common/commands.py``), and the assembler's pre-solve audit
(``assembler.validate_model``). Lives in this module so the analytic
core doesn't have to import the GUI layer.

The snap engine in ``gui_qt/snap.py`` uses a separate *pixel*-space
radius (10 px) for visual targeting — that's a different concern
(cursor-snap UX) and intentionally not linked to this constant.
"""


# ── Nodes ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Node:
    """A node in the structural model."""

    id: int
    x: float
    y: float


# ── Material / Section ─────────────────────────────────────────


@dataclass(frozen=True)
class Material:
    """Pure material properties (E, α, ρ).

    Sections (A, I, depth) are stored separately on the model as
    :class:`Section` objects, with each section referencing a material.

    Attributes:
        id: Material identifier.
        name: Optional human-readable name (e.g. "C40/50", "S355").
        E: Modulus of elasticity (kN/m²).
        alpha: Coefficient of thermal expansion (1/°C). Default 0 (inert).
        density: Mass density (kg/m³). Default 0 (modal analysis disabled
            for elements whose material carries density = 0). The unit
            conversion to the kN-m-s consistent system used by the static
            solver is done inside :mod:`structural_analysis.mass`.
        nu: Poisson's ratio. Default 0.0. Used to derive G via the standard
            isotropic identity ``G = E / (2 * (1 + nu))``. Not consumed by
            the 2D solver yet — stored for future 3D / torsion features.
        template: Optional name of the preset (e.g. ``"Steel_S275"``) that
            populated this material's defaults. Free-form string.
    """

    id: int
    name: str = ""
    E: float = 0.0
    alpha: float = 0.0
    density: float = 0.0
    nu: float = 0.0
    template: str = ""

    @property
    def G(self) -> float:
        """Shear modulus, derived from E and ν.

        Computed always — when ν = 0 this returns ``E / 2`` (the correct
        isotropic identity at ν = 0, not a special case). Raises
        :class:`ValueError` if ν is outside ``[0, 0.5)`` so a stray
        invalid Material (e.g. constructed in a test) surfaces the
        problem clearly instead of yielding ``ZeroDivisionError`` or a
        physically meaningless negative G.
        """
        if not (0.0 <= self.nu < 0.5):
            raise ValueError(
                f"Material {self.id}: nu={self.nu!r} is outside the "
                "allowed range [0, 0.5); G is undefined."
            )
        return self.E / (2.0 * (1.0 + self.nu))


@dataclass(frozen=True)
class Section:
    """Cross-section properties, pointing back to a material.

    The original 2D solver only needs ``A``, ``I``, and ``depth`` (for
    thermal-gradient curvature). The remaining fields (``width``, ``J``,
    and the shape wizard data) are stored so that a section can record
    *how* its A and I were derived (e.g. "this is a 300 × 500 mm
    rectangle"). They are not consumed by the current solver.

    Attributes:
        id: Section identifier.
        name: Optional human-readable name (e.g. "W360x196", "50x50").
        material_id: id of the :class:`Material` this section uses.
        A: Cross-sectional area (m²).
        I: Moment of inertia (m⁴).
        depth: Section depth (m), used for frame thermal-gradient curvature.
            Default 0 (required only if a thermal gradient is applied).
        width: Section width (m). Storage-only for now.
        J: Torsion constant (m⁴). Storage-only — the 2D solver does not
            consume it. Rectangle / square shapes leave this at 0 by
            design; I-section provides an approximate thin-walled value
            for future 3D / reporting use.
        shape_type: One of ``"manual"``, ``"rectangle"``, ``"square"``,
            ``"i_section"``. Records which dialog page produced the
            section. ``"manual"`` means the user entered A/I/depth/width
            directly without going through a shape calculator.
        b, h, tf, tw: Raw dimensions used by the shape calculator (m).
            Their meaning depends on ``shape_type``:
              - rectangle/square: b = width, h = depth
              - i_section: b = flange width, h = overall depth,
                tf = flange thickness, tw = web thickness
            All default to 0.0 for ``shape_type="manual"``.
    """

    id: int
    name: str = ""
    material_id: int = 0
    A: float = 0.0
    I: float = 0.0
    depth: float = 0.0
    width: float = 0.0
    J: float = 0.0
    shape_type: str = "manual"
    b: float = 0.0
    h: float = 0.0
    tf: float = 0.0
    tw: float = 0.0


# ── Supports ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Support:
    """Boundary condition at a node with optional support settlement.

    Booleans indicate whether a DOF is restrained (True) or free (False).
    Float fields indicate prescribed (non-zero) displacements at restrained DOFs.
    A non-None settlement value overrides the zero default at that DOF.

    Attributes:
        node_id: Node identifier.
        ux, uy, rz: True if the DOF is restrained.
        settle_ux, settle_uy, settle_rz: Prescribed displacement at a restrained
            DOF. None means the standard zero displacement. Non-zero values are
            used to model support settlement.
    """

    node_id: int
    ux: bool = False
    uy: bool = False
    rz: bool = False
    settle_ux: float | None = None
    settle_uy: float | None = None
    settle_rz: float | None = None

    def prescribed(self, dof: str) -> float:
        """Return the prescribed displacement for a DOF ('ux', 'uy', 'rz').

        Args:
            dof: DOF name ('ux', 'uy', or 'rz').

        Returns:
            Prescribed displacement value, or 0.0 if no settlement defined.
        """
        return getattr(self, f"settle_{dof}") or 0.0


# ── Loads ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class NodalLoad:
    """Applied load at a node (global coordinates)."""

    node_id: int
    fx: float = 0.0   # force in x (kN)
    fy: float = 0.0   # force in y (kN)
    mz: float = 0.0   # moment about z (kN·m)
    load_case: str = "DEFAULT"


@dataclass(frozen=True)
class UniformDistributedLoad:
    """Full-length distributed line load on a member.

    Components ``wx`` (axial) and ``wy`` (transverse) are interpreted
    per the ``coord_system`` token:

    * ``"local"`` (default): ``wx`` is in the element's local
      +x_local (axial, tip-to-tip), ``wy`` is in the element's local
      +y_local (transverse). The classic ``UniformDistributedLoad(wy=...)``
      construction retains its original meaning.
    * ``"global"``: ``wx`` is interpreted as ``qX`` (global +X) and
      ``wy`` as ``qY`` (global +Y) — both **force per unit member
      length** (NOT per horizontal projection). The solver projects to
      local axes so inclined members pick up both axial and transverse
      FEMs. The field NAMES stay ``wx`` / ``wy`` for backward-compat
      storage even though the semantic name is qX / qY in this mode.
    * ``"gravity"``: a direction token stored in ``coord_system`` so the
      same field handles all three cases. Magnitude lives in ``wy``;
      ``wx`` MUST be 0 (validated in __post_init__). Positive
      magnitude acts in the global gravity direction — for this 2-D
      program, ``global -Y``. Internally projected exactly as a
      global load with components ``(0, -wy)``.
    """

    wy: float
    wx: float = 0.0
    coord_system: str = "local"
    load_case: str = "DEFAULT"

    def __post_init__(self):
        if self.coord_system not in ("local", "global", "gravity"):
            raise ValueError(
                f"UniformDistributedLoad.coord_system must be 'local', "
                f"'global', or 'gravity' (got {self.coord_system!r})."
            )
        if self.coord_system == "gravity" and self.wx != 0.0:
            raise ValueError(
                "UniformDistributedLoad with coord_system='gravity' has a "
                "single magnitude in wy; wx must be 0 because gravity is "
                "a 1-D direction (global -Y) by definition."
            )


@dataclass(frozen=True)
class PointLoad:
    """Point load on a member at distance *a* from start node.

    Components ``px`` (axial) and ``py`` (transverse) are interpreted
    per the ``coord_system`` token:

    * ``"local"`` (default): ``px`` along +x_local, ``py`` along
      +y_local. The classic ``PointLoad(py=..., a=...)`` construction
      retains its original meaning.
    * ``"global"``: ``px`` is interpreted as the global +X force
      component and ``py`` as the global +Y component. The solver
      projects to local axes so inclined members pick up both axial
      and transverse fixed-end forces.
    * ``"gravity"``: magnitude in ``py``; ``px`` MUST be 0
      (validated). Positive magnitude acts in global -Y (gravity).
    """

    py: float
    a: float
    px: float = 0.0
    coord_system: str = "local"
    load_case: str = "DEFAULT"

    def __post_init__(self):
        if self.coord_system not in ("local", "global", "gravity"):
            raise ValueError(
                f"PointLoad.coord_system must be 'local', 'global', or "
                f"'gravity' (got {self.coord_system!r})."
            )
        if self.coord_system == "gravity" and self.px != 0.0:
            raise ValueError(
                "PointLoad with coord_system='gravity' has a single "
                "magnitude in py; px must be 0 because gravity is a "
                "1-D direction (global -Y) by definition."
            )


@dataclass(frozen=True)
class TrussTemperatureLoad:
    """Uniform thermal change on a truss element.

    Produces axial fixed-end forces N_T = E·A·α·ΔT on a restrained bar.
    The α value is read from the element's Material (not stored here).

    Attributes:
        delta_T: Uniform temperature change (°C). Positive = heating.
    """

    delta_T: float
    load_case: str = "DEFAULT"


@dataclass(frozen=True)
class FrameTemperatureLoad:
    """Thermal loading on a frame element, parameterized by top/bottom temps.

    The mean temperature produces an axial effect (N_T = E·A·α·ΔT_mean).
    The difference (t_bottom − t_top) produces a curvature effect
    (M_T = E·I·α·(t_bottom − t_top)/depth) — both derived internally.

    This parameterization is physically more natural than the combined
    uniform+gradient form: users specify what the member's top and bottom
    fibers feel directly, and the axial/bending decomposition happens
    inside the element.

    The α and depth values are read from the element's Material.

    Attributes:
        t_top: Temperature change at top fiber (°C).
        t_bottom: Temperature change at bottom fiber (°C).
    """

    t_top: float = 0.0
    t_bottom: float = 0.0
    load_case: str = "DEFAULT"


MemberLoad = (
    UniformDistributedLoad | PointLoad
    | TrussTemperatureLoad | FrameTemperatureLoad
)


# ── Load Cases (v0.18) ────────────────────────────────────────


@dataclass
class LoadCase:
    """A named bucket that user-created loads can be tagged with.

    PR-A foundation: each load (nodal / member-mechanical / member-thermal)
    carries a ``load_case`` string (added in PR #27); ``LoadCase`` is the
    object the model holds for each unique name. Per-case solving filters
    the load list by this tag at ``main.run_analysis`` time — the solver
    itself stays case-agnostic.

    Attributes:
        name: Unique key in ``StructuralModel.load_cases`` (also the
            ``load_case`` string on attached loads). Must not contain
            whitespace or ``#`` — matches the load-case combo
            normalisation in ``gui_qt.dialogs._normalize_load_case``.
        enabled: When True, ``run_multi_case_analysis`` solves this
            case. Disabled cases are skipped (and excluded from the
            SUM_ALL view).
        description: Free-text. Reserved for future PR-B; not written
            to the input-file format in v0.18.
    """

    name: str
    enabled: bool = True
    description: str = ""

    def __post_init__(self):
        if not self.name:
            raise ValueError("LoadCase.name must be non-empty.")
        if any(ch.isspace() or ch == "#" for ch in self.name):
            raise ValueError(
                f"LoadCase.name {self.name!r} contains invalid characters "
                "(whitespace or '#'); case names must be a single token."
            )


# ── Load Combinations (v0.19) ─────────────────────────────────


@dataclass
class LoadCombination:
    """A coefficient-weighted linear combination of solved load cases.

    PR #29: combinations are DERIVED results — never separately solved.
    The combined displacements / reactions / member forces are computed
    by scaling each referenced case's :class:`AnalysisResult` by its
    coefficient and summing (see
    ``multi_case_result.MultiCaseAnalysisResult.combination``). This is
    exact because the solver is linear elastic.

    Attributes:
        name: Unique key in ``StructuralModel.load_combinations``. Same
            single-token rule as :class:`LoadCase` (no whitespace,
            no ``#``). Must NOT collide with a load-case name or the
            ``SUM_ALL`` sentinel — enforced by the CRUD command, not
            here (the model layer doesn't know the case set).
        terms: ``{case_name: coefficient}``. At least one term is
            required. Coefficients must be finite; zero coefficients are
            rejected at construction (a zero term is noise — drop it
            instead). Negative coefficients ARE allowed (e.g. load
            reversal) and are handled intentionally.
        description: Optional free-text.
    """

    name: str
    terms: dict[str, float] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self):
        if not self.name:
            raise ValueError("LoadCombination.name must be non-empty.")
        if any(ch.isspace() or ch == "#" for ch in self.name):
            raise ValueError(
                f"LoadCombination.name {self.name!r} contains invalid "
                "characters (whitespace or '#'); combination names must "
                "be a single token."
            )
        if self.name == "SUM_ALL":
            raise ValueError(
                "SUM_ALL is a built-in derived view and cannot be used "
                "as a user-defined combination name."
            )
        if not self.terms:
            raise ValueError(
                f"LoadCombination {self.name!r} must have at least one "
                "term (case_name: coefficient)."
            )
        import math
        for case_name, coeff in self.terms.items():
            if not case_name:
                raise ValueError(
                    f"LoadCombination {self.name!r} has an empty case "
                    "name in its terms."
                )
            if not isinstance(coeff, (int, float)) or isinstance(coeff, bool):
                raise ValueError(
                    f"LoadCombination {self.name!r} term {case_name!r} "
                    f"coefficient must be a number (got {coeff!r})."
                )
            if not math.isfinite(coeff):
                raise ValueError(
                    f"LoadCombination {self.name!r} term {case_name!r} "
                    f"coefficient must be finite (got {coeff!r})."
                )
            if coeff == 0.0:
                raise ValueError(
                    f"LoadCombination {self.name!r} term {case_name!r} "
                    "has a zero coefficient; drop the term instead of "
                    "giving it a zero weight."
                )


# ── Structural Model ──────────────────────────────────────────


@dataclass
class StructuralModel:
    """Complete structural model container.

    Elements are stored as a list of Element2D subclass instances
    (FrameElement2D or TrussElement2D) — see element.py.
    """

    title: str = "Untitled"
    nodes: dict[int, Node] = field(default_factory=dict)
    materials: dict[int, Material] = field(default_factory=dict)
    sections: dict[int, "Section"] = field(default_factory=dict)
    elements: list = field(default_factory=list)        # list[Element2D]
    supports: dict[int, Support] = field(default_factory=dict)
    nodal_loads: list[NodalLoad] = field(default_factory=list)

    # ── analysis settings ──
    # When True, the static assembler injects gravity loads on every
    # element (frame: full local fixed-end forces; truss: half-weight
    # lumped at each endpoint in global -Y). Gravity direction is
    # hard-coded to global -Y at g = STANDARD_GRAVITY m/s² in v0.9.0.
    # The loads are applied during assembly only — never persisted into
    # ``nodal_loads`` or any element's ``member_loads``.
    include_self_weight: bool = False

    # ── load cases (v0.18 — PR-A) ──
    # Every user-created load (nodal_loads + each elem.member_loads
    # entry) carries a ``load_case: str`` tag (added in PR #27). The
    # ``load_cases`` dict here defines which case names exist on this
    # model and whether each is currently enabled for the multi-case
    # static solve.
    #
    # Invariant: ``"DEFAULT"`` always exists. The Load Case Manager
    # dialog blocks deletion of DEFAULT; the file reader auto-creates
    # any case name it encounters in a load row that isn't already in
    # the dict. ``StructuralModel.__post_init__`` injects DEFAULT into
    # freshly-constructed models so a from-scratch model is never in an
    # invalid "no cases" state.
    load_cases: dict[str, "LoadCase"] = field(default_factory=dict)

    # When the static run includes self-weight, that contribution is
    # bundled into exactly this case. SUM_ALL therefore includes
    # self-weight exactly once (avoiding double-count when multiple
    # cases are superposed). Must be the name of an entry in
    # ``load_cases``.
    self_weight_case: str = "DEFAULT"

    # ── load combinations (v0.19 — PR #29) ──
    # User-defined coefficient-weighted combinations of solved cases.
    # Keyed by combination name. Combinations are DERIVED results: they
    # are never separately solved — the combined response is computed
    # from the per-case results by scaling + summing (see
    # ``multi_case_result.MultiCaseAnalysisResult.combination``). The
    # built-in SUM_ALL view is NOT stored here; it stays a derived view
    # on the result wrapper and is never serialised.
    load_combinations: dict[str, "LoadCombination"] = field(
        default_factory=dict,
    )

    def __post_init__(self):
        # Guarantee the DEFAULT case exists. file_io / new-from-blank
        # paths both rely on this so they never have to insert it
        # themselves.
        if "DEFAULT" not in self.load_cases:
            self.load_cases["DEFAULT"] = LoadCase(name="DEFAULT")

    # ── convenience helpers ──

    def node(self, node_id: int) -> Node:
        """Return the Node with the given id.

        Args:
            node_id: The node identifier to look up.

        Returns:
            The Node object.
        """
        return self.nodes[node_id]

    def support_for(self, node_id: int) -> Support:
        """Return the Support for a node, or an all-free default.

        Args:
            node_id: The node identifier to look up.

        Returns:
            The Support object, or Support(node_id) with all DOFs free.
        """
        return self.supports.get(node_id, Support(node_id=node_id))

    @property
    def node_ids(self) -> list[int]:
        """Sorted list of node IDs in the model.

        Returns:
            List of integer node IDs in ascending order.
        """
        return sorted(self.nodes)


# ── Effective material resolver ───────────────────────────────


def effective_material(model: "StructuralModel", elem) -> Material:
    """Return the Material that drives ``elem``'s E / α / ρ / G.

    Per-element overrides take precedence over the section default:

        effective_material_id =
            elem.material_id_override or section.material_id

    Centralises effective-material resolution for callers that need the
    Material object behind an element's E / α / ρ / G — primarily the
    command-propagation paths in
    :class:`structural_analysis.gui_common.commands.AddOrUpdateMaterialCmd`
    and :class:`structural_analysis.gui_common.commands.AddOrUpdateSectionCmd`,
    which use it to decide which elements should refresh when a material
    or a section is edited.

    The GUI detail inspector resolves the same lookup inline because it
    must tolerate partially-broken models (e.g. mid-edit, with a dangling
    section_id or material id), whereas this helper raises ``KeyError``
    in that case.

    Raises:
        KeyError: if the resolved material id or the element's section id
            is missing from the model.
    """
    section = model.sections[elem.section_id]
    mid = getattr(elem, "material_id_override", None) or section.material_id
    return model.materials[mid]


# ── Analysis Result ───────────────────────────────────────────


@dataclass
class AnalysisResult:
    """Structured container for all analysis outputs."""

    status: str                                          # "ok" or "error"
    title: str = ""
    warnings: list[str] = field(default_factory=list)

    # Step B
    E_map: dict[int, dict[str, int | None]] = field(default_factory=dict)
    num_eq: int = 0
    G_vectors: dict[int, list[int | None]] = field(default_factory=dict)

    # Step C
    K: object = None   # np.ndarray — kept as object to avoid import
    F: object = None

    # Step D
    D: object = None
    residual: float = 0.0

    # Step E
    member_results: dict[int, dict] = field(default_factory=dict)

    # Step F
    reactions: dict[int, dict[str, float]] = field(default_factory=dict)
    eq_residual: float = 0.0

    # Storage
    elem_data: dict[int, dict] = field(default_factory=dict)

    # Diagnostics
    diagnostics: dict[str, object] = field(default_factory=dict)
