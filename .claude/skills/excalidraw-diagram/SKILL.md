---
name: excalidraw-diagram
description: Create Excalidraw diagram JSON files that make visual arguments. Use when the user wants to visualize workflows, architectures, or concepts.
---

# Excalidraw Diagram Creator

Generate `.excalidraw` JSON files that **argue visually**, not just display information.

**Setup:** If the user asks you to set up this skill (renderer, dependencies, etc.), see `README.md` for instructions.

## Customization

**All colors and brand-specific styles live in one file:** `references/color-palette.md`. Read it before generating any diagram and use it as the single source of truth for all color choices — shape fills, strokes, text colors, evidence artifact backgrounds, everything.

To make this skill produce diagrams in your own brand style, edit `color-palette.md`. Everything else in this file is universal design methodology and Excalidraw best practices.

---

## Core Philosophy

**Diagrams should ARGUE, not DISPLAY.**

A diagram isn't formatted text. It's a visual argument that shows relationships, causality, and flow that words alone can't express. The shape should BE the meaning.

**The Isomorphism Test**: If you removed all text, would the structure alone communicate the concept? If not, redesign.

**The Education Test**: Could someone learn something concrete from this diagram, or does it just label boxes? A good diagram teaches—it shows actual formats, real event names, concrete examples.

---

## Depth Assessment (Do This First)

Before designing, determine what level of detail this diagram needs:

### Simple/Conceptual Diagrams
Use abstract shapes when:
- Explaining a mental model or philosophy
- The audience doesn't need technical specifics
- The concept IS the abstraction (e.g., "separation of concerns")

### Comprehensive/Technical Diagrams
Use concrete examples when:
- Diagramming a real system, protocol, or architecture
- The diagram will be used to teach or explain (e.g., YouTube video)
- The audience needs to understand what things actually look like
- You're showing how multiple technologies integrate

**For technical diagrams, you MUST include evidence artifacts** (see below).

---

## Research Mandate (For Technical Diagrams)

**Before drawing anything technical, research the actual specifications.**

If you're diagramming a protocol, API, or framework:
1. Look up the actual JSON/data formats
2. Find the real event names, method names, or API endpoints
3. Understand how the pieces actually connect
4. Use real terminology, not generic placeholders

Bad: "Protocol" → "Frontend"
Good: "AG-UI streams events (RUN_STARTED, STATE_DELTA, A2UI_UPDATE)" → "CopilotKit renders via createA2UIMessageRenderer()"

**Research makes diagrams accurate AND educational.**

---

## Evidence Artifacts

Evidence artifacts are concrete examples that prove your diagram is accurate and help viewers learn. Include them in technical diagrams.

| Artifact Type | When to Use | How to Render |
|---------------|-------------|---------------|
| **Code snippets** | APIs, integrations, implementation details | Dark rectangle + syntax-colored text (see color palette for evidence artifact colors) |
| **Data/JSON examples** | Data formats, schemas, payloads | Dark rectangle + colored text (see color palette) |
| **Event/step sequences** | Protocols, workflows, lifecycles | Timeline pattern (line + dots + labels) |
| **UI mockups** | Showing actual output/results | Nested rectangles mimicking real UI |
| **Real input content** | Showing what goes IN to a system | Rectangle with sample content visible |
| **API/method names** | Real function calls, endpoints | Use actual names from docs, not placeholders |

The key principle: **show what things actually look like**, not just what they're called.

---

## Multi-Zoom Architecture

Comprehensive diagrams operate at multiple zoom levels simultaneously.

### Level 1: Summary Flow
A simplified overview showing the full pipeline or process at a glance.

### Level 2: Section Boundaries
Labeled regions that group related components.

### Level 3: Detail Inside Sections
Evidence artifacts, code snippets, and concrete examples within each section.

**For comprehensive diagrams, aim to include all three levels.** The summary gives context, the sections organize, and the details teach.

### Simple vs Comprehensive (Know Which You Need)

| Simple Diagram | Comprehensive Diagram |
|----------------|----------------------|
| Generic labels: "Input" → "Process" → "Output" | Specific: shows what the input/output actually looks like |
| Named boxes: "API", "Database", "Client" | Named boxes + examples of actual requests/responses |
| "Events" or "Messages" label | Timeline with real event/message names from the spec |
| "UI" or "Dashboard" rectangle | Mockup showing actual UI elements and content |
| ~30 seconds to explain | ~2-3 minutes of teaching content |
| Viewer learns the structure | Viewer learns the structure AND the details |

---

## Container vs. Free-Floating Text

**Not every piece of text needs a shape around it.** Default to free-floating text. Add containers only when they serve a purpose.

| Use a Container When... | Use Free-Floating Text When... |
|------------------------|-------------------------------|
| It's the focal point of a section | It's a label or description |
| It needs visual grouping with other elements | It's supporting detail or metadata |
| Arrows need to connect to it | It describes something nearby |
| The shape itself carries meaning (decision diamond, etc.) | Typography alone creates sufficient hierarchy |
| It represents a distinct "thing" in the system | It's a section title, subtitle, or annotation |

**Typography as hierarchy**: Use font size, weight, and color to create visual hierarchy without boxes.

---

## Design Process (Do This BEFORE Generating JSON)

### Step 0: Assess Depth Required
Determine if this needs to be **Simple/Conceptual** or **Comprehensive/Technical**. If comprehensive, research first.

### Step 1: Understand Deeply
For each concept, ask:
- What does this concept **DO**?
- What relationships exist between concepts?
- What's the core transformation or flow?
- **What would someone need to SEE to understand this?**

### Step 2: Map Concepts to Patterns

| If the concept... | Use this pattern |
|-------------------|------------------|
| Spawns multiple outputs | **Fan-out** |
| Combines inputs into one | **Convergence** |
| Has hierarchy/nesting | **Tree** |
| Is a sequence of steps | **Timeline** |
| Loops or improves continuously | **Spiral/Cycle** |
| Is an abstract state or context | **Cloud** |
| Transforms input to output | **Assembly line** |
| Compares two things | **Side-by-side** |
| Separates into phases | **Gap/Break** |

### Step 3: Ensure Variety
For multi-concept diagrams: **each major concept must use a different visual pattern**. No uniform cards or grids.

### Step 4: Sketch the Flow
Before JSON, mentally trace how the eye moves through the diagram.

### Step 5: Generate JSON
Only now create the Excalidraw elements.

### Step 6: Render & Validate (MANDATORY)
After generating the JSON, run the render-view-fix loop until the diagram looks right.

---

## Large / Comprehensive Diagram Strategy

**For comprehensive or technical diagrams, build the JSON one section at a time.**

### The Section-by-Section Workflow
1. **Create the base file** with the JSON wrapper (`type`, `version`, `appState`, `files`) and the first section of elements.
2. **Add one section per edit.** Each section gets its own dedicated pass.
3. **Use descriptive string IDs** (e.g., `"trigger_rect"`, `"arrow_fan_left"`) so cross-section references are readable.
4. **Namespace seeds by section** (e.g., section 1 uses 100xxx, section 2 uses 200xxx) to avoid collisions.
5. **Update cross-section bindings** as you go.

### What NOT to Do
- Don't generate the entire diagram in one response for large comprehensive diagrams.
- Don't use a coding agent to generate the JSON.
- Don't write a Python generator script; hand-crafted JSON with descriptive IDs is more maintainable.

---

## Visual Pattern Library

### Fan-Out (One-to-Many)
Central element with arrows radiating to multiple targets. Use for sources, PRDs, root causes, central hubs.

### Convergence (Many-to-One)
Multiple inputs merging through arrows to single output. Use for aggregation, funnels, synthesis.

### Tree (Hierarchy)
Parent-child branching with connecting lines and free-floating text.

### Spiral/Cycle (Continuous Loop)
Elements in sequence with arrow returning to start.

### Cloud (Abstract State)
Overlapping ellipses with varied sizes.

### Assembly Line (Transformation)
Input → Process Box → Output with clear before/after.

### Side-by-Side (Comparison)
Two parallel structures with visual contrast.

### Gap/Break (Separation)
Visual whitespace or barrier between sections.

### Lines as Structure
Use lines (`type: "line"`, not arrows) as primary structural elements instead of boxes: timelines, tree structures, dividers, and flow spines.

---

## Shape Meaning

Choose shape based on what it represents—or use no shape at all:

| Concept Type | Shape | Why |
|--------------|-------|-----|
| Labels, descriptions, details | **none** | Typography creates hierarchy |
| Section titles, annotations | **none** | Font size/weight is enough |
| Markers on a timeline | small `ellipse` | Visual anchor, not container |
| Start, trigger, input | `ellipse` | Soft, origin-like |
| End, output, result | `ellipse` | Completion, destination |
| Decision, condition | `diamond` | Classic decision symbol |
| Process, action, step | `rectangle` | Contained action |
| Abstract state, context | overlapping `ellipse` | Fuzzy, cloud-like |
| Hierarchy node | lines + text | Structure through lines |

**Rule**: Default to no container. Aim for <30% of text elements to be inside containers.

---

## Color as Meaning

Colors encode information, not decoration. Every color choice should come from `references/color-palette.md`.

**Do not invent new colors.** If a concept doesn't fit an existing semantic category, use Primary/Neutral or Secondary.

---

## Modern Aesthetics

- **Roughness**: default to `roughness: 0` for professional diagrams.
- **Stroke Width**: use 1 for subtle structure, 2 for standard shapes/arrows, 3 sparingly for emphasis.
- **Opacity**: always use `opacity: 100` for all elements.

---

## Layout Principles

- **Hierarchy Through Scale**: Hero 300×150, Primary 180×90, Secondary 120×60, Small 60×40.
- **Whitespace = Importance**: Important elements have more empty space around them.
- **Flow Direction**: typically left→right or top→bottom for sequences, radial for hub-and-spoke.
- **Connections Required**: If A relates to B, there must be an arrow.

---

## Text Rules

**CRITICAL**: The JSON `text` property contains ONLY readable words.

Settings: `fontSize: 16`, `fontFamily: 3`, `textAlign: "center"`, `verticalAlign: "middle"`.

---

## JSON Structure

```json
{
  "type": "excalidraw",
  "version": 2,
  "source": "https://excalidraw.com",
  "elements": [],
  "appState": { "viewBackgroundColor": "#ffffff", "gridSize": 20 },
  "files": {}
}
```

## Element Templates

See `references/element-templates.md` for copy-paste JSON templates for each element type.

---

## Render & Validate (MANDATORY)

You cannot judge a diagram from JSON alone. After generating or editing the Excalidraw JSON, render it to PNG, view the image, and fix what you see.

### How to Render

```bash
cd .claude/skills/excalidraw-diagram/references && uv run python render_excalidraw.py <path-to-file.excalidraw>
```

This outputs a PNG next to the `.excalidraw` file.

### The Loop
1. **Render & View** — Run the render script, then read the PNG.
2. **Audit against your original vision** — Confirm structure, flow, hierarchy, and evidence artifacts.
3. **Check for visual defects** — Clipping, overlaps, arrow routing issues, ambiguous labels, uneven spacing, unreadable text, imbalance.
4. **Fix** — Edit coordinates, sizes, routes, and labels.
5. **Re-render & re-view**.
6. **Repeat** until the diagram is ready to show without caveats.

### First-Time Setup

```bash
cd .claude/skills/excalidraw-diagram/references
uv sync
uv run playwright install chromium
```

---

## Quality Checklist

### Depth & Evidence
1. Research done
2. Evidence artifacts included
3. Multi-zoom structure present
4. Concrete over abstract
5. Educational value

### Conceptual
6. Isomorphism
7. Argument
8. Variety
9. No uniform containers

### Container Discipline
10. Minimal containers
11. Lines as structure
12. Typography hierarchy

### Structural
13. Connections
14. Flow
15. Hierarchy

### Technical
16. Text clean
17. Font family 3
18. Roughness 0 for modern diagrams
19. Opacity 100
20. Container ratio <30%

### Visual Validation
21. Rendered to PNG
22. No text overflow
23. No overlapping elements
24. Even spacing
25. Arrows land correctly
26. Readable at export size
27. Balanced composition
