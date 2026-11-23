---
name: StudPerf Probability Atlas
description: A precise, colorful visual system for trustworthy student probability analysis.
colors:
  prediction-cobalt: "#2056C7"
  evidence-teal: "#087F75"
  uncertainty-amber: "#A85E00"
  risk-red: "#B42318"
  graphite: "#15181D"
  muted-slate: "#5A6572"
  cool-rule: "#DDE2E8"
  paper: "#FFFFFF"
  ground: "#F4F6F8"
typography:
  display: "Manrope, sans-serif"
  body: "Source Sans 3, sans-serif"
rounded:
  control: "8px"
  panel: "10px"
  focal: "12px"
spacing:
  xs: "0.55rem"
  sm: "0.8rem"
  md: "1.15rem"
  lg: "1.55rem"
  section: "2.7rem"
components:
  button-primary:
    background: "{colors.prediction-cobalt}"
    color: "{colors.paper}"
    radius: "{rounded.control}"
  probability-panel:
    background: "{colors.paper}"
    color: "{colors.graphite}"
    borderColor: "{colors.cool-rule}"
    radius: "{rounded.focal}"
---

# Design System: StudPerf Probability Atlas

## Overview

**Creative North Star: "The Probability Atlas"**

StudPerf treats model output as a map to be read, not a score to be celebrated. The system is light, precise, and research-minded: graphite typography sits on a cool paper field, open sections replace dense dashboard card grids, and color always explains meaning.

The interface is intentionally restrained. Cobalt marks prediction and active control, teal marks evidence, amber marks uncertainty, and red appears only for material risk. The visual anti-reference is the generic dark AI dashboard with glowing tiles, equal-weight cards, and decorative charts.

**Key Characteristics:**

- Open, editorial sections separated by fine rules.
- One dominant probability reading per prediction view.
- Semantic color instead of decorative color.
- Human-readable research context stays adjacent to every result.
- Responsive stacking without a reduced mobile experience.

## Colors

The palette combines cool institutional neutrals with four disciplined semantic accents.

### Primary

- **Prediction Cobalt:** Drives selected navigation, probability meters, primary actions, and interactive sensitivity curves.

### Secondary

- **Evidence Teal:** Marks validated evidence, calibrated model information, and positive research context.
- **Uncertainty Amber:** Marks thresholds, limitations, and conclusions that need careful interpretation.

### Tertiary

- **Risk Red:** Reserved for warnings where a research or decision boundary is genuinely material.

### Neutral

- **Graphite:** Primary text and numerical output.
- **Muted Slate:** Supporting copy and metadata.
- **Cool Rule:** Borders, dividers, and chart axes.
- **Paper:** Reading surfaces and focal panels.
- **Ground:** The application field behind the content.

**The Semantic Color Rule.** Every accent must communicate prediction, evidence, uncertainty, or risk; never use accent color merely to fill space.

## Typography

**Display Font:** Manrope (with sans-serif fallback)  
**Body Font:** Source Sans 3 (with sans-serif fallback)

**Character:** Manrope gives probability values and page titles a compact analytical confidence. Source Sans 3 keeps dense methodological copy legible and neutral.

### Hierarchy

- **Display** (600, 3.25–5rem, 1): The single dominant probability value.
- **Headline** (650, 2–2.8rem, 1.08): Page identity and primary question.
- **Title** (620, 1.45–2rem, 1.18): Section wayfinding.
- **Body** (400–500, 1rem, 1.6): Explanations, limitations, and findings, generally held below 72 characters per line.
- **Label** (600, 0.75–0.86rem): Inputs, evidence ribbon labels, and metadata.

**The One Number Rule.** Only the current pass probability may use display scale; supporting metrics remain subordinate.

## Layout

The application uses a fixed 254px navigation rail and a content canvas capped at 1240px. Major tasks use a two-column evidence layout on desktop and stack naturally below 900px. Section spacing follows an open 2.7rem rhythm; borders, not shadows, organize related content. At 640px, margins tighten to 1rem, contextual facts wrap, and probability ribbons become two-column grids.

## Elevation & Depth

The system is flat by default. Depth comes from tonal layering—cool ground, white paper, and fine cool-gray borders—not from ambient shadow.

**The Flat Evidence Rule.** Analytical surfaces never float above the page; borders and spacing must be sufficient to explain hierarchy.

## Shapes

Controls use gently curved 8px corners, supporting panels use 10px corners, and the focal probability panel uses 12px. Pills and ornamental organic shapes are absent. All borders are quiet, one-pixel rules.

## Components

### Buttons

- **Shape:** Compact and gently curved (8px), with a 44px minimum target.
- **Primary:** Prediction cobalt with white text.
- **Hover / Focus:** One-pixel lift on hover; a visible cobalt focus ring that does not rely on color alone.
- **Secondary:** White paper with a cool rule border and cobalt hover state.

### Cards / Containers

- **Corner Style:** Supporting panels use 10px; the probability result uses 12px.
- **Background:** White paper over cool ground.
- **Shadow Strategy:** None.
- **Border:** One-pixel cool rule.
- **Internal Padding:** 1–1.55rem according to information density.

### Inputs / Fields

- **Style:** White field, cool border, 8px radius, explicit label.
- **Focus:** Two-pixel cobalt outline with offset.
- **Error / Disabled:** Preserve readable text contrast and state wording; do not rely on red alone.

### Navigation

The desktop rail uses plain text and native radio affordances. The selected row receives a pale cobalt field and darker cobalt text. On mobile the rail collapses into Streamlit's accessible drawer, leaving the analysis full width.

### Probability Reading

The signature component combines one large pass probability, a cobalt meter, classification text, and a four-cell evidence ribbon. Fail probability, decision threshold, model, and target stay visible in the same reading unit.

## Do's and Don'ts

### Do:

- **Do** keep the model estimate, threshold, target definition, and limitations visually connected.
- **Do** use teal for evidence and amber for uncertainty consistently across pages.
- **Do** let charts inherit the same neutral field, typography, and semantic accent logic.
- **Do** preserve full workflows on mobile by stacking, wrapping, and allowing intentional table scroll.

### Don't:

- **Don't** return to dark dashboard chrome, glowing cards, gradients, or decorative metric tiles.
- **Don't** use red for ordinary selection or emphasis.
- **Don't** make every metric visually equal to the student probability.
- **Don't** imply causation, certainty, or institutional validity through visual confidence.
