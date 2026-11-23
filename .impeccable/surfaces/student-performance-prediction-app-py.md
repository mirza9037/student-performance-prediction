---
version: 1
slug: "student-performance-prediction-app-py"
primary_target: "student-performance-prediction/app.py"
related_targets: ["streamlit_app.py"]
---

# StudPerf application surface

Mode: Operate. The surface serves students making a prediction and research reviewers verifying how that prediction was produced. The primary task is to adjust one profile, read complementary pass/fail probabilities, and move naturally into explanation or evidence without losing context. One verified Portuguese-language dataset and one trained model power the public product; dataset selection and uploads are intentionally absent.

## Direction contract

**THESIS:** StudPerf is a probability atlas: a working analytics product in which every important number is visually connected to its evidence and uncertainty. It refuses the generic dark dashboard made from equal cards and decorative metric tiles.

**OWN-WORLD:** A light paper-like ground, white analytical surfaces, graphite text, and precise cool-grey rules. Cobalt carries prediction and interaction, teal carries statistical evidence, amber carries uncertainty and caution, and restrained red appears only for risk or error. Manrope leads headings and Source Sans 3 carries controls and reading text. Components use open sections, ruled rows, compact controls, and a horizontal evidence ribbon instead of nested cards.

**STORY:** A student enters four factors and immediately sees a pass/fail estimate. The interface then reveals the classification threshold, raw Bayes reasoning, held-out model quality, and dataset limitations in progressively deeper layers. A reviewer can move from the same prediction into statistics, validation, methodology, and downloadable evidence without encountering a separate visual language.

**FIRST VIEWPORT:** A compact light navigation rail occupies the left edge. The work surface opens with a concise page title, fixed dataset context, and a three-step orientation strip. On Predict, a four-factor profile column sits left and a dominant result field sits right; the result contains the pass probability, fail complement, classification, threshold, and a four-part evidence ribbon. Advanced threshold control is disclosed only when requested. The primary actions are the input controls and the adjacent route into the Bayes calculation.

**FORM:** Probability Atlas, position 4 in the ordered grounded directions. Code-led Operate surface based on the approved 1440×900 HTML sketch. Signature interaction: changing an input updates both the probability field and a stepped sensitivity trace, while semantic colour reconnects the result to evidence, uncertainty, and risk across every page. seed: none (pinned — user chose “Probability Atlas … with colors”).

**FINISH:** unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
