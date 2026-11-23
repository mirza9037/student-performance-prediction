# StudPerf UI/UX Audit and Redesign Report

## Existing architecture audit

StudPerf is a Python 3.12 Streamlit application. The root `streamlit_app.py` is the deployment entry point and delegates to `student-performance-prediction/app.py`. The application layer calls focused modules in `src/`:

- `data.py` validates the single official UCI Portuguese-language research dataset.
- `modeling.py` owns the pass probability, classification, Bayes breakdown, and what-if calculations.
- `evaluation.py` owns split, cross-validation, calibration, selection, and test evaluation.
- `reporting.py` produces descriptive statistics and the downloadable evidence archive.
- `uncertainty.py` produces Wilson and bootstrap intervals.
- `ui.py` now owns presentation tokens and reusable Probability Atlas components.

The bundled model artifact is loaded from `models/`. The six plain-language workflows are Predict, Understand the probability, Explore the data, Check model quality, Read the method, and Research evidence. The dataset and model are fixed so first-time users are never asked to make a research-design decision before predicting.

## Findings before redesign

- The underlying research and probability workflow was complete and test-covered.
- Default Streamlit styling created a generic dark dashboard with weak product identity.
- Large headings and dataframes were cramped on phone widths.
- Every metric had similar visual weight, so the current student probability did not lead the page.
- Important dataset limitations were presented as ordinary framework alerts instead of part of the analytical reading flow.
- Charts used unrelated default styling and deprecated container-width calls.
- Navigation changes could retain the previous page's scroll position.
- The nested project structure made the correct deployment entry point easy to misunderstand.

The Mathematics cohort selector and custom upload workflow were removed after user testing showed that multiple dataset paths obscured the primary task. The validated Portuguese model, calculations, charts, exports, and research evidence remain intact.

## Implemented design system

The selected direction is **Probability Atlas with color**. It uses a light cool-paper field, white analytical surfaces, graphite type, and four semantic accents:

- Cobalt: prediction, active state, and primary interaction.
- Teal: evidence and calibrated model context.
- Amber: thresholds, uncertainty, and limitations.
- Red: material research risk only.

Manrope carries page titles and large numerical output; Source Sans 3 carries controls and research text. Sections remain open and ruled rather than becoming a grid of interchangeable cards. The signature probability reading combines the main pass probability, classification, meter, fail probability, threshold, model, and target in one unit.

The complete reusable rules and tokens are documented in `DESIGN.md` and `.impeccable/design.json`.

## Functional verification

- 25 automated tests pass.
- The single Portuguese-language model loads without retraining.
- Prediction, threshold classification, what-if analysis, Bayes breakdown, descriptive statistics, correlations, conditional rates, model evaluation, uncertainty, and evidence export remain covered.
- Legacy bundle upgrades and cloud cache refresh behavior remain covered.
- The design detector returned no findings for the changed interface.

## Visual QA

Desktop was reviewed at 1440×900 and mobile at 390×844. The phone layout preserves every workflow through vertical stacking; the navigation moves into Streamlit's accessible drawer; context facts wrap; the probability ribbon becomes a two-column grid; charts remain responsive; and wide research tables scroll within the viewport. Typography, borders, semantic callouts, focus visibility, and reduced-motion behavior were checked in the implemented CSS.

The browser screenshot runner was intermittent with Streamlit's websocket startup, so valid captures were reviewed alongside Streamlit component tests and direct local runs. No source-level overflow or deprecated sizing calls remain.

## Deployment

Streamlit remains the justified platform: the application is a research tool with native Python model execution, uploads, tables, and Plotly analytics, and the approved design quality is achievable without separating the frontend and backend. A Next.js/FastAPI migration would add deployment and state complexity without improving the present research workflow enough to justify it.

Streamlit Community Cloud configuration:

- Repository: `mirza9037/student-performance-prediction`
- Branch: `master`
- Main file: `streamlit_app.py`
- Python: 3.12
- Secrets: none

Pushing to `master` triggers the existing Streamlit deployment. If the service does not refresh automatically, use **Reboot app** from the Streamlit workspace; no configuration change is required.
