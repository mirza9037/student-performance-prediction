# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- Students exploring how academic factors affect an estimated pass/fail probability.
- Educators and research reviewers examining the statistical method, evidence, uncertainty, and limitations.
- Student researchers preparing a probability- or statistics-based academic project and its supporting evidence.

## Product Purpose

StudPerf estimates complementary pass and fail probabilities using interpretable categorical Naive Bayes. It lets users change inputs, inspect the Bayes calculation, analyze descriptive statistics and correlations, evaluate held-out model performance, and export aggregate research evidence. Success means a first-time user can make and understand a prediction while a reviewer can verify how the data, model, uncertainty, and limitations support the research claims.

## Positioning

StudPerf combines an interactive student-performance predictor with an inspectable Bayes calculation and research-grade statistical evidence. It does not present a black-box score: the interface connects each prediction to priors, conditional likelihoods, calibration, held-out evaluation, and declared data limitations.

## Operating Context

- The public application is hosted on Streamlit Community Cloud and must remain usable without sign-in.
- The bundled workflow uses separate verified UCI Portuguese-language and mathematics cohorts with committed, pre-trained model artifacts.
- A second workflow accepts an anonymous four-factor CSV and trains a session-only model without persisting student rows.
- Users move among prediction, explanation, statistics, validation, methodology, and research-evidence tasks.
- Aggregate tables, charts, scenario CSVs, confidence intervals, and research evidence are downloadable.

## Capabilities and Constraints

- Preserve all existing data validation, probability calculations, model training and selection, charts, uploads, downloads, exports, tests, and research logic.
- Preserve both the bundled UCI workflow and the custom CSV workflow.
- Keep custom uploads and trained custom models in session memory only.
- Continue to disclose that UCI provides absences, study-time bands, G1, and G2 rather than the proposal's exact four factors.
- Predictions support research and human review; they are not guaranteed outcomes or final academic judgments.
- Remain on Streamlit for this redesign. A Next.js and FastAPI migration is not authorized.

## Brand Commitments

- Product name: StudPerf.
- Premium, minimal, sophisticated analytics experience.
- Craft references: Apple, Linear, Vercel, and premium analytics products.
- Avoid generic AI-dashboard styling, decorative card grids, excessive gradients, and visual clutter.
- Use excellent typography, intentional whitespace, restrained color, subtle borders, sophisticated data visualization, and purposeful micro-interactions.

## Evidence on Hand

- Official UCI Student Performance Portuguese-language and mathematics course datasets under CC BY 4.0; cohorts remain separate because some students occur in both files.
- Reproducible 80/20 stratified experiment with training-only model selection.
- Saved calibrated categorical Naive Bayes bundle and exact Bayes explanation tests.
- Aggregate descriptive statistics, correlations, conditional probabilities, held-out metrics, Wilson intervals, and paired bootstrap intervals.
- Twenty-five passing automated tests covering the data contract, model, uncertainty calculations, dashboard flows, custom training, and exports.
- No approved StudPerf logo, custom typeface, institutional identity, student photography, testimonial, or commercial claim is available and none should be fabricated.

## Product Principles

1. Make the probability understandable, not merely visible.
2. Separate prediction, classification, and research evidence clearly.
3. Preserve statistical honesty and surface limitations at the moment they matter.
4. Keep the primary prediction workflow fast while allowing deeper inspection on demand.
5. Present dense evidence with progressive disclosure rather than hiding or oversimplifying it.

## Accessibility & Inclusion

The redesign should target WCAG 2.2 AA, preserve keyboard operation and visible focus, support reduced motion, avoid meaning conveyed only by color, maintain readable contrast, and remain usable at 320px width and 200% text zoom.
