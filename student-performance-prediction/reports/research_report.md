# AI-Based Student Performance Prediction Using Probability

## Abstract
This project estimates pass and fail probabilities using categorical Naive Bayes.
The target is explicitly defined as Pass = G3 >= 10; Fail = below 10.
The primary model is Calibrated Naive Bayes; it is selected by training-only cross-validation Brier score.
This version replaces the earlier dropout project. Dropout outcomes are not reused or renamed.

## Aim and objectives
Predict pass/fail probabilities, explain Bayes' theorem, analyze mean, sample standard deviation and correlation,
and let users change factors to observe prediction changes.
The four-factor CSV workflow uses attendance percentage, weekly study hours, previous marks and assignment scores.
The bundled real-data demonstration uses the closest verifiable UCI measurements and discloses the differences.

## Research questions
1. How are observed student factors associated with passing?
2. Does sigmoid calibration improve Naive Bayes probability quality on training folds?
3. How do probabilities change when one entered factor changes?
4. What data is needed to validate the four-factor model for a local institution?

## Dataset and target
Source: Official UCI mathematics dataset; real observational records.
Rows: 395; passes: 265; fails: 130.
Duplicates removed: 0; missing predictor entries: 0.
SHA-256: e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80.
Schema: uci. UCI records absences and study-time bands. It has no attendance percentage, exact study hours, or assignment scores. Previous marks are G1 and G2; neither is an assignment score.
The official demonstration uses only mathematics records, avoiding cross-subject student overlap.
Its pass mark is a declared project convention of 10/20. Custom data defaults to 50/100, configurable before training.
No synthetic training records are generated. Final marks are used only to construct labels, never as predictors.

## Feature selection and timing
Predictors: absences, studytime, G1, G2.
Gender, identifiers, final marks and other columns are excluded from model input.
G1 and G2 are prior-period grades, so the UCI model is a late-course prediction after G2, not an enrollment forecast.
The source does not establish an absence snapshot at that point: absences may include later information.
Consequently these are retrospective research results, not proof of prospective early-warning performance.
Local records must capture all predictors before the target assessment; assignment scores should be genuinely prior work.

## Probability methodology
Each input enters one of four predefined intervals. The study-time code is treated as categorical.
The model estimates a training prior P(Pass) and, for each factor, P(interval | Pass) and P(interval | Fail).
Laplace smoothing uses (class-and-interval count + 1)/(class count + 4).
Under conditional independence, J(c) = P(c) × product of P(interval_j | c).
Bayes' theorem gives P(Pass | inputs) = J(Pass)/(J(Pass) + J(Fail)).
P(Fail | inputs) = 1 - P(Pass | inputs).
Calculations use log space for numerical stability. The app reproduces every likelihood for an entered profile.
If calibrated Bayes is selected, sigmoid calibration adjusts the raw posterior; the app labels both separately.
The model is deliberately Bayesian; it is not selected against unrelated classifiers.
Fixed intervals improve interpretability but lose within-interval detail, so some slider changes leave the result unchanged.
Related grades violate the conditional independence approximation and can exaggerate certainty.

## Experimental design
Seed 42; stratified 80/20 split: 316 training and 79 testing rows.
Both raw Bayes and sigmoid-calibrated Bayes use five outer training folds.
Calibration uses five inner folds. Missing categories use most-frequent imputation fitted inside each fold.
Selection minimizes mean CV Brier score, with log loss as tie breaker, before touching test labels.
The final models remain fitted only to the training portion; the test set is not used for refitting.
A prevalence-only baseline uses the training pass rate.
The default decision threshold is 0.50; dashboard changes do not alter the fixed report metrics.

## Results
Selected model: Calibrated Naive Bayes.
Held-out accuracy: 0.873; pass precision: 0.978;
pass recall: 0.830; fail recall: 0.962;
pass F1: 0.898; ROC-AUC: 0.905;
pass average precision: 0.958;
Brier score: 0.112; log loss: 0.366.
See model_comparison.csv for CV means and standard deviations, cv_fold_metrics.csv for every fold,
and test_metrics.csv for selected, raw and prior-only comparisons.
Calibration is measured rather than assumed; small held-out samples make reliability curves uncertain.

## Statistical analysis
All exploratory statistics use the training subset. summary.csv reports count, mean, sample standard deviation
(ddof=1), minimum, median and maximum. by_outcome.csv separates pass/fail groups.
Pearson and Spearman correlation matrices include binary pass (1) / fail (0).
The mean/std of ordinal study-time codes describe codes, not hours; Spearman is the preferred ordinal summary.
conditional.csv gives observed P(Pass | factor interval) with its denominator.
These single-factor observed frequencies differ from the smoothed model likelihood P(interval | class).
Missing observations are excluded pairwise from descriptive statistics, not filled with invented measurements.
Correlations and what-if curves show associations and model sensitivity, not causal effects.

## Ethics and limitations
The dataset is small, historical and from two Portuguese secondary schools. It does not establish validity
for universities or Pakistani institutions. Gender exclusion does not remove proxy bias.
No completed subgroup fairness audit is claimed in this revision.
Predicted probabilities are uncertain estimates, not guaranteed outcomes, intelligence or potential.
A user can receive a likely-fail label and still pass. Use human review and supportive action.
Uploads must be anonymous; the app processes custom records in session memory and does not write them to disk.
The hosting provider still operates the network infrastructure.
Custom data must use one row per student: repeated students require grouped splitting beyond this prototype.
Users must verify outcome completeness, source quality, collection time, sample size and local pass criteria.

## Conclusion and future work
The application demonstrates pass/fail Bayes prediction, transparent conditional probability and descriptive statistics.
Collect a representative, consented four-factor dataset for the intended institution, validate on a future cohort,
assess subgroup performance and probability calibration, and choose an operational threshold with educators.
Do not invent assignment or attendance measurements to make the UCI schema appear to match the proposal.

## References
- Cortez, P. (2008). Student Performance. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T (CC BY 4.0).
- Scikit-learn Naive Bayes: https://scikit-learn.org/stable/modules/naive_bayes.html
- Scikit-learn calibration: https://scikit-learn.org/stable/modules/calibration.html
- UNESCO Recommendation on the Ethics of AI: https://www.unesco.org/en/articles/recommendation-ethics-artificial-intelligence
