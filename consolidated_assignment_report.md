# DSC8307 — Data Mining, Modeling and Analytics
## Advent 2026 Semester Assignment
**Programme:** MSc Data Science and Analytics, Year 2  
**Course code:** DSC8307  
**Projects:** Bank Marketing and Concrete Compressive Strength

> **Academic integrity note:** This is a reproducible worked draft based on the supplied datasets. The student should review the analysis, run both notebooks, verify the outputs in their own environment, add their own critical reflections and ensure compliance with Uganda Christian University submission and attribution rules before submission.

# Executive summary
This assignment develops two predictive workflows. Project 1 classifies whether a bank customer subscribed to a term deposit. The target is imbalanced (88.7% no; 11.3% yes), so a majority-class accuracy baseline is insufficient. Duration is excluded because it is unavailable before a call and would leak information about the outcome. Of the three tested models, histogram gradient boosting ranked first by held-out average precision (0.495) and ROC-AUC (0.814), followed by random forest (average precision 0.481) and logistic regression (0.460). Under an illustrative cost ratio in which missing a subscriber costs three times an unnecessary call, a threshold near 0.20 is a defensible starting point for a pilot, subject to real campaign economics.

Project 2 predicts concrete compressive strength. The dataset has 1,030 observations and eight numeric predictors. A training-mean baseline has test MAE 13.05 MPa and RMSE 16.05 MPa. Linear regression improves substantially (MAE 7.75 MPa; RMSE 9.80 MPa; R² 0.628), but nonlinear models perform better. Histogram gradient boosting achieved MAE 2.92 MPa, RMSE 4.46 MPa and R² 0.923 on the fixed test split. This predictive performance does not establish that increasing cement will cause a particular strength change: mixture ingredients and curing conditions vary together, and causal conclusions require controlled experiments.

# Project 1 — Bank Marketing

## 1. Problem formulation and data understanding
The dataset contains 41,188 telephone-marketing records and 21 columns, including the binary outcome `y`. The task is to estimate the probability of subscription before deciding whom to contact. This is a binary classification problem with asymmetric operational consequences: unnecessary calls consume staff time, while missed subscribers represent lost opportunities.

The class distribution is 36,548 non-subscribers (88.73%) and 4,640 subscribers (11.27%). A classifier that always predicts `no` therefore achieves about 88.7% accuracy while identifying no subscribers. Accuracy alone would conceal this failure. Precision, recall, F1, average precision (area under the precision-recall curve), ROC-AUC and Brier score are more informative.

The features cover demographics and financial status (age, job, marital status, education, credit default, housing and personal loans); contact details (communication type, month and weekday); campaign history (campaign, pdays, previous contacts and previous outcome); and macroeconomic context (employment variation, consumer price index, consumer confidence, Euribor and number employed). The data dictionary identifies `unknown` as a value used for missing categorical information. It also identifies 999 in `pdays` as “not previously contacted”.

## 2. Preparation and methodological decisions
The target was encoded as 1 for `yes` and 0 for `no`. `duration` was excluded. Although it is strongly associated with the outcome, call duration is only known after a call has taken place; using it would make the model unsuitable for selecting customers before calling and would introduce target leakage. The categorical `unknown` values were retained as explicit categories because they may carry information and deleting them could remove many observations. The `pdays` sentinel was transformed into missing and a separate `previously_contacted` indicator was created to retain the distinction. Numeric fields were median-imputed and standardized; categorical fields were most-frequent-imputed and one-hot encoded. These operations were fitted within a pipeline using training data only.

An 80/20 stratified train-test split with random state 42 preserves the target prevalence in both sets. A temporal validation split would be preferable for final deployment because the intended use is to predict future campaign outcomes.

## 3. Baseline and model comparison
The majority-class baseline predicts `no` for all test records. It obtains high accuracy solely from class imbalance, but its subscriber recall and positive-class F1 are zero.

Three approaches were selected:
1. Logistic regression — interpretable linear log-odds benchmark; class weighting addresses imbalance.
2. Random forest — captures nonlinear effects and interactions through bagged decision trees; class-weighted sampling helps prevent majority domination.
3. Histogram gradient boosting — captures nonlinearities and interactions by sequentially improving tree predictions.

Held-out evidence:

| Model | ROC-AUC | Average precision | Brier score |
|---|---:|---:|---:|
| Histogram gradient boosting | 0.814 | 0.495 | 0.074 |
| Random forest | 0.801 | 0.481 | 0.090 |
| Logistic regression | 0.801 | 0.460 | 0.162 |

The ranking is based primarily on average precision because it measures precision-recall performance for the minority subscriber class. Histogram gradient boosting ranks first, random forest second and logistic regression third. ROC-AUC supports the ranking, while lower Brier score indicates better probabilistic accuracy on this split. These are holdout results, not guarantees of future performance.

## 4. Decision threshold
A threshold of 0.50 is a convention, not necessarily an economically optimal choice. Lowering the threshold increases recall and the number of potential subscribers contacted, but also increases unnecessary calls. Raising it reduces unnecessary calls but misses more subscribers.

For a transparent illustrative decision rule, assign a false negative (missed subscriber) cost of 3 and a false positive (unnecessary call) cost of 1. The threshold is selected by minimizing `(FP + 3 × FN) / number of test records` over a prespecified grid. In the supplied holdout, a threshold around 0.20 is the cost-minimizing tested option for histogram gradient boosting. This is a starting point only: the institution should estimate the actual cost of a call, expected net contribution from a deposit, agent capacity and contact policy. If call capacity is fixed, ranking customers and selecting the top-N may be more practical than a fixed threshold.

## 5. Recommendation and limitations
Pilot histogram gradient boosting as a ranking model, with a provisional threshold near 0.20 only under the stated cost assumption. Monitor precision, recall, conversion, cost per acquired depositor and performance drift. Use a later campaign as temporal validation, calibrate predicted probabilities, and review customer consent, privacy, fairness and contact-frequency rules. The model estimates associations in historical data; it does not establish that contacting a high-scoring customer causes subscription.

# Project 2 — Concrete Compressive Strength

## 1. Problem formulation and data understanding
The dataset has 1,030 observations and nine numeric columns. Eight predictors describe cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate and curing age. The response is compressive strength in MPa. Ingredient measurements are kg/m³ and age is measured in days. This is a supervised regression task.

The response ranges from approximately 2.33 to 82.60 MPa, with mean about 35.82 MPa and standard deviation about 16.71 MPa. The sample contains a wide range of ages and mixture proportions. Exploratory plots and the correlation matrix are generated in the notebook. Correlation is useful for initial exploration but does not describe all nonlinear interactions.

## 2. Preparation and baseline
The supplied data are numeric, so categorical encoding is unnecessary. The analysis retains the measured observations rather than deleting extreme values without engineering justification. A median imputer is included in the model pipeline for robustness. An 80/20 random train-test split with seed 42 is used. The baseline predicts the training-set mean for every test case.

The mean baseline test MAE is 13.05 MPa and RMSE is 16.05 MPa, with R² approximately zero. This is the reference against which more informative models are evaluated.

## 3. Linear versus nonlinear models
Linear regression is an interpretable additive benchmark. Random forest captures nonlinearities and interactions by averaging decision trees. Histogram gradient boosting builds trees sequentially to reduce residual errors. All models are evaluated on the same held-out records.

| Model | Test MAE (MPa) | Test RMSE (MPa) | Test R² |
|---|---:|---:|---:|
| Histogram gradient boosting | 2.92 | 4.46 | 0.923 |
| Random forest | 4.03 | 5.69 | 0.875 |
| Linear regression | 7.75 | 9.80 | 0.628 |
| Mean baseline | 13.05 | 16.05 | approximately 0.000 |

The nonlinear models substantially outperform the linear model, supporting the view that strength depends on nonlinear relationships and interactions among age and mixture ingredients. Histogram gradient boosting is the strongest tested model on this fixed split. Five-fold cross-validation on the training data is also included in the notebook to examine stability beyond one holdout.

## 4. Residual diagnostics and weakest predictions
Residuals are defined as actual minus predicted strength. The notebook plots residuals against fitted values, the residual distribution and actual against predicted strength. It also lists the ten largest absolute errors and summarizes absolute error by age bands. These diagnostics help identify systematic bias, heteroscedasticity, outliers and subgroups where the model is less reliable.

On the fixed split, the histogram gradient boosting model has an overall MAE of approximately 2.92 MPa and RMSE of approximately 4.46 MPa. The age-band diagnostic should be interpreted with the group sample counts: some bands have fewer test observations, and their mean absolute errors may be unstable. Large-error records deserve inspection for unusual mixture combinations or sparse regions of the predictor space. Predictions should not be extrapolated beyond the range of the supplied formulations.

## 5. Cement-strength association and causal interpretation
A predictive association between cement content and compressive strength does not prove what will happen if cement is changed in a real mixture. The dataset is observational: cement content is not varied independently of water, slag, fly ash, superplasticizer, aggregate composition and curing age. These factors can confound or modify the observed association. A correlation coefficient, regression coefficient or feature-importance score describes predictive association under the observed data distribution; it is not automatically the causal effect of an intervention.

To estimate the effect of changing cement, conduct controlled and replicated concrete trials. Vary cement according to a planned experimental design while controlling or systematically varying water-to-binder ratio, supplementary cementitious materials, aggregates, admixtures, mixing, curing and testing conditions. Randomization, replication, uncertainty intervals and validation against applicable engineering standards are required. A predictive model can help screen candidate mixtures, but laboratory tests and engineering judgment remain essential.

# Overall conclusion
The two projects demonstrate why baselines, suitable metrics and critical interpretation matter. For bank marketing, class imbalance makes accuracy inadequate and the leakage-prone call-duration variable must be excluded from a pre-call model. Histogram gradient boosting is the strongest of the tested classifiers on the supplied holdout, with threshold selection dependent on business costs. For concrete strength, nonlinear tree ensembles outperform linear regression and the mean baseline, but residual diagnostics and independent validation remain necessary. Neither predictive model should be interpreted as evidence of causation or deployed without monitoring and domain oversight.

# Reproducibility
The package contains two notebooks, both CSV datasets, the supplied data dictionaries and `requirements.txt`. From this directory, install dependencies using `pip install -r requirements.txt`, then launch Jupyter and run each notebook from top to bottom. Fixed random seeds are used. The notebooks contain the data preparation, model fitting, metrics and diagnostic plots required to reproduce the reported analysis.
