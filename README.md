# Predicting Student Dropout Using Early Academic Indicators

## Overview
This project builds a machine learning model to identify students at risk of dropping out, using only information available by the end of their **first semester**. The goal is to support early-intervention policies in higher education: if institutions can flag at-risk students early, they can act before it's too late.

## Dataset
[Predict Students' Dropout and Academic Success](https://www.kaggle.com/datasets/ankanhore545/dropout-or-academic-success) (Kaggle, originally from the UCI Machine Learning Repository). It contains records for 4,424 students across demographic, socio-economic, and academic variables, with a target outcome of **Dropout**, **Enrolled**, or **Graduate**.

## Methodology

### 1. Exploratory Data Analysis
- Checked data structure, types, and missing values (none found)
- Examined the target distribution: Graduate (2,209), Dropout (1,421), Enrolled (794)
- Cross-tabulated key categorical variables (e.g. Debtor status, Marital status) against the target
- Built a correlation heatmap and ranked all numeric features by their correlation with the target

### 2. Feature Engineering
Created two new features from first-semester data:
- **Grade_change** — difference between 1st and 2nd semester grades (later dropped, see below)
- **Approval_rate_1st** — ratio of approved to enrolled curricular units in semester 1, capturing academic performance more informatively than raw counts

### 3. Addressing Data Leakage
An earlier version of the model included second-semester academic data (grades, approved units), which pushed accuracy to ~77%. However, this information is only available *after* a student has already stayed enrolled or dropped out — it cannot be used for genuine early intervention. All second-semester features were removed, and the model was rebuilt using only first-semester and enrollment-time data.

### 4. Modeling
- Trained a `RandomForestClassifier` (with `class_weight='balanced'` to address class imbalance in the target)
- Compared it against `XGBoost` and `LightGBM` using 5-fold cross-validation

## Results

**Final model (first-semester data only):**
- Accuracy: 74.5%
- Enrolled class recall improved from 30% to 48% after balancing class weights

**Model comparison (5-fold CV accuracy):**
| Model | Accuracy |
|---|---|
| Random Forest | 0.744 |
| XGBoost | 0.740 |
| LightGBM | 0.749 |

**Most important features:**
1. Approval_rate_1st (engineered feature)
2. Curricular units 1st sem (grade)
3. Curricular units 1st sem (approved)
4. Admission grade
5. Previous qualification (grade)

## Key Finding
The engineered **Approval_rate_1st** feature — the proportion of enrolled courses a student actually passed in their first semester — turned out to be the single strongest predictor of dropout, ahead of raw grades or admission scores. This suggests that *consistency of performance*, not just grade level, is an early warning signal institutions could monitor from semester one.

## Policy Implication
Because the model relies only on first-semester data, it could realistically support an early-warning system: flagging students for academic or financial support (e.g. scholarships, tuition assistance) before a second semester of struggle turns into dropout.

## Tools
Python, pandas, scikit-learn, XGBoost, LightGBM, seaborn, matplotlib

## Note on the modeling process
An initial version of this model used second-semester data and reported ~77% accuracy. This was identified as data leakage and corrected — the honest, deployable model reports ~74.5% accuracy using only information available at the end of semester one.
