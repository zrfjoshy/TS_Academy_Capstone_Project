#  Smart Credit Booster
### AI-Driven Alternative Credit Scoring for Financial Inclusion in Nigeria

> **TS Academy — Data Science Capstone Project**  
> **Group 6 | Track: Supervised Learning → Classification**  
> **Submission Date: 15th March 2026**

---

##  Group Members

| Name | Email | GitHub |
|---|---|---|
| Uzoagba Ikechukwu Joshua | uzoagbajoshua47@email.com | [zrfjoshy](https://github.com/zrfjoshy/TS_Academy_Capstone_Project) |
| Mbaogu Charles Chimezie | Pulsechi3@gmail.com | [Chazgrey](https://github.com/Chazgrey/TS_Academy_Capstone_Project) |
| Olaleye Michael Temitope | omtdboss@gmail.com | [omtdboss](https://github.com/omtdboss/TS_Academy_Capstone_Project) |
| Kolapo Ifedotun Johnson | Kolapoifedotun@gmail.com | [dotunspice](https://github.com/dotunspice/TS_Academy_Capstone_Project) |
| Adeola Adesanya | adesanyaadeola1504@gmail.com | [AdeolaAdesanya](https://github.com/AdeolaAdesanya/TS_Academy_Capstone_Project) |
| Andrew Ibhagbemien | andrewibhagbemien@gmail.com | [AndrewIbhagbemien](https://github.com/AndrewIbhagbemien/TS_Academy_Capstone_Project) |
| Onah Onyedikachukwu Gaius | 001gaius@gmail.com | [Gaius-byte1](https://github.com/Gaius-byte1/TS_Academy_Capstone_Project) |
| Oluwatobi Omofade Chukwuebuka | — | — |

---

## Table of Contents

- [Project Overview](#-project-overview)
- [The Problem We're Solving](#-the-problem-were-solving)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Data Cleaning & Preprocessing](#-data-cleaning--preprocessing)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering](#-feature-engineering)
- [Handling Class Imbalance](#-handling-class-imbalance)
- [Models & Results](#-models--results)
- [Key Findings](#-key-findings)
- [How to Run](#-how-to-run)
- [Repository Structure](#-repository-structure)
- [Future Work](#-future-work)

---

##  Project Overview

A large portion of working adults in Nigeria — traders, freelancers, students, gig workers — have never taken a formal bank loan. No loan history means no credit score. No credit score means no access to financial products. It is a frustrating cycle that locks out people who are actually responsible with money, simply because their financial behaviour was never formally recorded anywhere.

This project builds a machine learning solution that breaks that cycle. Instead of relying on credit bureau records that most Nigerians do not have, we use **financial behaviour data** — how often someone delays payments, how much debt they carry relative to income, how diverse their credit mix is — to predict which credit category a person falls into: **Good**, **Standard**, or **Poor**.

We framed this as a **multi-class classification problem** and trained three models against each other to find the best-performing approach, with a strong focus on handling real-world data messiness and being transparent about how predictions are made.

---

## The Problem We're Solving

Traditional credit scoring has a few well-known issues:

- It only works for people who already have a credit history — which immediately excludes a huge portion of the population
- It tends to put people from lower-income or informal employment backgrounds at a disadvantage
- It gives rejected applicants no real explanation for why they were turned down or what they can do to improve

Our approach tackles this by building a model that learns from broader financial behaviour patterns. The goal is a system that is fast, reasonably accurate, and — importantly — explainable. If someone is classified as `Poor`, there should be identifiable reasons why and clear signals they can act on.

---

## Dataset

| Detail | Info |
|---|---|
| **Source** | Kaggle — Credit Score Classification Dataset |
| **Author** | Rohan Paris |
| **Link** | https://www.kaggle.com/datasets/parisrohan/credit-score-classification |
| **File Used** | `train.csv` |
| **Size** | 100,000 rows × 28 columns |
| **Target Variable** | `Credit_Score` → Good / Standard / Poor |

We picked this dataset because it contains the exact kinds of features that matter for credit risk — payment delays, outstanding debt, credit utilisation, credit mix, loan history — and it came with realistic data quality issues (dirty values, non-standard nulls, free-text fields) that gave us genuine preprocessing work to do. It is from an approved public platform and has a labelled target, making it a natural fit for supervised classification.

### Key Features

| Feature | Description |
|---|---|
| `Age` | Customer age — had dirty values (negatives, trailing underscores) |
| `Annual_Income` | Annual income |
| `Monthly_Inhand_Salary` | Net take-home monthly salary |
| `Outstanding_Debt` | Total outstanding debt |
| `Credit_Utilization_Ratio` | Ratio of credit used to credit available |
| `Credit_History_Age` | Length of credit history — stored as free text, required parsing |
| `Num_of_Delayed_Payment` | Number of times payments were delayed |
| `Delay_from_due_date` | Average days past the payment due date |
| `Credit_Mix` | Quality of credit diversity — Bad / Standard / Good |
| `Payment_of_Min_Amount` | Whether minimum payment is consistently made |
| `Payment_Behaviour` | Spending and repayment pattern category |
| `Type_of_Loan` | Comma-separated list of loan types held |
| `Credit_Score` | **Target** — Good / Standard / Poor |

---

## Project Workflow

```
1.  Load & inspect the raw dataset
2.  Data cleaning & preprocessing
3.  Exploratory Data Analysis (EDA)
4.  Feature engineering
5.  Train-test split (stratified 80/20)
6.  Class imbalance handling with SMOTE
7.  Baseline model  — Logistic Regression
8.  Advanced model  — Random Forest
9.  Advanced model  — XGBoost
10. Model comparison across all metrics
11. Confusion matrix interpretation & error analysis
12. Feature importance analysis
13. ROC-AUC analysis
14. Findings, discussion & conclusion
```

---

##  Data Cleaning & Preprocessing

The raw dataset came with a collection of real data quality issues. Here is how we handled each one:

**Non-standard null markers** — the data used things like `_`, `#F%$@D!`, and `!@9#%8` to represent missing values. We standardised all of these to `np.nan` before doing anything else to prevent silent type conversion errors downstream.

**Dirty numeric columns** — several numeric columns had trailing underscores (e.g. `28_`, `4_`). We stripped all non-numeric characters using regex, then cast to float with `errors='coerce'` so any remaining junk became `NaN` rather than crashing the pipeline.

**Impossible Age values** — negative ages and values above 100 were replaced with `NaN` and later filled using median imputation.

**`Credit_History_Age`** — stored as free text like `"22 Years and 3 Months"`. We wrote a regex parser to extract the year and month components and convert the whole thing to a single integer representing total months of credit history (`Credit_History_Months`).

**`Credit_Mix`** — ordinally encoded as Bad=0, Standard=1, Good=2 because it reflects a natural quality ordering.

**`Payment_of_Min_Amount`** — binary encoded: Yes=1, No=0, NM=NaN.

**`Payment_Behaviour`** — entries that matched no valid category were nullified before one-hot encoding.

**`Type_of_Loan`** — a multi-label comma-separated field. Rather than attempting multi-hot encoding, we converted it to a count of distinct loan types per customer, which preserves the numeric signal in a usable form.

**`Occupation`** — one-hot encoded since it is nominal with no natural ordering.

**Missing value imputation** — median imputation for all remaining numeric nulls. We used median rather than mean because financial columns like `Annual_Income` and `Outstanding_Debt` are right-skewed — the mean gets pulled upward by outliers and does not represent the typical customer well.

---

## Exploratory Data Analysis

We covered all mandatory EDA components from the project specification.

### Dataset Overview
100,000 rows, 28 original columns. After preprocessing and one-hot encoding, the feature count expands to reflect the encoded columns for `Payment_Behaviour` and `Occupation`. All columns confirmed as numeric types following the cleaning pipeline.

### Missing Values
Nine features had missing data before preprocessing:

| Feature | Missing % |
|---|---|
| Credit_Mix | 20.2% |
| Monthly_Inhand_Salary | 15.0% |
| Type_of_Loan | 11.4% |
| Credit_History_Age | 9.0% |
| Others | < 8% each |

The missing value heatmap showed gaps distributed randomly across rows rather than concentrated in specific records — a Missing At Random (MAR) pattern — which supported imputation over row deletion.

### Target Variable Distribution

| Class | Count | Percentage |
|---|---|---|
| Standard | 53,174 | 53.2% |
| Poor | 28,998 | 29.0% |
| Good | 17,828 | 17.8% |

The target is **moderately imbalanced**. `Good` credit customers are the minority class at 17.8%. Left unaddressed, this would cause models to systematically under-identify creditworthy individuals — directly contradicting the financial inclusion goal of this project. We addressed this with SMOTE.

### Numerical Feature Analysis
- Histograms with skewness annotations for 9 key features
- Boxplots confirming upper-end outliers in `Annual_Income`, `Outstanding_Debt`, and `Total_EMI_per_month` — these were retained because they carry genuine credit risk signal, and our tree-based models handle them natively
- Skewness/kurtosis table confirming most financial variables are right-skewed, justifying median imputation

### Categorical Feature Analysis
Bar charts and percentage breakdown tables for `Credit_Mix`, `Payment_of_Min_Amount`, `Payment_Behaviour`, and `Occupation`.

### Bivariate & Multivariate Analysis
- Correlation heatmap across 14 core features — `Annual_Income` and `Monthly_Inhand_Salary` are strongly correlated as expected; `Credit_Mix` is negatively correlated with `Outstanding_Debt`
- Scatter plots coloured by credit class for 4 feature pairs — clear visual separation between `Good` and `Poor` clusters in the `Outstanding_Debt` vs `Annual_Income` and payment delay plots
- Cross-tabulations of `Credit_Mix` and `Payment_of_Min_Amount` against the target — confirmed both as highly discriminative features

### Class-Stratified Distributions
Overlaid histograms by credit class for `Outstanding_Debt`, `Credit_History_Months`, `Num_of_Delayed_Payment`, `Interest_Rate`, `Credit_Utilization_Ratio`, and `Annual_Income`. This visually confirmed which features separate the classes most clearly before we even started modelling.

---

##  Feature Engineering

We built three new features from existing columns to capture financial stress signals that are not directly visible in the raw data:

| Feature | Calculation | What It Captures |
|---|---|---|
| `Debt_to_Income_Ratio` | Outstanding Debt ÷ (Annual Income + 1) | How heavily debt burden consumes a customer's income — a standard underwriting metric |
| `EMI_Burden` | Total EMI ÷ (Monthly Salary + 1) | Fraction of take-home pay locked into loan repayments; values near or above 1.0 indicate serious financial stress |
| `Credit_Pressure_Index` | Delay from Due Date × Num of Delayed Payments | A compound delinquency score combining how often and how severely payments are missed |

All three features showed class-level separation when plotted against the target, and all three appeared in the top-15 feature importance rankings for both ensemble models — confirming they added real predictive signal.

---

## Handling Class Imbalance

| Class | Before SMOTE | After SMOTE |
|---|---|---|
| Standard | ~42,500 | Equalised |
| Poor | ~23,200 | Equalised |
| Good | ~14,300 | Equalised |

We used **SMOTE (Synthetic Minority Over-sampling Technique)** rather than simple row duplication. SMOTE generates synthetic samples by interpolating between existing minority class observations in feature space, which adds diversity rather than just repeating existing data points.

Two important implementation details:
- SMOTE was applied **only to the training set** — the test set was left completely untouched so our evaluation reflects real-world class proportions
- The StandardScaler for Logistic Regression was refit on the resampled training data to stay consistent

---

## Models & Results

### Evaluation Metrics
All five metrics required by the classification track specification:

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision (weighted) | Of all predicted positives — how many were actually positive |
| Recall (weighted) | Of all actual positives — how many did we correctly identify |
| F1 Score (weighted) | Balances precision and recall — our primary comparison metric given class imbalance |
| Confusion Matrix | Full prediction breakdown enabling FP/FN error analysis |

> RMSE and other regression metrics are **not used anywhere** in this project — this is a classification task.

---

### Model 1 — Logistic Regression (Baseline)

Multinomial logistic regression trained on SMOTE-resampled, StandardScaler-normalised features. Used as the mandatory baseline — it is interpretable and gives us a performance lower-bound that the advanced models need to beat.

```
              precision    recall  f1-score   support
        Good       0.46      0.77      0.57      3,566
        Poor       0.59      0.65      0.62      5,799
    Standard       0.76      0.54      0.63     10,635
    accuracy                           0.61     20,000
```

The linear decision boundary is limiting here. Credit risk is driven by feature interactions that a straight hyperplane cannot capture well, which is exactly why the ensemble models perform significantly better.

---

### Model 2 — Random Forest (Advanced)

200 decision trees, `max_depth=15`, `min_samples_leaf=5`. Trained on unscaled SMOTE data since tree-based models are scale-invariant. The depth and leaf-size constraints act as regularisation to prevent individual trees from fitting to noise.

```
              precision    recall  f1-score   support
        Good       0.55      0.79      0.65      3,566
        Poor       0.71      0.79      0.75      5,799
    Standard       0.84      0.67      0.74     10,635
    accuracy                           0.72     20,000
```

A significant jump from the baseline — the ensemble of trees captures the non-linear relationships in the data that Logistic Regression cannot.

---

### Model 3 — XGBoost (Advanced)

300 boosting rounds, `learning_rate=0.1`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`. The subsampling parameters introduce stochasticity to guard against overfitting; the conservative learning rate allows stable convergence across all 300 rounds.

```
              precision    recall  f1-score   support
        Good       0.63      0.73      0.68      3,566
        Poor       0.74      0.76      0.75      5,799
    Standard       0.80      0.74      0.77     10,635
    accuracy                           0.75     20,000
```

Best overall performance. XGBoost's sequential error-correction mechanism — where each new tree focuses on fixing the mistakes of the previous ensemble — gives it an edge over Random Forest's parallel averaging approach on this kind of structured tabular data.

---

### Final Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.613 | 0.654 | 0.613 | 0.617 |
| Random Forest | 0.725 | 0.751 | 0.725 | 0.729 |
| **XGBoost** | **0.747** | **0.752** | **0.747** | **0.749** |

XGBoost is our best model across every metric. The progression from Logistic Regression → Random Forest → XGBoost tells a clear story: the more expressive the model, the better it handles the complexity of credit risk prediction.

---

### Error Analysis

Two error types matter most in a credit decision context:

**False Positives — predicting `Good` when the actual class is `Poor`**  
This is the most costly mistake for a lender. Approving credit for a high-risk customer can lead to loan defaults and direct financial losses. Minimising this error matters for institutional stability.

**False Negatives — predicting `Poor` when the actual class is `Good`**  
This means denying credit to someone who would have been a reliable borrower. In a financial inclusion context, this is the error that actively reinforces the problem we are trying to solve — creditworthy individuals get locked out of the system for no good reason.

XGBoost minimised both error types most effectively across all models. Its boosting mechanism specifically focuses corrective rounds on previously misclassified observations, which is why it handles the critical `Poor → Good` boundary better than the other approaches.

---

### Feature Importance

Both ensemble models independently ranked the same features as most predictive:

| Rank | Feature | Notes |
|---|---|---|
| 1 | `Credit_Mix` | Most important feature in both models by a clear margin |
| 2 | `Outstanding_Debt` | Consistently top 5 |
| 3 | `Interest_Rate` | Top 5 in both models |
| 4 | `Credit_History_Months` | Confirms the relationship between credit tenure and creditworthiness |
| 5 | `Credit_Pressure_Index` | Engineered feature — top 15 in both models |

The fact that both models independently ranked these features similarly gives us confidence that they reflect genuine patterns rather than noise.

---

## Key Findings

- **Credit mix quality is the single most predictive feature** — more than income, and more than debt levels alone
- **Ensemble models significantly outperform logistic regression** — the performance gap confirms credit risk is a non-linear problem
- **SMOTE measurably improved recall for `Good` credit customers** — the class that matters most for financial inclusion outcomes
- **All three engineered features made the top-15 importance rankings** — domain-informed feature engineering added real value
- **Payment behaviour and debt burden together tell most of the story** — customers with high delay frequency, high outstanding debt, and poor credit mix are reliably identified as `Poor` regardless of income level

---

## How to Run?

**1. Clone the repository**
```bash
git clone https://github.com/[your-username]/TS_Academy_Capstone_Project.git
cd TS_Academy_Capstone_Project
```

**2. Download the dataset**

Download `train.csv` from [Kaggle](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) and place it in the same folder as the notebook.

**3. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

**4. Run the notebook**

Open `smart_credit_booster_final_group_6.ipynb` in Jupyter and run all cells from top to bottom.

---

## Repository Structure

```
TS_Academy_Capstone_Project/
│
├── smart_credit_booster_final_group_6.ipynb   ← Main notebook (all code + results)
├── README.md                                   ← This file
└── train.csv                                   ← Download from Kaggle (link above)
```

> `train.csv` is not included in this repository. Download it from the Kaggle link above and place it in the root folder before running the notebook.

---

## Future Work

A few directions we would explore with more time:

**Real mobile money & telecom data** — integrating airtime usage, mobile wallet transactions, and digital payment patterns would move the model much closer to a real Nigeria-specific deployment use case.

**SHAP explainability** — SHAP values would generate per-customer explanations, showing exactly which features pushed a score up or down. Beyond being useful for applicants, this is a regulatory requirement in many jurisdictions before a credit model can be deployed in production.

**Fairness auditing** — a formal assessment across demographic groups (age, occupation) using metrics like equalised odds and demographic parity before any production consideration.

**Temporal modelling** — the dataset has monthly snapshots per customer that we treated as independent rows. A future version could exploit this longitudinal structure to track how creditworthiness trends over time.

**Hyperparameter optimisation** — Bayesian search or Optuna-based tuning, particularly focused on further improving recall for the `Good` minority class.

---


*Completed as part of the TS Academy Data Science Capstone Programme — March 2026*
