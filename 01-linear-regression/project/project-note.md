# Project: House Price Prediction

**Dataset**: Kaggle Advanced House Price Prediction (79 features)  
**Goal**: Compare linear models on real-world data

## Models & Results

| Model | CV RMSE     | Key Insight |
|-------|-------------|-------------|
| Linear Regression | 0.12403     | Baseline - overfits with 79 features |
| Ridge (L2) | 0.11512     | Shrinks all coefficients, handles correlated features |
| Lasso (L1) | 0.11457     | Feature selection - sets some θ to zero |
| ElasticNet (L1+L2) | 0.11343     | Best of both: selection + grouping |
| **Ensemble** | **0.11428** | Averaging reduces variance |

**Kaggle Result**: **0.12492**

## Key Learnings

**1. Regularization is Essential**
- 79 features → high risk of overfitting
- Adding λ penalty trades bias for lower variance

**2. L1 vs L2 Choice**

Ridge (L2):     Keeps all features, shrinks proportionally
Lasso (L1):     Automatic feature selection (sparse θ)
ElasticNet:     Combines both - ideal for correlated features

**3. Theory → Practice**
- CS229 convexity analysis → confidence these models converge
- Feature scaling matters: regularization is scale-dependent
- Ensemble averaging reduces individual model errors

**4. When to Use What**
- Many irrelevant features → **Lasso**
- All features relevant but correlated → **Ridge**  
- Both issues (like this dataset) → **ElasticNet**

## Takeaway

Linear models + proper regularization are surprisingly powerful even on high-dimensional problems. Understanding the math from Lecture 1 directly informed model selection and hyperparameter tuning.