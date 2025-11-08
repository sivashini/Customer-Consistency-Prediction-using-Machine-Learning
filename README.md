# 🧠 Customer Consistency Prediction using Machine Learning

## 📊 Overview
This project focuses on predicting **customer consistency**, a critical business goal for identifying and retaining valuable users.  
The model categorizes customers into:
- **High-Value Consistent**
- **Medium-Value Degrowth**
- **Low-Value Consistent**

By anticipating customer behavior, businesses can deploy targeted **retention strategies** to minimize churn and maximize long-term revenue.

---

## 🧩 Problem Statement
The objective is to build a **machine learning model** that accurately predicts a customer's **Consistency Tag** based on their usage and revenue data.

This tag helps identify whether a user is likely to stay consistent, grow, or degrow in value, enabling **data-driven customer management** and **personalized marketing actions**.

---

## 📂 Dataset Overview
The dataset contains **100 customer records** with ~12 features, including:

| Category | Features |
|-----------|-----------|
| **Revenue Metrics** | `tot_rev`, `tot_voice_rev`, `total_data_rev` |
| **Usage Metrics** | `total_data_volume`, `total_calls_m_new` |
| **Categorical Variables** | `voip_usage_tag`, `Consistency_Tag` |

---

## 🧹 Data Preprocessing
Key cleaning and preparation steps:
1. **Missing Values:** Imputed using the **median** (robust against skewness).  
2. **Outlier Handling:** Applied **IQR capping** to retain all data while minimizing outlier distortion.  
3. **Encoding:**  
   - Label Encoding → `Consistency_Tag`  
   - One-Hot Encoding → `voip_usage_tag`

---

## 🔍 Exploratory Data Analysis (EDA)
- **Target Imbalance:** The ‘MV_Degrowth_New’ group was dominant → emphasized F1-score over accuracy.  
- **Feature Relationships:** Revenue and call volume are strong indicators of consistency.  
- **Multicollinearity:** Noted high correlations among revenue-based features → used selective inclusion during modeling.

---

## ⚙️ Feature Engineering
Created new derived features to improve predictive power:
- **Revenue per Call:** `tot_rev / (total_calls_m_new + voip_tot_calls)`
- **Data-to-Voice Revenue Ratio:** `total_data_rev / tot_voice_rev`

Feature selection combined:
- Random Forest feature importance  
- ANOVA F-test (`SelectKBest`)  

✅ Final Top 7 predictive features selected.

---

## 🧠 Modeling Approach
**Train-Test Split:** 70% Training | 30% Testing  
**Scaling:** StandardScaler applied to numerical features.  

### Models Tested
| Type | Model |
|------|--------|
| Baseline | Logistic Regression, Decision Tree |
| Ensemble | Random Forest, Gradient Boosting |
| Advanced | XGBoost, ANN (Multi-Layer Perceptron) |

**Hyperparameter Tuning:**  
Performed via `GridSearchCV` for ensemble models (Random Forest & Gradient Boosting).

---

## 🏆 Results
| Model | Accuracy | Key Notes |
|--------|-----------|------------|
| Logistic Regression | ~72% | Basic linear baseline |
| Random Forest (Tuned) | ~86% | Solid performance |
| **Gradient Boosting (Tuned)** | **💥 90%** | Best performing model |
| XGBoost | ~88% | Slightly below GB |
| ANN (MLP) | ~85% | Stable but lower precision |

---

## 💡 Insights & Business Value
- **Revenue & Usage** metrics are dominant predictors of customer stability.
- The **Gradient Boosting model** captures complex nonlinear relationships effectively.
- Enables proactive identification of **at-risk customers** (degrowth group).
- Facilitates **targeted retention campaigns** for high-value users.

---

## 🧰 Tech Stack
- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, xgboost, tensorflow/keras  
- **Visualization:** matplotlib, seaborn  
- **Optimization:** GridSearchCV  
- **Notebook Environment:** Jupyter / Google Colab

---

