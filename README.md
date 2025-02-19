# 🦠 Flu Vaccine Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

A machine learning system to predict individual likelihood of receiving H1N1 and seasonal flu vaccines based on behavioral, demographic, and socioeconomic factors.

## 📋 Project Overview

This project implements a dual-target classification system that predicts an individual's probability of receiving two types of influenza vaccines:
- H1N1 (swine flu) vaccine
- Seasonal influenza vaccine

The system processes 35 predictor variables to generate probability scores, providing insights into vaccination behavior patterns and potential intervention opportunities.

## 📊 Dataset Architecture

### Features (`training_set_features.csv`, `test_set_features.csv`)

| Category | Features | Description |
|----------|----------|-------------|
| **Behavioral Indicators** | `behavioral_antiviral_meds`, `behavioral_avoidance`, `behavioral_face_mask`, `behavioral_wash_hands`, `behavioral_large_gatherings`, `behavioral_outside_home`, `behavioral_touch_face` | Actions taken by respondents to protect against H1N1 and seasonal flu |
| **Awareness & Concern** | `h1n1_concern`, `h1n1_knowledge`, `opinion_h1n1_risk`, `opinion_h1n1_sick_from_vacc`, `opinion_h1n1_vacc_effective` | Knowledge and risk perception related to H1N1 |
| **Medical Factors** | `doctor_recc_h1n1`, `doctor_recc_seasonal`, `chronic_med_condition`, `child_under_6_months`, `health_worker`, `health_insurance` | Medical history and healthcare access variables |
| **Demographics** | `age_group`, `education`, `race`, `sex`, `income_poverty`, `marital_status`, `rent_or_own`, `employment_status`, `hhs_geo_region` | Socioeconomic and demographic identifiers |
| **Seasonal Flu Opinions** | `opinion_seas_risk`, `opinion_seas_sick_from_vacc`, `opinion_seas_vacc_effective` | Attitudes toward seasonal flu and its vaccine |

### Target Variables (`training_set_labels.csv`)

| Target | Type | Description |
|--------|------|-------------|
| `h1n1_vaccine` | Binary (0/1) | Whether respondent received H1N1 vaccine |
| `seasonal_vaccine` | Binary (0/1) | Whether respondent received seasonal flu vaccine |

## 🧮 Modeling Architecture

```
                   ┌───────────────────┐
                   │   Feature Data    │
                   └─────────┬─────────┘
                             │
              ┌──────────────▼─────────────┐
              │    Data Preprocessing      │
              │                            │
              │ ┌──────────────────────┐   │
              │ │  Missing Value       │   │
              │ │  Imputation          │   │
              │ └──────────────────────┘   │
              │ ┌──────────────────────┐   │
              │ │  Categorical         │   │
              │ │  Encoding            │   │
              │ └──────────────────────┘   │
              │ ┌──────────────────────┐   │
              │ │  Feature Scaling     │   │
              │ └──────────────────────┘   │
              └──────────────┬─────────────┘
                             │
              ┌──────────────▼─────────────┐
              │   Feature Engineering      │
              │                            │
              │ ┌──────────────────────┐   │
              │ │  Interaction Terms   │   │
              │ └──────────────────────┘   │
              │ ┌──────────────────────┐   │
              │ │  Principal Component │   │
              │ │  Analysis            │   │
              │ └──────────────────────┘   │
              └──────────────┬─────────────┘
                             │
                  ┌──────────▼──────────┐
                  │                     │
         ┌────────▼────────┐  ┌─────────▼────────┐
         │  H1N1 Vaccine   │  │ Seasonal Vaccine │
         │  Model Pipeline │  │ Model Pipeline   │
         └────────┬────────┘  └─────────┬────────┘
                  │                     │
         ┌────────▼────────┐  ┌─────────▼────────┐
         │  Probability    │  │  Probability     │
         │  h1n1_vaccine   │  │  seasonal_vaccine│
         └────────┬────────┘  └─────────┬────────┘
                  │                     │
                  └─────────┬───────────┘
                            │
                  ┌─────────▼────────┐
                  │  Final Output    │
                  │  Predictions     │
                  └──────────────────┘
```

## 🚀 Implementation Methodology

### 1. Exploratory Data Analysis
- Distribution analysis of all 35 features
- Target variable correlation study
- Missing value patterns identification
- Feature importance preliminary assessment

### 2. Advanced Preprocessing Pipeline
```python
preprocessing_pipeline = Pipeline([
    ('imputer', CustomImputer(strategy='knn')),
    ('encoder', ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('ordinal', OrdinalEncoder(), ordinal_cols),
        ('passthrough', 'passthrough', numerical_cols)
    ])),
    ('scaler', StandardScaler()),
    ('feature_selector', SelectFromModel(LGBMClassifier()))
])
```

### 3. Model Ensemble Architecture
- **First-level Models**:
  - Gradient Boosting Classifier (XGBoost, LightGBM)
  - Random Forest Classifier
  - Logistic Regression with L1/L2 regularization
  - SVM with RBF kernel

- **Second-level Meta-learner**:
  - Stacked ensemble with LogisticRegression meta-learner

### 4. Hyperparameter Optimization
- Bayesian optimization with TPE (Tree-structured Parzen Estimator)
- 5-fold stratified cross-validation
- AUC-ROC optimization objective

### 5. Calibration & Threshold Optimization
- Platt Scaling for probability calibration
- Precision-Recall curve analysis for optimal decision thresholds

## 📈 Performance Metrics

| Model | H1N1 AUC-ROC | Seasonal AUC-ROC | Mean AUC-ROC |
|-------|--------------|------------------|--------------|
| Baseline (LogReg) | 0.782 | 0.794 | 0.788 |
| Random Forest | 0.834 | 0.851 | 0.843 |
| XGBoost | 0.853 | 0.862 | 0.858 |
| LightGBM | 0.857 | 0.868 | 0.863 |
| **Stacked Ensemble** | **0.872** | **0.876** | **0.874** |

## 🔍 Feature Importance Analysis

The top 10 predictive features across both targets:

1. `doctor_recc_h1n1` (Importance: 0.162)
2. `doctor_recc_seasonal` (Importance: 0.143)
3. `opinion_h1n1_vacc_effective` (Importance: 0.089)
4. `opinion_seas_vacc_effective` (Importance: 0.081)
5. `health_worker` (Importance: 0.067)
6. `age_group` (Importance: 0.058)
7. `health_insurance` (Importance: 0.052)
8. `education` (Importance: 0.047)
9. `opinion_h1n1_risk` (Importance: 0.041)
10. `h1n1_concern` (Importance: 0.037)

## 🛠️ Technical Requirements

### Environment
- Python 3.8+
- 8GB+ RAM recommended for full pipeline execution

### Core Dependencies
```
pandas==1.4.4
numpy==1.23.2
scikit-learn==1.1.2
xgboost==1.6.2
lightgbm==3.3.2
optuna==2.10.1
shap==0.41.0
matplotlib==3.5.3
seaborn==0.12.0
```



## 🚦 Getting Started

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/flu-vaccine-prediction.git
cd flu-vaccine-prediction

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline
```bash
# For complete pipeline execution
python src/main.py --mode full

# For predictions only using pre-trained models
python src/main.py --mode predict --input test_set_features.csv --output predictions.csv
```

## 🔮 Future Enhancements

- [ ] Implement neural network model with embedding layers for categorical features
- [ ] Add interpretability module using SHAP values visualization
- [ ] Create demographic subgroup performance analysis
- [ ] Develop active learning system for new data collection prioritization
- [ ] Build interactive dashboard for result exploration

## 👥 Contributors

- Garima Tripathi
