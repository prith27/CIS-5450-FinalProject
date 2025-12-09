# Health & Lifestyle Analysis - CIS 5450 Final Project

## Video Link: [Project Presentation Video](https://drive.google.com/file/d/1Wq_g_4Nbs5JRGb5VOnJ73Qawm4WYkICU/view?usp=sharing)

**Authors:**
- Prithvi Seshadri (ps27@seas.upenn.edu)
- Vamsi Krishna (nkvk@seas.upenn.edu)
- Anaya Choudhari (anaya01@seas.upenn.edu)

## 📋 Project Overview

This project analyzes a comprehensive dataset of 100,000 individuals to identify key lifestyle factors—such as physical activity, sleep, and diet—that contribute to disease risk. By leveraging machine learning techniques, we build predictive models that assess an individual's risk profile and identify distinct lifestyle clusters.

### Problem Statement

In an era where lifestyle diseases are becoming increasingly prevalent, understanding the relationship between daily habits and health outcomes is critical. Cardiovascular diseases, obesity, and diabetes are often linked to modifiable lifestyle factors. This project aims to:

1. **Exploratory Data Analysis (EDA)**: Uncover hidden patterns and correlations between lifestyle choices and health metrics using advanced interactive visualizations and SQL queries.
2. **Classification**: Predict `disease_risk` using ensemble methods, addressing the challenge of class imbalance and optimizing model performance through hyperparameter tuning and threshold optimization.

## 📊 Dataset Overview

The dataset contains **100,000 records** with a mix of numerical and categorical features:

- **Lifestyle Factors**: `daily_steps`, `sleep_hours`, `water_intake_l`, `calories_consumed`, `smoker`, `alcohol`
- **Health Metrics**: `bmi`, `systolic_bp`, `diastolic_bp`, `cholesterol`, `resting_hr`
- **Demographics**: `age`, `gender`, `family_history`
- **Target Variable**: `disease_risk` (Binary: 1 = High Risk, 0 = Low Risk)

**Dataset Source**: [Kaggle - Health and Lifestyle Dataset](https://www.kaggle.com/datasets/chik0di/health-and-lifestyle-dataset)

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required Python packages (see installation below)

### Installation

1. Clone or download this repository
2. Install required packages:

```bash
pip install plotly imbalanced-learn pandasql duckdb kagglehub[pandas-datasets] pandas numpy matplotlib seaborn scikit-learn
```

Or run the installation cell in the notebook:

```python
%pip install -q plotly imbalanced-learn pandasql duckdb kagglehub[pandas-datasets]
```

### Data Loading

The dataset is automatically loaded from Kaggle using the `kagglehub` library:

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "health_lifestyle_dataset.csv"
health_df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "chik0di/health-and-lifestyle-dataset",
    file_path
)
```

## 📁 Project Structure

```
BigData-Proj/
├── lifestyle_project_v4.ipynb    # Main analysis notebook
├── README.md                      # This file
└── [other project files]
```

## 🔍 Methodology

### 1. Exploratory Data Analysis (EDA)

- **Univariate Analysis**: Distribution analysis of key numerical variables (age, BMI, daily steps, sleep hours)
- **Bivariate Analysis**: Violin plots showing lifestyle factors vs. disease risk
- **Multivariate Analysis**: 
  - Correlation heatmaps
  - Parallel categories diagrams for risk flow visualization
- **Feature Engineering Insights**: Mutual Information analysis to detect non-linear dependencies

### 2. Data Preprocessing

- **Feature Engineering**: 
  - `cardio_stress`: Product of systolic BP and resting heart rate
  - `activity_density`: Steps per unit of BMI
  - `metabolic_strain`: Calories consumed per water intake
  - `bp_index`: Product of systolic and diastolic BP
  - `lifestyle_risk_score`: Composite of binary risks + obesity indicator

- **Encoding**: Label encoding for categorical variables (gender, smoker, alcohol, family_history)

- **Scaling**: StandardScaler for numerical features

- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique) to balance training data

### 3. Model Implementation

#### Baseline Model: Logistic Regression
- Provides interpretable baseline with explicit odds ratios
- Expected to struggle with non-linear relationships

#### Primary Model: Random Forest Classifier
- **Rationale**: Captures non-linear relationships and feature interactions
- **Hyperparameter Tuning**: RandomizedSearchCV for:
  - `n_estimators`: Number of trees
  - `max_depth`: Tree depth control
  - `min_samples_split`: Leaf size control
  - `min_samples_leaf`: Minimum samples in leaf nodes
  - `bootstrap`: Bootstrap sampling option

#### Threshold Optimization
- Experimented with different decision thresholds (default 0.50 vs. tuned 0.15)
- Strategic decision to prioritize high recall for healthy individuals (Class 0) to maintain model credibility

### 4. Risk Profiling

- Decision Tree visualization (max_depth=3) for interpretable risk rules
- Feature importance analysis to identify top risk drivers

## 📈 Key Findings

### Data Insights

1. **Class Imbalance**: High Risk class (1) represents ~25% of the dataset, requiring SMOTE for balanced training
2. **Non-Linear Relationships**: Correlation analysis revealed weak linear relationships, validating the use of tree-based models
3. **Physiological Metrics Dominate**: Features like `cholesterol`, `sleep_hours`, `systolic_bp`, and `resting_hr` show highest Mutual Information scores (~0.28-0.30)
4. **Lifestyle Factors Show Lower Dependency**: Self-reported habits (smoking, alcohol) have lower predictive power than measurable vital signs

### Model Performance

**Final Deployed Configuration:**
- **Model**: Hyperparameter-tuned Random Forest Classifier
- **Threshold**: Default (0.50)
- **Performance**:
  - High recall for healthy individuals (0.92)
  - Moderate precision for high-risk cases (0.25)
  - Accuracy: ~0.71

**Strategic Decision**: Prioritized high recall for Class 0 (healthy individuals) to ensure model credibility and minimize false alarms, rather than maximizing sensitivity for high-risk cases.

## 🛠️ Technologies Used

- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, imbalanced-learn
- **SQL Querying**: PandasSQL, DuckDB
- **Data Source**: Kaggle Hub API

## 📝 Key Visualizations

1. **Distribution Plots**: Histograms for age, BMI, daily steps, sleep hours
2. **Violin Plots**: Lifestyle factors distribution by disease risk
3. **Correlation Heatmap**: Feature correlation matrix
4. **Parallel Categories Diagram**: Multidimensional risk flow visualization
5. **Feature Importance Bar Chart**: Top risk drivers from Random Forest
6. **Mutual Information Scores**: Non-linear feature dependencies
7. **Decision Tree Visualization**: Interpretable risk classification rules

## 🎯 Business Implications

### Deployment Strategy

The model serves as a **"High Certainty" screening tool**:
- Not a general population filter (too many misses)
- Is a confirmation tool: If the model flags a patient, the probability of risk is genuinely high
- Prioritizes trust and precision over maximum sensitivity

### Limitations

- Low sensitivity for high-risk cases at default threshold (~8% recall)
- Significant class overlap in feature space
- Relies on static survey data rather than continuous monitoring

## 🔮 Future Improvements

1. **Enhanced Data Collection**:
   - Longitudinal data tracking
   - Wearable device integration
   - Genetic markers

2. **Advanced Modeling**:
   - Deep learning approaches
   - Ensemble methods (XGBoost, LightGBM)
   - Cost-sensitive learning

3. **Feature Engineering**:
   - Temporal features
   - Explicit interaction terms
   - Clinical risk score integration

4. **Evaluation Improvements**:
   - Multi-threshold strategy
   - Probability calibration
   - External validation

5. **Interpretability**:
   - SHAP values for patient-specific explanations
   - Clinical decision support interface
   - Continuous learning feedback loop

## 📚 References

- Dataset: [Health and Lifestyle Dataset on Kaggle](https://www.kaggle.com/datasets/chik0di/health-and-lifestyle-dataset)
- Course: CIS 5450 - Big Data Analytics (University of Pennsylvania)

## 📄 License

This project is part of an academic course assignment. Please refer to the course guidelines for usage and distribution policies.

## 👥 Contact

For questions or collaboration inquiries, please contact:
- Prithvi Seshadri: ps27@seas.upenn.edu
- Vamsi Krishna: nkvk@seas.upenn.edu
- Anaya Choudhari: anaya01@seas.upenn.edu

---

**Note**: This project is for educational purposes as part of the CIS 5450 course at the University of Pennsylvania.

