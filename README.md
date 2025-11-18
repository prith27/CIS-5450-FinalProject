# CIS 5450 Final Project - Health & Lifestyle Analysis

## Authors

- *Prithvi Seshadri (ps27@seas.upenn.edu)*
- *Vamsi Krishna (nkvk@seas.upenn.edu)*
- *Anaya Choudhari (anaya01@seas.upenn.edu)*

## Project Overview

This project analyzes the relationships between lifestyle factors and health outcomes using a comprehensive health & lifestyle dataset containing 100,000 records of individuals with various lifestyle and health indicators.

The dataset includes both lifestyle factors (daily steps, sleep hours, water intake, calories consumed, smoking, alcohol consumption) and health metrics (BMI, blood pressure, cholesterol, resting heart rate). Additionally, the dataset contains a `disease_risk` variable that serves as our target for classification tasks.

## Objectives

Our analysis explores:

- **Classification**: Predicting disease risk based on lifestyle factors and health indicators
- **Regression**: Estimating health metrics (cholesterol, BMI, blood pressure) from lifestyle factors
- **Clustering**: Grouping individuals by lifestyle similarities
- **Feature Engineering & EDA**: Understanding relationships and distributions within the data

Through this analysis, we aim to identify which lifestyle factors are most predictive of health outcomes, providing insights that could inform public health initiatives and personalized health recommendations.

## Dataset

The dataset is sourced from Kaggle: [Health and Lifestyle Dataset](https://www.kaggle.com/datasets/chik0di/health-and-lifestyle-dataset)

- **Records**: 100,000 individuals
- **Features**: 16 attributes including lifestyle factors and health metrics
- **Source**: Synthetic data designed for machine learning, data science, and statistical modeling tasks

### Features

**Lifestyle Factors:**
- `age`: Age of the individual
- `gender`: Gender (Male/Female)
- `daily_steps`: Number of daily steps
- `sleep_hours`: Hours of sleep per day
- `water_intake_l`: Water intake in liters
- `calories_consumed`: Daily calories consumed
- `smoker`: Smoking status (0/1)
- `alcohol`: Alcohol consumption (0/1)

**Health Metrics:**
- `bmi`: Body Mass Index
- `resting_hr`: Resting heart rate (bpm)
- `systolic_bp`: Systolic blood pressure (mmHg)
- `diastolic_bp`: Diastolic blood pressure (mmHg)
- `cholesterol`: Cholesterol level (mg/dL)
- `family_history`: Family history of disease (0/1)
- `disease_risk`: Disease risk classification (0=Low, 1=High)

## Project Structure

```
BigData_Project/
│
├── lifestyle_project.ipynb          # Main analysis notebook
├── health_lifestyle_dataset.csv     # Dataset (downloaded from Kaggle)
├── Example_Final_Project.ipynb      # Example project template
└── README.md                        # This file
```

## Requirements

### Dependencies

Install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn dask xgboost category_encoders torch kagglehub[pandas-datasets]
```

Or install from the notebook using:

```python
%pip install -q category_encoders kagglehub[pandas-datasets]
```

### Key Libraries

- **Data Processing**: pandas, numpy, dask
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, xgboost
- **Deep Learning**: torch (PyTorch)
- **Data Loading**: kagglehub

## Getting Started

### 1. Setup

Make sure you have Python 3.7+ installed and all required dependencies.

### 2. Kaggle Authentication

To download the dataset from Kaggle, you'll need to set up Kaggle API credentials:

1. Sign up for a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to Account settings → API section
3. Download your `kaggle.json` API token
4. Place it in `~/.kaggle/kaggle.json` (on Mac/Linux) or `C:\Users\<Windows-username>\.kaggle\kaggle.json` (on Windows)
5. Set appropriate permissions: `chmod 600 ~/.kaggle/kaggle.json`

### 3. Run the Notebook

Open `lifestyle_project.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab, and run all cells sequentially.

The notebook will:
1. Load the dataset from Kaggle
2. Perform exploratory data analysis
3. Engineer features
4. Train and evaluate multiple machine learning models
5. Perform clustering analysis
6. Generate visualizations and insights

## Analysis Methodology

### 1. Data Loading & Preprocessing
- Load dataset from Kaggle using kagglehub
- Convert to Dask dataframe for efficient processing
- Data cleaning and outlier detection
- Feature engineering (BMI categories, activity levels, etc.)

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis of health metrics and lifestyle factors
- Categorical variable analysis
- Correlation heatmaps
- Box plots comparing groups
- Scatter plots showing relationships

### 3. Feature Engineering
- One-hot encoding of categorical variables
- Creation of derived features (BMI category, activity level, BP risk)
- Data preparation for different modeling tasks

### 4. Modeling

#### Classification (Disease Risk Prediction)
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

#### Regression (Cholesterol Prediction)
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

#### Clustering (Lifestyle Groups)
- K-Means clustering with elbow method for optimal k
- Cluster analysis and visualization using PCA

## Key Findings

1. **Lifestyle factors matter**: Daily steps, sleep hours, water intake, and calorie consumption are important predictors of health outcomes

2. **Health metrics are interconnected**: Strong correlations exist between different health indicators (e.g., blood pressure components, BMI and cholesterol)

3. **Model performance**: Ensemble methods consistently outperformed simpler models, suggesting complex non-linear relationships in the data

4. **Actionable findings**: The analysis can inform public health recommendations and personalized health interventions

## Results Summary

- **Classification**: Multiple models achieved strong performance in predicting disease risk, with ensemble methods showing superior accuracy
- **Regression**: Tree-based methods demonstrated better performance than linear regression for predicting cholesterol levels
- **Clustering**: K-Means successfully identified distinct lifestyle groups that relate to disease risk

## Future Work

1. **Feature Engineering**: Explore polynomial features, interactions between lifestyle factors, and time-based features
2. **Model Optimization**: Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
3. **Additional Models**: Experiment with neural networks for potentially better capture of complex relationships
4. **Advanced Clustering**: Try different clustering algorithms (DBSCAN, hierarchical clustering) and validate cluster quality
5. **External Validation**: Test models on new datasets to assess generalizability
6. **Interpretability**: Use SHAP values or LIME to better understand model predictions

## License

This project is part of CIS 5450 coursework. The dataset is publicly available on Kaggle under the [Health and Lifestyle Dataset](https://www.kaggle.com/datasets/chik0di/health-and-lifestyle-dataset).

## Acknowledgments

- Dataset provided by Kaggle user [chik0di](https://www.kaggle.com/chik0di)
- Course materials and example project template from CIS 5450

## Contact

For questions or issues, please contact:
- Prithvi Seshadri: ps27@seas.upenn.edu
- Vamsi Krishna: nkvk@seas.upenn.edu

