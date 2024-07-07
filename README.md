# Telecom Customer Churn Prediction

This project aims to predict customer churn for a telecom company using machine learning techniques. By identifying high-risk customers, the company can take proactive measures to retain them and reduce revenue loss.

## Table of Contents
- [Business Problem Overview](#business-problem-overview)
- [Understanding and Defining Churn](#understanding-and-defining-churn)
  - [Definitions of Churn](#definitions-of-churn)
  - [High-value Churn](#high-value-churn)
- [Understanding the Business Objective and the Data](#understanding-the-business-objective-and-the-data)
  - [Understanding Customer Behaviour During Churn](#understanding-customer-behaviour-during-churn)
- [Data Preparation](#data-preparation)
  - [Derive New Features](#1-derive-new-features)
  - [Filter High-value Customers](#2-filter-high-value-customers)
  - [Tag Churners and Remove Attributes of the Churn Phase](#3-tag-churners-and-remove-attributes-of-the-churn-phase)
- [Modelling](#modelling)
  - [Steps](#steps)
  - [Identifying Important Predictors](#identifying-important-predictors)
- [Code Overview](#code-overview)
  - [Libraries Used](#libraries-used)
  - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Feature Engineering](#feature-engineering)
  - [Dimensionality Reduction with PCA](#dimensionality-reduction-with-pca)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Identifying Important Predictors with Logistic Regression](#identifying-important-predictors-with-logistic-regression)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Business Problem Overview

In the highly competitive telecom industry, the average annual churn rate is 15-25%. Acquiring a new customer is 5-10 times more expensive than retaining an existing one. Therefore, predicting and preventing customer churn is crucial for telecom companies to remain profitable.

## Understanding and Defining Churn

The telecom industry has two main payment models:
1. **Postpaid**: Customers pay after using the services on a monthly/annual basis. Churn is easily identified when customers terminate services.
2. **Prepaid**: Customers pay in advance for services. Churn is harder to identify as customers can stop usage without notice. This project focuses on the prepaid model, which is more common in India and Southeast Asia.

### Definitions of Churn
- **Revenue-based churn**: Customers who have not used any revenue-generating facilities (calls, SMS, internet) over a given period.
- **Usage-based churn**: Customers with no usage (incoming or outgoing) over a period of time. This definition is used in this project.

### High-value Churn
In the Indian and Southeast Asian market, 80% of revenue comes from the top 20% of customers. Focusing on reducing churn among these high-value customers can significantly reduce revenue leakage.

## Understanding the Business Objective and the Data

The dataset spans four months (June to September, encoded as 6 to 9). The goal is to predict churn in the fourth month using features from the first three months.

### Understanding Customer Behaviour During Churn
Customers typically go through three phases before churning:
1. **Good phase**: Customer is satisfied with the service.
2. **Action phase**: Customer experiences issues or receives compelling offers from competitors. Identifying high-risk customers in this phase allows for corrective actions.
3. **Churn phase**: Customer has churned. Data from this phase is not available at the time of prediction.

## Data Preparation

Key steps in data preparation include:

### 1. Derive New Features
Use business understanding to create features that could indicate churn risk.

### 2. Filter High-value Customers
Define high-value customers as those who recharged ≥70th percentile of the average recharge amount in the first two months.

### 3. Tag Churners and Remove Attributes of the Churn Phase 
Tag churned customers based on no calls or mobile internet usage in the fourth month. Remove attributes from the churn phase.

## Modelling

Build models to predict churn and identify key churn predictors. 

### Steps
1. Preprocess data 
2. Conduct exploratory analysis
3. Derive new features
4. Reduce dimensionality with PCA
5. Train and tune models, handling class imbalance
6. Evaluate models using metrics focused on identifying churners 
7. Select the best model

### Identifying Important Predictors
Build a separate logistic regression model to identify important churn predictors. Handle multicollinearity. Visualize key predictors and recommend strategies to reduce churn based on insights.

## Code Overview

### Libraries Used
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Data Loading and Preprocessing
- Load data from CSV
- Convert date columns to datetime format
- Handle missing values

### Exploratory Data Analysis
- Analyze churn rate
- Visualize feature distributions
- Examine correlations between features

### Feature Engineering 
- Derive new features based on usage patterns
- Aggregate features over the first two months

### Dimensionality Reduction with PCA
- Scale features
- Apply PCA to reduce dimensionality

### Model Training and Evaluation
- Split data into train and test sets
- Train logistic regression, decision tree, random forest, and XGBoost models
- Tune hyperparameters using grid search
- Evaluate models using precision, recall, and F1-score
- Select the best model based on evaluation metrics

### Identifying Important Predictors with Logistic Regression
- Train a logistic regression model with L1 regularization
- Identify top predictors based on coefficient magnitudes
- Visualize important predictors using bar plots


## Requirements
- Python 3.7+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Usage
1. Install the required packages: `pip install -r requirements.txt`
2. Run the Jupyter notebooks in the `notebooks/` directory to reproduce the analysis and modelling steps.

## Results

The best performing model for predicting customer churn was an XGBoost classifier with the following hyperparameters:
- `max_depth`: 5
- `min_child_weight`: 1
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `learning_rate`: 0.1
- `n_estimators`: 100

The model achieved the following performance metrics on the test set:
- Accuracy: 0.92
- Precision: 0.85
- Recall: 0.63
- F1-score: 0.72

The top 5 important predictors of churn identified by the logistic regression model were:
1. `total_og_mou_8`: Total outgoing minutes of usage in the action phase
2. `total_ic_mou_8`: Total incoming minutes of usage in the action phase
3. `vol_2g_mb_8`: Volume of 2G data usage in the action phase
4. `vol_3g_mb_8`: Volume of 3G data usage in the action phase
5. `arpu_8`: Average revenue per user in the action phase

Based on these findings, some recommended strategies to reduce churn include:
- Offering personalized retention offers to high-risk customers, such as discounted plans or bonus data allowances
- Improving network quality and coverage to reduce usage-related issues that may lead to churn
- Proactively reaching out to customers with declining usage patterns to address any concerns or issues
- Developing targeted marketing campaigns to highlight the value and benefits of the company's services compared to competitors
  
## Contributing
Contributions are welcome! Please open an issue or submit a pull request with any suggestions or improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
