# 10 Year Coronary Heart Disease Prediction

## Table of contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Data cleaning](#data-cleaning)
- [Exploratory data analysis](#exploratory-data-analysis)
- [Data analysis](#data-analysis)
- [Findings](#findings)
- [ Recommendations](#recommendations)
- [Limitations](#limitations)
- [References](#references)

### Project Overview

The project uses machine learning to predict 10 year CHD risk based on clinical and lifestyle factors.The approach includes data preprocessing, feature engineering and model training for accurate predictions. The goal is to enhance early diagnoses, support medical decision-making, and contribute to preventive  healthcare strategies.

### Data Sources

The primary dataset used for this machine learning prediction is the "MGH_PredictionDataset.csv" file containing columns (features) that provided information on individual health and lifestyle.

### Tools

- Python- For data analysis and model development
- Pandas,Numpy- Data manipulation and preprocessing
- Matplotlib,Seaborn- Data visualization
- Jupyter Notebook- Interactive development environment

  ### Data cleaning

  1. Handling missing values- Imputed missing data using mean/median for numerical features and mode for categorical features.
  2. Data type convesion- Ensured correct data types (converting object data to numerical or datetime formats).
  3. Feature engineering- Created new features and extracted meaningful insights to improve model performance.
  4. Duplicate data removal- Identified and eliminated duplicate records to prevent bias in model training.
  5. Data splitting- Divided the dataset into training and testing sets to evaluate model performance effectively.
 
  ### Exploratory data analysis

  EDA involved exploring the CHD Prediction data to answer key questions such as
  1. Demographics and risk factors
     - How does age, gender, or lifestyle habits correlate with CHD risk?
  2. Featureimportance and relationships
     - Which features ( eg cholesterol levels, blood pressure and BMI) show the strongest corelation with high CHD.
     - How does different risk factors interact? (eg cholesterol, blood pressure) amoung patients?
  3. Time-based trends
     - Does CHD risk increase with age, or have there been historical trends in CHD.
  4. Predictive analysis
  5. - Can we predict whether a person is likely to develop CHD within the next 10 years (Yes/No) ?.
    
    ### Data analysis

  ```python
  # Replace the nulls with mode or mean depending on the data type
fill_values = {
'education': dataset['education'].mode()[0],
'cigsPerDay': dataset['cigsPerDay'].mean(),
'BPMeds': dataset['BPMeds'].mode()[0],
'totChol': dataset['totChol'].mean(),
'BMI': dataset['BMI'].mean(),
'heartRate': dataset['heartRate'].mean(),
'glucose': dataset['glucose'].mean()
}
```
2.```python
MODEL SELECTION TRAINING AND VALIDATION
1. Decision Tree Classifier¶
from sklearn.tree import DecisionTreeClassifier

# Initialize/Declare the Decision Tree Model
decision_tree_model = DecisionTreeClassifier()
# Fitting the model with the training data
decision_tree_model.fit(X_train, y_train)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
# calculate the accuracy
accuracy_score = decision_tree_model.score (X_test,y_test)
round(accuracy_score*100,2) # Represnted as a percantage with 2 decimal plaqce
75.24
```

  ### Findings

  1.Model performance
  - We trained a logistic regression model on the cleaned and balanced dataset.The model showed a strong accuracy of 82% on the test set.The metrics indicate that the model was relatively good at identifying both true positives and true negatives.
2.Feature correlation and importance
  - From the correlation matrix and feature analysis , we found out that significant features influencing CHD risk were cholesterol levels, blood pressure and age .Other factors like smoking, exercise and diabetes also showed a strong association with the likehood of developing CHD .For example, the correlation between smoking status and CHD was found to be particulary strong (close to 0.6), confirming the importance  of lifestyle factors in predicting CHD.
3. Limitations and further improvements
  - The model could benefit from additional features such as dietary habits or family medical history, which may provide more insights into CHD prediction.

### Recommendations

Based on the analysis, we recommendthe following actions:
- include additional features like dietary habits and family medical history to enhance the model's predictive power
- Educate healthcare providers to use predictive tools effectively ensuring they understand the factors contributing to CHD.
- Collaborate with Research Institutions to explore advanced machine learning models and stay updated on the latest methodologies in heart disease prediction
- promote lifestyle interventions (eg smoking cessation, exercise programs,and dietary counseling) for high-risk patients to reduce CHD prevalence.
- Invest in patient data by collecting and maintaining comprehensive and accurate patient data including lifestyle factors and medical history to improve accuracy of predictive models.

  ### Limitations

  1. The dataset lacked critical features such as generic predisposition, stress levels and dietary habits which could improve prediction accuracy
  2. Some features, such as systolic and diastolic blood pressure may be highly correlated, potentially leading to multicollinearity issues.
  3. Deploying the model intoreal-world hospital settings require integration with Electonic Health Records (EHRs), which may involve technical and regulatory challenges.
 
  ### References

   - [Smart Data Learning, TX], CHD Patient dataset, provided for academic research purposes.
   - Python libraries:Pandas, Numpy, Matplotlib and seaborn.
     

  

    
  

