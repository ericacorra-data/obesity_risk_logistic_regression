# Obesity Risk Prediction using Multiclass Logistic Regression
This project focuses on predicting *obesity risk levels* using a multiclass Logistic Regression approach.  
The analysis compares two common multiclass strategies: **One-vs-All (OvA)** and **One-vs-One (OvO)**. 
## Objective
- Predict obesity levels based on demographic, physical, and lifestyle features
- Implement and compare OvA vs OvO multiclass strategies
- Evaluate models using accuracy, F1-score, and confusion matrices
- Interpret feature importance for practical and clinical insights
## Dataset 
- Source: [Obesity Level Prediction Dataset - UCI](https://archive.ics.uci.edu/ml/datasets/Obesity+Level+Prediction)
- File used: `Obesity_level_prediction_dataset.csv`
## Analysis and Methods
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Modeling: One-vs-All (OvA) using `LogisticRegression(multi_class="ovr")` and One-vs-One (OvO) using `OneVsOneClassifier`
## Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score (per class and macro-average)
- Normalized confusion matrices
- Class distribution comparison (real vs predicted)
## Observations
- OvO clearly outperforms OvA, especially for neighboring obesity classes
- OvA performs well for extreme classes but struggles with mid-range categories
- OvO better reproduces the real class distribution
## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
## Limitations
- No causal inference
- Lifestyle variables are self-reported and may contain bias
- Dataset may not represent all populations equally
- Strong dominance of weight and height may mask lifestyle effects
