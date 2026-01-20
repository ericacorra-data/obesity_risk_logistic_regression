#!/usr/bin/env python
# coding: utf-8

# # Obesity Risk Prediction – Logistic Regression

# ## 1.  OBJECTIVES:
# Predict obesity levels of individuals and  implement logistic regression with multi-class strategies (One-vs-All and One-vs-One) and evaluate model performance.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score


# ## 2. DATASET AND METHOD
# The dataset being used for this lab is the "Obesity Risk Prediction" dataset publically available on UCI Library under the CCA 4.0 license. 
# You can download it here: Dataset: [Obesity Level Prediction Dataset - UCI](https://archive.ics.uci.edu/ml/datasets/Obesity+Level+Prediction)(dataset included in the IBM Data Science Professional Certificate)
# 

# In[2]:


# read  into a pandas dataframe:
obesity_data = pd.read_csv("Obesity_level_prediction_dataset.csv")
obesity_data.head()


# The data set has 17 attributes in total along with 2111 samples: 

# <style type="text/css">
# .tg  {border-collapse:collapse;border-spacing:0;}
# .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
#   overflow:hidden;padding:10px 5px;word-break:normal;}
# .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
#   font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
# .tg .tg-7zrl{text-align:left;vertical-align:bottom}
# </style>
# <table class="tg"><thead>
#   <tr>
#     <th class="tg-7zrl">Variable Name</th>
#     <th class="tg-7zrl">Type</th>
#     <th class="tg-7zrl">Description</th>
#   </tr></thead>
# <tbody>
#   <tr>
#     <td class="tg-7zrl">Gender</td>
#     <td class="tg-7zrl">Categorical</td>
#     <td class="tg-7zrl"></td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">Age</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl"></td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">Height</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl"></td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">Weight</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl"></td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">family_history_with_overweight</td>
#     <td class="tg-7zrl">Binary</td>
#     <td class="tg-7zrl">Has a family member suffered or suffers from overweight?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">FAVC</td>
#     <td class="tg-7zrl">Binary</td>
#     <td class="tg-7zrl">Do you eat high caloric food frequently?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">FCVC</td>
#     <td class="tg-7zrl">Integer</td>
#     <td class="tg-7zrl">Do you usually eat vegetables in your meals?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">NCP</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl">How many main meals do you have daily?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">CAEC</td>
#     <td class="tg-7zrl">Categorical</td>
#     <td class="tg-7zrl">Do you eat any food between meals?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">SMOKE</td>
#     <td class="tg-7zrl">Binary</td>
#     <td class="tg-7zrl">Do you smoke?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">CH2O</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl">How much water do you drink daily?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">SCC</td>
#     <td class="tg-7zrl">Binary</td>
#     <td class="tg-7zrl">Do you monitor the calories you eat daily?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">FAF</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl">How often do you have physical activity?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">TUE</td>
#     <td class="tg-7zrl">Integer</td>
#     <td class="tg-7zrl">How much time do you use technological devices such as cell phone, videogames, television, computer and others?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">CALC</td>
#     <td class="tg-7zrl">Categorical</td>
#     <td class="tg-7zrl">How often do you drink alcohol?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">MTRANS</td>
#     <td class="tg-7zrl">Categorical</td>
#     <td class="tg-7zrl">Which transportation do you usually use?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">NObeyesdad</td>
#     <td class="tg-7zrl">Categorical</td>
#     <td class="tg-7zrl">Obesity level</td>
#   </tr>
# </tbody></table>
# 

# In[3]:


obesity_data.shape


# In[4]:


# check missing values
print(obesity_data.isnull().sum()) 


# In[5]:


# Dataset summary
print(obesity_data.info())


# In[6]:


obesity_data.describe().round(2)


# In[7]:


# Distribution of target variable
sns.countplot(y='NObeyesdad', data=obesity_data)
plt.title('Distribution of Obesity Levels')
plt.show()


# The dataset shows a fairly balanced distribution across obesity levels. The model will therefore not be biased toward any dominant class. In healthcare contexts, this balance allows reliable predictions for both moderate and severe overweight. 

# ## 3. DATA PREPROCESSING
# ### 3.1 Standardization of Continuous Variables
# Standardization ensures features are on the same scale, improving model performance.

# In[8]:


# Standardizing continuous numerical features
continuous_columns = obesity_data.select_dtypes(include=['float64']).columns.tolist()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(obesity_data[continuous_columns])

# Converting to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

# Combining with the original dataset
scaled_data = pd.concat([obesity_data.drop(columns=continuous_columns), scaled_df], axis=1)


# ### 3.2 One-Hot Encoding of Categorical Variables

# In[9]:


# Identifying categorical columns
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')  # Exclude target column

# Applying one-hot encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

# Converting to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Combining with the original dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)


# In[10]:


# the overall number of fields is increased to 24.
print(prepped_data.info())


# ### 3.3 Encode Target Variable

# In[11]:


# Encoding the target variable
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
prepped_data.head()


# ## 4. MODEL TRAINING AND EVALUATION

# In[12]:


## Separate the input and target data
# Preparing final dataset
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']


# In[13]:


# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ## 5. LOGISTIC REGRESSION -  One-vs-All

# In[14]:


model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)


# In[15]:


# Predictions
y_pred_ova = model_ova.predict(X_test)


# In[16]:


print("=== One-vs-All (OvA) ===")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_ova, target_names=obesity_data['NObeyesdad'].astype('category').cat.categories))


# * The OvA model achieves 76.12% accuracy, performing well overall but showing some confusion between neighboring obesity classes. Extreme classes (Insufficient_Weight, Obesity_Type_III) are predicted very well.
# * Clases in the middle (e.g., Overweight_Level_II) are more difficult to predict (f1= 0.52).
# Clinically, this means that the model is reliable for identifying high or low risk, but less reliable for distinguishing light/moderate overweight.

# In[17]:


# Normalized Confusion matrix
cm_ova = confusion_matrix(y_test, y_pred_ova, normalize='true') * 100

plt.figure(figsize=(7,5))
sns.heatmap(cm_ova, annot=True, fmt='.1f', cmap='coolwarm',
           xticklabels=obesity_data['NObeyesdad'].astype('category').cat.categories,
           yticklabels=obesity_data['NObeyesdad'].astype('category').cat.categories)
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("Real Class", fontsize=12)
plt.title("Normalized Confusion Matrix - OvA (%)", fontsize=14)
plt.show()


# The majority of the predictions are located along the diagonal. Most misclassifications occur mainly between neighboring obesity classes (e.g., Overweight , Obesity Type_I), which is reasonable. 

# In[18]:


# Coefficients reflect association strength after standardization, not causal impact
feature_importance = np.mean(np.abs(model_ova.coef_), axis=0)
sorted_idx = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(8,6))
plt.barh(X.columns[sorted_idx], feature_importance[sorted_idx], color='skyblue')
plt.gca().invert_yaxis() # the most important at the top 
plt.title("Feature Importance - OvA Logistic Regression", fontsize=14)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

top3_features = X.columns[sorted_idx][:3]
print("Top 3 most important features for OvA model:", list(top3_features))


# * Key features driving predictions are Weight, Gender_Male and Height. Weight is directly correlated with obesity, a key indicator of risk. 

# ## 6. LOGISTIC REGRESSION -  One-vs-One

# In[19]:


# Training logistic regression model using One-vs-One
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)


# In[20]:


# Predictions
y_pred_ovo = model_ovo.predict(X_test)

# Evaluation metrics for OvO
print("=== One-vs-One (OvO) ===")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_ovo, target_names=obesity_data['NObeyesdad'].astype('category').cat.categories))


# * The OvO model achieves 92.2% accuracy, which is higher than OvA. It shows strong ability to distinguish between obesity classes, with fewer misclassifications, especially between neighboring categories. The overweight Classes show lower precision and recall respect to extrem overweight classes, but acceptable (~0.80–0.91).
# 

# In[21]:


# Normalized Confusion matrix for OvO
cm_ovo = confusion_matrix(y_test, y_pred_ovo, normalize='true') * 100

plt.figure(figsize=(7,5))
sns.heatmap(cm_ovo, annot=True, fmt='.1f', cmap='coolwarm',
           xticklabels=obesity_data['NObeyesdad'].astype('category').cat.categories,
           yticklabels=obesity_data['NObeyesdad'].astype('category').cat.categories)
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("Real Class", fontsize=12)
plt.title("Normalized Confusion Matrix - OvO (%)", fontsize=14)
plt.show()



# This matrix shows fewer errors of predictions and a stronger diagonal, which means that this model distinguishes better the obesity classes.

# In[22]:


# feature importance
coefs = np.array([est.coef_[0] for est in model_ovo.estimators_])
feature_importance = np.mean(np.abs(coefs), axis=0)
sorted_idx = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(8,6))
plt.barh(X.columns[sorted_idx], feature_importance[sorted_idx],color='skyblue')
plt.gca().invert_yaxis() # the most important at the top 
plt.title("Feature Importance - OvO Logistic Regression", fontsize=14)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

top3_features = X.columns[sorted_idx][:3]
print("Top 3 most important features for OvA model:", list(top3_features))


# * Key features driving predictions are Weight, Height and Gender_Male

# ## 7. CLASS DISTRIBUTION COMPARISON

# In[23]:


# OvA
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.countplot(x=y_test, label="Real")
sns.countplot(x=y_pred_ova, color='red', alpha=0.5, label="Predicted OvA")
plt.title("Class Distribution - OvA")
plt.xlabel("Class")
plt.ylim(0,90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

# Ovo
plt.subplot(1,2,2)
sns.countplot(x=y_test, label="Real")
sns.countplot(x=y_pred_ovo, color='green', alpha=0.5, label="Predicted OvO")
plt.title("Class Distribution - OvO")
plt.xlabel("Class")
plt.ylim(0,90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()


# The comparison between actual and predicted distributions shows that the OvO model closely reproduces the real frequencies of obesity levels, while OvA seems to overestimate some classes (0,2,3) and underestimate others (1,6).  

# ## 8. PCA Visualization

# In[24]:


# PCA with 2 components
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

# Class names
class_labels = obesity_data['NObeyesdad'].astype('category').cat.categories

# Colors for each class
cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(len(class_labels))]

plt.figure(figsize=(10,8))

# Plot OvA predictions with circle markers
for i, label in enumerate(class_labels):
    idx = (y_pred_ova == i)
    plt.scatter(
        X_test_pca[idx, 0], X_test_pca[idx, 1],
        c=[colors[i]], marker='o', alpha=0.7, s=120, label=f"{label} (OvA)"
    )

# Plot OvO predictions with triangle markers
for i, label in enumerate(class_labels):
    idx = (y_pred_ovo == i)
    plt.scatter(
        X_test_pca[idx, 0], X_test_pca[idx, 1],
        c=[colors[i]], marker='^', alpha=0.5, s=120, label=f"{label} (OvO)"
    )

plt.xlabel("PCA Component 1", fontsize=12)
plt.ylabel("PCA Component 2", fontsize=12)
plt.title("PCA Projection of Predicted Classes (OvA vs OvO)", fontsize=14)

# Single legend
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# PCA shows the distribution of classes according to the predictions of OvA and OvO models and provides a low-dimensional projection that helps visually inspect overlap between predicted classes. Compared to OvO, Ova can separate the most extreme classes (Normal Weight and Obesity Type III), but struggles with the central ones (e.g., Overweight & Obesity Type I). 
# 

# In[25]:


## Summary:
# Accuracy
accuracy_ova = accuracy_score(y_test, y_pred_ova)
accuracy_ovo = accuracy_score(y_test, y_pred_ovo)

# Macro F1-score
macro_f1_ova = f1_score(y_test, y_pred_ova, average='macro')
macro_f1_ovo = f1_score(y_test, y_pred_ovo, average='macro')

# Top 3 classes with lowest F1
# Get per-class F1 scores
f1_per_class_ova = f1_score(y_test, y_pred_ova, average=None)
f1_per_class_ovo = f1_score(y_test, y_pred_ovo, average=None)

# Map back to class names
class_names = list(obesity_data['NObeyesdad'].astype('category').cat.categories)

# Pair class names with F1
f1_pairs_ova = list(zip(class_names, f1_per_class_ova))
f1_pairs_ovo = list(zip(class_names, f1_per_class_ovo))

# Sort by F1 ascending and take 3 lowest
worst3_ova = sorted(f1_pairs_ova, key=lambda x: x[1])[:3]
worst3_ovo = sorted(f1_pairs_ovo, key=lambda x: x[1])[:3]


# In[26]:


## Summary:

def print_summary_table():
    models = ["OvA", "OvO"]
    accuracies = [accuracy_ova, accuracy_ovo]
    macro_f1s = [macro_f1_ova, macro_f1_ovo]
    worst_classes = [worst3_ova, worst3_ovo]

    print(f"{'Model':<5} | {'Accuracy':<8} | {'Macro F1-score':<13} | Top 3 Classes with Lowest F1")
    print("-"*80)
    
    for i in range(len(models)):
        print(f"{models[i]:<5} | {accuracies[i]*100:>7.2f}% | {macro_f1s[i]:>12.2f} |")
        for cls, f1 in worst_classes[i]:
            print(f"{'':<31}- {cls}: {f1:.2f}")
        print("-"*80)

print_summary_table()


# ## 9. CONCLUSIONS
# - Multi-Class logistic regression models (OvA and OvO) can predict obesity levels with good accuracy
# - The OvO model is more precise, making it more suitable in clinical contexts where distinguishing between obesity levels has specific implications. This ensures that predictions do not overestimate or underestimate certain risk categories, critical for resource allocation. 
# - The most important features  (e.g., Weight, Height and Gender) are easy to measure for clinical purposes. This indicator could be the base for guiding targeted prevention programs (e.g., standardized weight/height screening for children and adults, gender specific awareness campaigns)

# ## 10. LIMITATIONS:
# - No causal interpretation: the model cannot determine whether a variable causes obesity.
# - Self-reported data Bias: lifestyle variables (diet, physical activity) are self-reported and may contain bias or inaccuracies.
# - The dataset may not represent all demographic groups equally.
# - The model should be considered a "decision-support tool", rather than a diagnostic system. 

# ## 11. OUTLOOK:
# - Compare logistic regression with more complex models such as Random Forests.
# - Test the model on a different population or dataset to assess generalizability.
# - Weight and height strongly dominate predictions, potentially masking the contribution of lifestyle variables.
# - This is not a diagnostic system and a clinical judgment remains essential.

# In[ ]:




