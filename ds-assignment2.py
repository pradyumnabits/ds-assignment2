#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Import all the required python modules


# In[2]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pydotplus
from IPython.display import Image


# In[2]:


### Load and understand the data 


# In[16]:


# Load the dataset (Source Kaggle.com, Credit Card Fraud Data)
df = pd.read_csv('/Users/pradyumna/pradyapps/bits/3rd-sem/data-science/assignment2/creditcard-fraud-data.csv')

# Get number of rows and columns in the dataset
display_shape = df.shape


# In[4]:


# Display the first few rows and columns in the dataset
df.head()


# In[37]:


### Prepare the data


# In[17]:


# Drop non-numeric columns and not-required fields
non_numeric_cols = ['trans_date_trans_time', 'first', 'last', 'cc_num', 'trans_num', 'dob']
df = df.drop(columns=non_numeric_cols)

# Define categorical columns for label encoding
categorical_cols = ['merchant', 'category', 'gender', 'street', 'city', 'state', 'job']

# Encode categorical variables using Label Encoding
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Split data into features and target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# In[ ]:


### Model Evaluation


# In[11]:


### Create and build the machine learning model


# In[18]:


# Create and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# In[19]:


# Make predictions
y_pred = model.predict(X_test)


# In[13]:


### Visualize the decision tree


# In[20]:


# Visualize the decision tree
feature_names = X.columns
class_names = ['Non-Fraudulent', 'Fraudulent']

dot_data = export_graphviz(model, out_file=None,
                           filled=True, rounded=True,
                           feature_names=feature_names,
                           class_names=class_names,
                           special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


# In[ ]:


### Calculate and Print the evaluation metrics


# In[21]:


# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Number of rows, columns:', display_shape)

# Print evaluation metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')


# In[22]:


# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)


# In[55]:


### Hyperparameter Tuning 


# In[23]:


# Hyperparameter tuning using GridSearchCV
param_grid = {
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred_tuned = best_model.predict(X_test)

# Calculate evaluation metrics for the tuned model
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)
recall_tuned = recall_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)

print('\nTuned Model Metrics:')
print(f'Accuracy: {accuracy_tuned:.4f}')
print(f'Precision: {precision_tuned:.4f}')
print(f'Recall: {recall_tuned:.4f}')
print(f'F1 Score: {f1_tuned:.4f}')


# In[24]:


# Compare the confusion matrices of the untuned and tuned models
conf_matrix_tuned = confusion_matrix(y_test, y_pred_tuned)
print('\nConfusion Matrix (Tuned Model):')
print(conf_matrix_tuned)


# In[ ]:




