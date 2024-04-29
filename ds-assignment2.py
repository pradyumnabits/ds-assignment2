#!/usr/bin/env python
# coding: utf-8

# In[51]:


### Import all the required python modules


# In[39]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pydotplus
from IPython.display import Image


# In[52]:


### Load and understand the data 


# In[89]:


# Load the dataset (Source Kaggle.com, Credit Card Fraud Data)
df = pd.read_csv('/Users/pradyumna/pradyapps/bits/3rd-sem/data-science/assignment2/creditcard-fraud-data.csv')

#Print count of rows and columns
print(f'The dataset contains {df.shape[0]} rows and {df.shape[1]} columns')


# In[90]:


# Display the first few rows and columns in the dataset
df.head()


# In[91]:


### Decision Tree Implementation


# In[92]:


# Prepare the data 

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


# In[93]:


# Create the Decision Tree Classifier with a fixed random state for reproducibility
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Print a summary of the model including the default parameter values
# This provides insight into the parameters used by the model


# In[94]:


# Make predictions on the testing dataset, using the trained Decision Tree Classifier
y_pred = model.predict(X_test)


# In[95]:


### Model Evaluation


# In[96]:


# Visualize the decision tree using graphviz
# Extract feature names and class names for visualization
feature_names = X.columns
class_names = ['Non-Fraudulent', 'Fraudulent']

# Generate dot data for the decision tree visualization
dot_data = export_graphviz(model, out_file=None,
                           filled=True, rounded=True,
                           feature_names=feature_names,
                           class_names=class_names,
                           special_characters=True)

# Create a graph from the dot data
graph = pydotplus.graph_from_dot_data(dot_data)

# Display the decision tree as an image
Image(graph.create_png())


# In[97]:


# Calculate and Print the evaluation metrics for the untuned model

# Calculate evaluation metrics for the untuned model
# Accuracy measures the overall correctness of the model's predictions.
# Precision measures the proportion of true positive predictions among all positive predictions made by the model.
# Recall measures the proportion of true positive predictions among all actual positive instances in the data.
# F1-score is the harmonic mean of precision and recall, providing a balance between the two metrics.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')


# In[98]:


# Generate the confusion matrix for the untuned model
# Build the confusion matrix using the actual test targets and the predicted test predictions.
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix for the untuned model
print('Confusion Matrix:')
print(conf_matrix)


# In[79]:


### Hyperparameter Tuning 


# In[102]:


# Hyperparameter tuning using GridSearchCV
param_grid = {
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Create GridSearchCV instance
grid_search = GridSearchCV(model, param_grid, cv=5)

# Perform hyperparameter tuning
grid_search.fit(X_train, y_train)

# Get the best estimator from the grid search
best_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred_tuned = best_model.predict(X_test)

# Calculate evaluation metrics for the tuned model
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)
recall_tuned = recall_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)

# Print evaluation metrics for the tuned model
print('\nTuned Model Metrics:')
print(f'Accuracy: {accuracy_tuned:.4f}')
print(f'Precision: {precision_tuned:.4f}')
print(f'Recall: {recall_tuned:.4f}')
print(f'F1 Score: {f1_tuned:.4f}')


# In[103]:


# Generate the confusion matrix for the tuned model
conf_matrix_tuned = confusion_matrix(y_test, y_pred_tuned)

# Display the confusion matrix for the tuned model
print('\nConfusion Matrix (Tuned Model):')
print(conf_matrix_tuned)


# In[ ]:

