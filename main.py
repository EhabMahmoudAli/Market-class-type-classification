# import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
# import numpy as np
# from sklearn.feature_selection import RFE
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
# from xgboost import XGBClassifier

data1 = pd.read_csv('train.csv')
data2 = pd.read_csv('test.csv')

# Handling missing values
x = data1['X2'].median()
data1['X2'].fillna(value=x, inplace=True)
data1['X9'].fillna(value='Unknown', inplace=True)

y = data2['X2'].mean()
data2['X2'].fillna(value=y, inplace=True)
data2['X9'].fillna(value='Unknown', inplace=True)

# Dropping column X8 due to unexplained outliers
data1.drop(columns='X8', inplace=True)
data2.drop(columns='X8', inplace=True)

# String manipulation in X3
data1['X3'] = data1['X3'].replace({
    'LF': 'Low Fat',
    'low fat': 'Low Fat',
    'reg': 'Regular'
})

data2['X3'] = data2['X3'].replace({
    'LF': 'Low Fat',
    'low fat': 'Low Fat',
    'reg': 'Regular'
})

# Encoding categorical features
encoder = LabelEncoder()
categorical_columns = ['X1', 'X3', 'X5', 'X7', 'X9', 'X10']
columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X9', 'X10']
for i in categorical_columns:
    data1[i] = encoder.fit_transform(data1[i])
    data2[i] = encoder.fit_transform(data2[i])

# Scaling features
scaler = MinMaxScaler()
data1[columns] = scaler.fit_transform(data1[columns])
data2[columns] = scaler.fit_transform(data2[columns])

# plt.figure(figsize=(13,13))
# sns.heatmap(data1.corr(),annot=True,cmap='coolwarm')
# plt.show()

# X10 AND X7 ARE HIGHLY CORRELATED TO EACH OTHER
columns_to_drop = ['X7', 'X10']

# Prepare training data
x_train = data1.drop(['Y'], axis=1)
y_train = data1['Y']
x_test = data2

# svm model :


# Create and train the SVM model
svm_model = SVC(kernel='linear', C=7, random_state=42)  # You can adjust kernel and C as needed

# Perform cross-validation to estimate accuracy
cv_scores = cross_val_score(svm_model, x_train, y_train, cv=5)  # 5-fold cross-validation

# Print cross-validated accuracy scores
print(f"Cross-validated accuracy scores: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean()}")

# Fit the model on the entire training data
svm_model.fit(x_train, y_train)

# Predict on the test data
y_pred_test = svm_model.predict(x_test)

# Create submission DataFrame
submission = pd.DataFrame({
    'row_id': range(0, len(y_pred_test)),  # Start the row_id column from 0
    'label': y_pred_test
})

# Save the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)
