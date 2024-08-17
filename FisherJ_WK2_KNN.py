#set working directory
import os
os.chdir('C:/Users/unkno/Desktop/MS Data Science/Class 9 - ANA680/Week 2/src')

#import libraries
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import confusion_matrix
import pickle as pkl

#fetch data
breast_cancer_wisconsin_original = fetch_ucirepo(id=15)

#separate features and target 
X = breast_cancer_wisconsin_original.data.features
y = breast_cancer_wisconsin_original.data.targets

#flatten y into a 1D array
y = np.ravel(y)

#check for NaN values
print('Number of NaN values in X: ', X.isnull().sum())

#replace NaN values of 'Bare_nuclei' feature column with the mode of that column
print('Number of NaN values in X: ', np.isnan(X).sum())
X['Bare_nuclei'] = X['Bare_nuclei'].fillna(X['Bare_nuclei'].mode()[0])

#check for NaN values
print('Number of NaN values in X: ', np.isnan(X).sum())

#split data into training and testing sets (testing = 25% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=66)

## RFE FEATURE SELECTION
#initialize a Logistic Regression model to use for feature selection
model = LogisticRegression()

# Create the RFE model and select the top features
rfe = RFE(estimator=model, n_features_to_select=7)  # Adjust n_features_to_select to your needs
rfe = rfe.fit(X_train, y_train)

# Print selected features
print('Selected Features: ', rfe.support_)
print('Feature Ranking: ', rfe.ranking_)

# Transform the dataset to contain only the selected features
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

#print the features selected
print('Features Selected: ', X.columns[rfe.support_])

## removing 'Single_epithelial_cell_size' and 'Normal_nucleoli' features had no impact to accuracy and 
## simplified model (maintained 98.857% accuracy)

# Reinitialize and train KNN model on the 7 selected features
model_rfe = KNeighborsClassifier(n_neighbors=5)
model_rfe.fit(X_train_rfe, y_train)

# Predict the test set
y_pred_rfe = model_rfe.predict(X_test_rfe)

# Calculate accuracy
accuracy_rfe = accuracy(y_test, y_pred_rfe)
print('Accuracy after RFE: ', accuracy_rfe)

# Confusion matrix
conf_matrix_rfe = confusion_matrix(y_test, y_pred_rfe)
print('Confusion Matrix after RFE: ')
print(conf_matrix_rfe)

#print the features selected
print('Features Selected: ', X.columns[rfe.support_])

# Save the model
pkl.dump(model_rfe, open('knn_model.pkl', 'wb'))