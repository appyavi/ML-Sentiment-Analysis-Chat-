from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pdb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# import data
a = pd.read_excel('Data_Scientist_Scenario_Exercise.xls')
# Drop rows which have (NaN) values(missing value)
b = a.dropna()

# Pre-Processing
values_assignment = {}
exclude_col = ['V5', 'V6', 'V7', 'V8', 'V14', 'V15', 'V16', 'V17', 'V18']
for col in b.columns:
    if col not in exclude_col:
        all_unique_val = b[col].unique().tolist()
        values_assignment[col] = {}
        for i, val in enumerate(all_unique_val):
            values_assignment[col][val] = i
        b[col] = [values_assignment[col][item] for item in b[col]]
b['V5'] = pd.DatetimeIndex(b['V5']).astype(np.int64)/1000000000


# Make a data and label seperate
data = b.drop(['V17'], axis=1)
label = b['V17']

# Feature Selection
X = data
y = label
model = ExtraTreesClassifier()
model.fit(X, y)

important_features = pd.Series(model.feature_importances_, index=X.columns)
important_features.nlargest(10).plot(kind='barh')
plt.show()
# Select 5 largest features
top = important_features.nlargest(5)
for col in X.columns:
    if col in top:
        pass
    else:
        X.drop(col, axis=1, inplace=True)
print(X.head())

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3)

# Classification using KNN algorithm
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
print("#"*30 + ' With KNN classifier ' + '#'*30)
print('Accuracy of K-NN classifier on training set: {:.2f}'
      .format(knn.score(x_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
      .format(knn.score(x_test, y_test)))

# Prediction
y_pred = knn.predict(x_test)
print("#"*30)
print("Accuracy score:", end='')
print(accuracy_score(y_test, y_pred))
print("#"*30)
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("#"*30)
print("Classification report:")
print(classification_report(y_test, y_pred))
print("#"*30)
f1 = f1_score(y_test, y_pred)
print('F1 score: %f' % f1)
# plt.show()

model = ExtraTreesClassifier()
model.fit(x_train, y_train)
print("#"*30 + ' With Extra Trees classifier ' + '#'*30)
print('Accuracy of ExtraTrees classifier on training set: {:.2f}'
      .format(model.score(x_train, y_train)))
print('Accuracy of ExtraTrees classifier on test set: {:.2f}'
      .format(model.score(x_test, y_test)))

# Prediction
y_pred = model.predict(x_test)
print("#"*30)
print("Accuracy score:", end='')
print(accuracy_score(y_test, y_pred))
print("#"*30)
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("#"*30)
print("Classification report:")
print(classification_report(y_test, y_pred))
print("#"*30)
f1 = f1_score(y_test, y_pred)
print('F1 score: %f' % f1)


# Import the model we are using
# Instantiate model with 1000 decision trees
randomforest = RandomForestClassifier(n_estimators=1000, random_state=42)
print("#"*30 + ' With RandomForest classifier ' + '#'*30)
print('Accuracy of ExtraTrees classifier on training set: {:.2f}'
      .format(model.score(x_train, y_train)))
print('Accuracy of ExtraTrees classifier on test set: {:.2f}'
      .format(model.score(x_test, y_test)))
# Train the model on training data
randomforest.fit(x_train, y_train)
# Prediction
y_pred = randomforest.predict(x_test)
print("#"*30 + ' With Extra Trees classifier')
print("#"*30)
print("Accuracy score:", end='')
print(accuracy_score(y_test, y_pred))
print("#"*30)
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("#"*30)
print("Classification report:")
print(classification_report(y_test, y_pred))
print("#"*30)
f1 = f1_score(y_test, y_pred)
print('F1 score: %f' % f1)
