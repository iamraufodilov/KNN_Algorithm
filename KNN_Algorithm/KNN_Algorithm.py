# load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder

# load dataset
path = 'G:/rauf/STEPBYSTEP/Data/Iris.csv'
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
mydataset=pd.read_csv(path, names=headernames)
print(mydataset.head())

# split dataset
X = mydataset.iloc[:, :-1].values
y = mydataset.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# create model and train it
KNClf = KNeighborsClassifier(n_neighbors=8)
KNClf.fit(X_train, y_train)

# predict new data
y_pred = KNClf.predict(X_test)

# get the report
my_confusion = confusion_matrix(y_test, y_pred)
print("our confusion matrix is: ", my_confusion)
my_classrep = classification_report(y_test, y_pred)
print("our classification report is: ", my_classrep)
my_accuracy = accuracy_score(y_test, y_pred)
print("our accuracy score is: ", my_accuracy)