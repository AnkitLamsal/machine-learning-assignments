import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


print("Modules Imported")
df = pd.read_csv('mushrooms.csv')
print("Dataframe read")
columns = df.columns
le = LabelEncoder()
for col in columns:
    df[col] = le.fit_transform(df[col])
print("Encoded data")
X = df.drop("class", axis = 1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2,
                                                    stratify=y, 
                                                    random_state =1111
                                                    )

def cal_score(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    print("AUC:"+str(auc))
    f1 = f1_score(y_true, y_pred)
    print("f1 Score:"+str(f1))
    precision = precision_score(y_true, y_pred)
    print("Precision:"+str(precision))
    recall = recall_score(y_true, y_pred)
    print("Recall:"+ str(recall))
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:"+str(accuracy))
    
print("K Nearest Neighbors")
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
cal_score(y_test, y_pred)
lr = LogisticRegression(max_iter = 20, n_jobs =1, solver="sag")
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Logistic Regression")
cal_score(y_test, y_pred)
