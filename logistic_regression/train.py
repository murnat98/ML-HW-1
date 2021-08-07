from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import pandas as pd
import numpy as np

def count_tp(labels):
    true_count = 0
    false_count = 0
    for label in labels:
        if label == 1:
            true_count += 1
        if label == 0:
            false_count += 1

    return true_count, false_count

df = pd.read_csv('classification.csv')
x = df.iloc[:, 5:7].to_numpy()
y = df.iloc[:, -1].to_numpy()

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print('acc:', accuracy_score(y_test, y_pred))
print('f1:', f1_score(y_test, y_pred))
print('recall:', recall_score(y_test, y_pred))
print('precision:', precision_score(y_test, y_pred))
print('confusion: ', confusion_matrix(y_test, y_pred))

