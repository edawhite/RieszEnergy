import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

df = pd.read_csv("TestData.csv")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Split up the data set into training and testing subsets
y = df['Vf binary']
X = df.drop(columns=['Vf', 'Vf binary'])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#Create the model and predict values.
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=2)
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)

#Checking accuracy using the classification report
accuracy_score(y_test, y_pred_test, normalize=False)
print(classification_report(y_test,y_pred_test))

#output = pd.DataFrame({'Vf': test.Vf, 'Vf binary': predictions})
#output.to_csv('Predictions2.csv', index=False)

#Feature Analysis
importances = model.feature_importances_
std = np.std([model.feature_importances_ for tree in model.estimators_], axis=0)

model_importances = pd.Series(importances)

#Ploting Features
fig, ax = plt.subplots()
model_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()
