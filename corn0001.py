import numpy as np
import openpyxl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt

filepath = "E:\\霉菌花生\\p-analysis\\H\\VNIR.xlsx"
x = pd.read_excel(filepath, header=None)
y = pd.read_excel('E:\\霉菌花生\\p-analysis\\H\\label.xlsx', header=None)
# Class_label = np.unique(y) #查看标签
# print(Class_label)

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=7)

forest = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=0)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
# y_predprob = forest.predict_proba(X_test)
# # print(y_predprob)
# print(metrics.roc_auc_score(y_test, y_predprob, multi_class='ovo'))  # 预测打分
print(forest.oob_score)
print("accuracy:%f" % forest.oob_score_)
print(accuracy_score(y_test, y_pred, normalize=True))
print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(f1_score(y_test, y_pred, average='macro'))

# 调参
param_test1 = {'n_estimators': range(1, 311, 10)}
search1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
                                                        min_samples_leaf=20, max_depth=8,
                                                        max_features='sqrt', random_state=10),
                       param_grid=param_test1, scoring='average_precision', cv=3)
search1.fit(X_train, y_train)
print(search1.cv_results_)
print(search1.best_params_)
print("best accuracy:%f" % search1.best_score_)

# param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,200,20)}
# search2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=60, min_samples_leaf=20, max_features='sqrt', oob_score=True, random_state=10),
#                        param_grid = param_test2, scoring='roc_auc', iid=False, cv=5)
# search2.fit(X_train, y_train)
# search2.grid_scores_, search2.best_params_, search2.best_score_
