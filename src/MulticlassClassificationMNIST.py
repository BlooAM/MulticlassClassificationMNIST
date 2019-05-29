# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:57:29 2019

@author: adam
"""
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score ,recall_score, f1_score 
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0,1])
    plt.savefig('precision_recall_vs_thresholds.png')    

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False positive fractions')
    plt.ylabel('True positive fraction')
    plt.title('ROC curve')
    plt.savefig('roc_curve.png')
    

#Fetch data
mnist = fetch_mldata('MNIST original')
X = mnist['data']
y = mnist['target']

#Split test and train set
N = 60000
X_train, X_test, y_train, y_test = X[:N], X[N:], y[:N], y[:N]
Shuffle_index = np.random.permutation(N)
X_train, y_train = X_train[Shuffle_index], y_train[Shuffle_index]

#Preprocess data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

#Example of binary classification
y_train_5 = (y_train==5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf_bin = SGDClassifier(random_state=42)
sgd_clf_bin.fit(X_train, y_train_5)
#Cross-validation
y_train_pred = cross_val_predict(sgd_clf_bin, X_train, y_train_5, cv=3)
precision_score(y_train_pred,y_train_5)
recall_score(y_train_pred,y_train_5)
f1_score(y_train_pred,y_train_5)
y_scores = cross_val_predict(sgd_clf_bin, X_train, y_train_5, cv=3, 
                             method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
fpr, tpr, tr = roc_curve(y_train_5, y_scores)
plot_roc_curve(fpr, tpr, tr)

#Multiclass classification
sgd_clf.fit(X_train_scaled,y_train)
#Cross-validation
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train_pred, y_train)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.savefig('confusion_matrix.png')
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.savefig('error_visualisation.png')

