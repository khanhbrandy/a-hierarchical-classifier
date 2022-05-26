"""
Created on 2021-02-04
Creator: khanh.tn
"""
import pandas as pd
import numpy as np
import time
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


class Mymetrics():
    def __init__(self):
        pass
    def accuracy(self, y_test, y_pred):
        acc = metrics.accuracy_score(y_test, y_pred)
        # print('Accuracy: {:.2f}%'.format(acc * 100))
        return acc
    def roc_curve(self, y_test, y_pred_proba):
        fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba)
        return fpr, tpr, threshold
    def auc_score(self, y_test, y_pred_proba):
        roc_auc = metrics.roc_auc_score(y_test, y_pred_proba)
        # print('Classifier AUC: {:.2f}%'.format(roc_auc*100))
        return roc_auc
    def precision_score(self, y_test, y_pred):
        precision_scr = metrics.precision_score(y_test, y_pred)
        # print('Precision score is {:.2f}'.format(float(precision_scr)))
        return precision_scr
    def recall_score(self, y_test, y_pred):
        recall_scr = metrics.recall_score(y_test, y_pred)
        # print('Recall score is {:.2f}'.format(float(recall_scr)))
        return recall_scr

class Myvisualization(Mymetrics):
    def __init__(self):
        pass
    def roc_auc_viz(self, y_test,y_pred_proba):
        fpr, tpr, threshold = self.roc_curve(y_test, y_pred_proba)
        roc_auc = self.auc_score(y_test, y_pred_proba)    
        gini_score=2*roc_auc-1
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = {:.2f} and GINI = {:.2f}'.format(roc_auc,gini_score))
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        pass

class Model(Myvisualization):
    def __init__(self):
        self.test_size = 0.3
        self.seed = 103
        self.xgb_clf = xgb.XGBClassifier(
                    reg_lambda= 10, 
                    reg_alpha= 10, 
                    objective= 'binary:logistic', 
                    eval_metric = "error",
                    n_estimators= 100, 
                    max_depth= 5, 
                    random_state=self.seed,
                    use_label_encoder=False
                    )

    def train_test_split(self, clf, df):
        # start = time.time()
        # print('Start splitting train/test set for {}...'.format(clf.__class__.__name__))
        X, y = df.iloc[:,1:-1].values, df.iloc[:,-1].values #Skip retailer_id
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=self.test_size, random_state=self.seed)
        # print('Done splitting train/test set for {}. Time taken = {:.1f}(s) \n'.format(clf.__class__.__name__, time.time()-start))
        return X, y, X_train, X_test, y_train, y_test


    def train(self, clf, X_train, y_train, X_test, y_test):
        if 'random_state' in clf.get_params().keys():
            clf.set_params(random_state=self.seed)
        print('Start fitting {}...'.format(clf.__class__.__name__))
        start = time.time()
        clf.fit(X_train, y_train)
        # Get predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:,1]
        # Accuracy
        acc = self.accuracy(y_test, y_pred)
        roc_auc = self.auc_score(y_test, y_pred_proba)
        precision_scr = self.precision_score(y_test, y_pred)
        recall_scr = self.recall_score(y_test, y_pred)
        print('Accuracy: {:.2f}%'.format(acc * 100))
        print('AUC: {:.2f}%'.format(roc_auc*100))
        print('Precision score: {:.2f}'.format(float(precision_scr)))
        print('Recall score: {:.2f}'.format(float(recall_scr)))
        print('Done training {}. Time taken = {:.1f}(s)'.format(clf.__class__.__name__, time.time()-start))
        pass

    def predict(self, clf, X):
        start = time.time()
        y_pred = clf.predict(X)
        y_pred_proba = clf.predict_proba(X)[:,1]
        print('Done predicting!. Time taken = {:.1f}(s)'.format(time.time()-start))
        return y_pred, y_pred_proba

    def cross_validate(self, clf, X, y):
        start = time.time()
        kfold = model_selection.StratifiedKFold(n_splits=4,shuffle=True, random_state=self.seed)
        results = model_selection.cross_val_score(clf, X, y, scoring='roc_auc',cv=kfold)
        print('Done cross-validation. Validated AUC: {:.2f} (+/- {}). Time taken = {:.1f}(s)'.format(results.mean()*100, results.std()*100, time.time()-start))
        pass



