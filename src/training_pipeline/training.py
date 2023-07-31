# -*- coding: utf-8 -*-
import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='(%(asctime)s) [%(process)d] %(levelname)s: %(message)s')

import argparse
import scipy.stats
import numpy as np
import numpy as np
import pandas as pd
import pickle as pkl

from datetime import datetime
from tqdm import tqdm

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold, cross_val_predict, train_test_split
from sklearn import metrics

from .config import ModelType, lr_tuned_parameters, rf_tuned_parameters, svm_tuned_parameters, gbdt_tuned_parameters

import sys
sys.path.append("..")
from common_utils.utils import pkl_dump,pkl_load

# def run_experiment(clf, params, task, nb, nit, model_type, train_test_path, output_path):
#     print(f"current task: {task} {model_type}")
#     model_dump = f"{task}year_{model_type}_model.pkl"

#     tr_dx, tr_dy, ts_dx, ts_dy = pkl_load(f"expr_data_CC{task}yr_train_test.pkl", train_test_path)
    
#     cv_model = RandomizedSearchCV(clf, params, scoring='roc_auc', n_jobs=nb, 
#                                     cv=StratifiedKFold(n_splits=5, shuffle=True), 
#                                     verbose=1, n_iter=nit)
#     cv_model.fit(tr_dx, tr_dy)
#     opt_clf = cv_model.best_estimator_
#     pkl_dump(opt_clf, model_dump, output_path)

#     preds = opt_clf.predict_proba(ts_dx)
#     pkl_dump(preds, f"{task}year_{model_type}_preds.pkl", output_path)

#     idx = np.argmax(opt_clf.classes_)
#     preds_1 = list(map(lambda x: x[idx], preds))

#     auc_score = roc_auc_score(ts_dy, preds_1)
#     fprs, tprs, ths = roc_curve(ts_dy, preds_1)
#     print("auc_score is : ",auc_score)
    
#     J_idx = np.argmax(tprs - fprs)
#     fpr, tpr, th = fprs[J_idx], tprs[J_idx], ths[J_idx]
#     auc_score1 = auc(fprs, tprs)

#     sen = tpr
#     spe = 1 - fpr
#     stats = [sen, spe, auc_score1]

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def print_with_ci(performance, name):
    m, m_l, m_h = mean_confidence_interval(performance[name])
    print('%s: %.3f (%.3f, %.3f)'%(name, m, m_l, m_h))

def run_experiment(X, y, clf, run_times, train_test_path, output_path):
    
    performance = pd.DataFrame()

    metrics_average = 'binary'
    labels = None #[0]
    for i in range(run_times):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        clf.fit(X_train,y_train.values.ravel())
        y_pred=clf.predict(X_test)
        y_prob=clf.predict_proba(X_test)[:,1]

        performance = performance.append({
                'i': i,
                'f1': metrics.f1_score(y_test, y_pred, average=metrics_average, labels = labels),
                'precision': metrics.precision_score(y_test, y_pred, average=metrics_average, labels = labels),
                'recall': metrics.recall_score(y_test, y_pred, average=metrics_average, labels = labels),
                'auc_roc': metrics.roc_auc_score(y_test, y_prob)
            }, ignore_index=True)

    print("--------------------------------------------------")
    print_with_ci(performance, 'f1')
    print_with_ci(performance, 'precision')
    print_with_ci(performance, 'recall')
    print_with_ci(performance, 'auc_roc')
    print("--------------------------------------------------")

def main(args):

    train_test_path = args.train_test_path
    output_path = args.output_path
    model_type = args.model_type
    run_times = args.run_times

    X = data.iloc[:,2:]
    y = data.iloc[:,1:2]

    if model_type is ModelType.M_LR:
        clf = LogisticRegression(warm_start=True)
        tuned_parameters = lr_tuned_parameters
    elif model_type is ModelType.M_RF:
        clf = RandomForestClassifier()
        tuned_parameters = rf_tuned_parameters

    elif model_type is ModelType.M_SVM:
        clf = svm.SVC(probability=True)
        tuned_parameters = svm_tuned_parameters
    else:
        clf = GradientBoostingClassifier()
        tuned_parameters = gbdt_tuned_parameters

    run_experiment(X, y, clf, run_times, train_test_path, output_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # train_test_path
    parser.add_argument("--train_test_path", required=True, type=str, 
        help="the train_test_path")

    # output_path
    parser.add_argument("--output_path", required=True, type=str, 
        help="the output_path")

    # model_type
    parser.add_argument("--model_type", required=True, type=str, 
        help="the model type")

    # run_times
    parser.add_argument("--run_times", required=True, type=int, 
        help="the running time for calculating confidence interval")

    args = parser.parse_args()
    main(args)