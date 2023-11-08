# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 05:13:17 2017

@author: benvenuto
"""

import os
import sys
import numpy
import math
import matplotlib
# set 'agg' (Anti-Grain Geometry) as renderer for matplotlib, using a non-interactive backend
# this is required in non-desktop environements (e.g. a Docker container running with a non-interactive shell)
# to activate an interactive backend un-comment the following code line:
#   matplotlib.use('TkAgg')
# see also: http://matplotlib.org/faq/usage_faq.html#what-is-a-backend
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, LogisticRegressionCV, \
    ElasticNetCV, MultiTaskLassoCV, MultiTaskLasso, RidgeCV
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from f_kmeans_new import FKMeans, PKMeans
from sim_ann_cc import SimAnnKMeans, SimAnnFKMeans
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier

if not os.path.isfile('./dfl/dfl.so'):
    os.chdir("./dfl")
    os.system("f2py -c dfl.pyf main_box.f90 wrap_mixed.f90 sd.f90")
    os.chdir("..")

# python - fortran module build from
#from dfl import H_DFL


def classification_skills(y_real, y_pred):

    cm = confusion_matrix(y_real, y_pred)

    if cm.shape[0] == 1 and sum(y_real) == 0:
        a = 0.
        d = float(cm[0, 0])
        b = 0.
        c = 0.
    elif cm.shape[0] == 1 and sum(y_real) == y_real.shape[0]:
        a = float(cm[0, 0])
        d = 0.
        b = 0.
        c = 0.
    elif cm.shape[0] == 2:
        a = float(cm[1, 1])
        d = float(cm[0, 0])
        b = float(cm[0, 1])
        c = float(cm[1, 0])
    TP = a
    TN = d
    FP = b
    FN = c

    if (TP + FP + FN + TN) == 0.:
        if (TP + TN) == 0.:
            acc = 0.  # float('NaN')
        else:
            acc = -100  # float('Inf')
    else:
        acc = (TP + TN) / (TP + FP + FN + TN)

    if TP + FN == 0.:
        if TP == 0.:
            tss_aux1 = 0.  # float('NaN')
        else:
            tss_aux1 = -100  # float('Inf')
    else:
        tss_aux1 = (TP / (TP + FN))

    if (FP + TN) == 0.:
        if FP == 0.:
            tss_aux2 = 0.  # float('NaN')
        else:
            tss_aux2 = -100  # float('Inf')
    else:
        tss_aux2 = (FP / (FP + TN))

    tss = tss_aux1 - tss_aux2

    if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) == 0.:
        if (TP * TN - FN * FP) == 0:
            hss = 0.  # float('NaN')
        else:
            hss = -100  # float('Inf')
    else:
        hss = 2 * (TP * TN - FN * FP) / ((TP + FN) *
                                         (FN + TN) + (TP + FP) * (FP + TN))

    if FP == 0.:
        if FN == 0.:
            fnfp = 0.  # float('NaN')
        else:
            fnfp = -100  # float('Inf')
    else:
        fnfp = FN / FP

    if (TP + FN) == 0.:
        if TP == 0.:
            pod = 0  # float('NaN')
        else:
            pod = -100  # float('Inf')
    else:
        pod = TP / (TP + FN)


    if (TP + FP) == 0.:
        if FP == 0.:
            far = 0.  # float('NaN')
        else:
            far = -100  # float('Inf')
    else:
        far = FP / (TP + FP)

    #acc = (a + d) / (a + b + c + d)
    #tpr = a / (a + b)
    #tnr = d / (d + c)
    #wtpr = a / (a + b) * (a + c) / (a + b + c + d) + d / (c + d) * (b + d) / (a + b + c + d)
    #pacc = a / (a + c)
    #nacc = d / (b + d)
    #wacc = a / (a + c) * (a + c) / (a + b + c + d) + d / (b + d) * (b + d) / (a + b + c + d)

    # if the cm has a row or a column equal to 0, we have bad tss
    if TP+FN == 0 or TN+FP == 0 or TP+FP == 0 or TN+FN == 0:
        tss = 0

    #, pod, acc, hss, tss, fnfp, tpr, tnr, pacc, nacc, wacc, wtpr)
    return cm.tolist(), far, pod, acc, hss, tss, fnfp


def metrics_classification(y_real, y_pred, print_skills=True):

    cm, far, pod, acc, hss, tss, fnfp = classification_skills(y_real, y_pred)

    if print_skills:
        print ('confusion matrix')
        print (cm)
        print ('false alarm ratio       \t', far)
        print ('probability of detection\t', pod)
        print ('accuracy                \t', acc)
        print ('hss                     \t', hss)
        print ('tss                     \t', tss)
        print ('balance                 \t', fnfp)

    balance_label = float(sum(y_real)) / y_real.shape[0]

    #cm, far, pod, acc, hss, tss, fnfp = classification_skills(y_real, y_pred)

    return {
        "cm": cm,
        "far": far,
        "pod": pod,
        "acc": acc,
        "hss": hss,
        "tss": tss,
        "fnfp": fnfp,
        "balance label": balance_label}


def metrics_regression(y_real, y_pred):

    mse = mean_squared_error(y_real, y_pred, multioutput='raw_values')
    r2 = r2_score(y_real, y_pred, multioutput='raw_values')
#    c_sta=float(C_statistics(y_real, y_pred))

    return {"mse": mse.tolist(), "r2": r2.tolist()}  # , "c_statistics": c_sta}


def C_statistics(y_real, y_pred):
    output = 0
    for i in range(len(y_real)):
        if y_real[i] == 0 or y_pred[i] == 0:
            continue
        output = output + \
            y_real[i] * math.log(y_real[i] / y_pred[i]) + y_pred[i] - y_real[i]

    return 2 / len(y_real) * output


def check_prediction_clustering(Y_training, labels):
    labels_01 = numpy.zeros((labels.shape))
    labels_01[labels == 0] = 1
    '''
    if sum(labels==0)>sum(labels_01==0):
        labels_ok = labels
        switch = 0
    else:
        labels_ok = labels_01
        switch = 1
    '''
    classification_1 = metrics_classification(Y_training, labels)
    classification_2 = metrics_classification(Y_training, labels_01)

    # done with tss
    if classification_1['tss'] >= classification_2['tss']:
        labels_ok = labels
        switch = 0
    else:
        labels_ok = labels_01
        switch = 1

    return labels_ok, switch


def clustering_classifier(Y_training_prediction):
    # KMeans
    est_ = KMeans(n_clusters=2,
                  init='k-means++',
                  n_init=10,
                  max_iter=300,
                  tol=0.0001,
                  precompute_distances='auto',
                  verbose=0,
                  random_state=None,
                  copy_x=True, n_jobs=1,
                  algorithm='auto')
    est_.fit(Y_training_prediction.reshape(-1, 1))

    # compute threshold
    centers = est_.cluster_centers_
    threshold = (centers[0] + centers[1]) / 2.0
    print ('centers  \t', centers[0], centers[1])
    print ('threshold\t', threshold)
    return threshold


def optimize_threshold(probability_prediction, Y_training, classification):

    if classification == 'hybrid':
        xss_vector = 0
        best_xss_threshold = clustering_classifier(probability_prediction)
        Y_best_predicted = probability_prediction > best_xss_threshold

    if classification == 'tss' or classification == 'hss':
        n_samples = 100
        step = 1. / n_samples
        # to be set in the learning phase: algo.threshold
        # TODO after the learning step, compute the thresholding parameter
        # with tss or hss.
        xss = -1.
        xss_threshold = 0
        Y_best_predicted = numpy.zeros((Y_training.shape))
        xss_vector = numpy.zeros(n_samples)
        a = probability_prediction.max()
        b = probability_prediction.min()
        for threshold in range(1, n_samples):
            xss_threshold = step * threshold * numpy.abs(a - b) + b
            # print(xss_threshold)
            Y_predicted = probability_prediction > xss_threshold
            res = metrics_classification(Y_training > 0, Y_predicted, print_skills=False)
            xss_vector[threshold] = res[classification]
            if res[classification] > xss:
                xss = res[classification]
                Y_best_predicted = Y_predicted
                best_xss_threshold = xss_threshold

    metrics_training = metrics_classification(Y_training > 0, Y_best_predicted)

    return best_xss_threshold, metrics_training, xss_vector


class HybridLogit:
    """
    FLARECAST wrapper for the scikit-learn logistic regression
    """

    def __init__(self, **parameters):

        global c_type
        try:
            HybridLogit_parameters = parameters.pop('HybridLogit')
            c_type = HybridLogit_parameters.pop('classification')
        except KeyError:
            HybridLogit_parameters = parameters

        self.classification = list(c_type.keys())[
            list(c_type.values()).index(True)]
        self.estimator = LogisticRegressionCV(**HybridLogit_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        #Xn, self._mean_, self._std_ = training_set_standardization(X)
        self.estimator.fit(X_training, Y_training)
        # predicts the probability of a flare
        probability_prediction = self.estimator.predict_proba(X_training)[:, 1]

        # threshold estimate
        # ['0'] to be uniform with multitask
        self.threshold, self.metrics_training['0'], dummy  = optimize_threshold(
            probability_prediction, Y_training, self.classification)
        self.metrics_training["feature importance"] = self.estimator.coef_.tolist()[
            0]

    def predict(self, X_testing, Y_testing_tupla=None):
        self.probability = self.estimator.predict_proba(X_testing)[:, 1]
        Y_testing_classification = self.probability > self.threshold
        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            self.metrics_testing['0'] = metrics_classification(
                Y_testing, Y_testing_classification)
        return Y_testing_classification

    def get_params(self, deep=True):
        return self.estimator.get_params(deep)

    def set_params(self, **params):
        return self.estimator.set_params(**params)


class HybridLasso:
    """
    FLARECAST wrapper for the scikit-learn logistic regression
    """

    def __init__(self, **parameters):
        global c_type
        try:
            HybridLasso_parameters = parameters.pop('HybridLasso')
            c_type = HybridLasso_parameters.pop('classification')
        except KeyError:
            HybridLasso_parameters = parameters

        self.classification = list(c_type.keys())[
            list(c_type.values()).index(True)]

        self.estimator = LassoCV(**HybridLasso_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0].ravel()
        Y_training_pre_std = Y_training_tupla[1]
        self.estimator.fit(X_training, Y_training)
        probability_prediction = self.estimator.predict(X_training)
        # threshold estimate
        '''
        self.threshold, self.metrics_training['0'], hss_vector = optimize_threshold(
            probability_prediction, Y_training, 'hss')
        self.threshold, self.metrics_training['0'], tss_vector = optimize_threshold(
            probability_prediction, Y_training, 'tss')
        '''
        self.threshold, self.metrics_training['0'], dummy = optimize_threshold(
            probability_prediction, Y_training, self.classification)
        self.metrics_training["feature importance"] = self.estimator.coef_.tolist(
        )
        '''
        ## PLOT
        fig, ax = plt.subplots()
        ax.plot( tss_vector, linewidth=2.5, linestyle="-", label="tss")
        ax.plot( hss_vector, linewidth=2.5, linestyle="-", label="hss")
        ax.set_xlabel('threshold', fontsize=20)
        # ax.set_ylabel(classification, fontsize=20)
        plt.legend(loc='upper right', frameon=False)
        ax.grid(True)
        fig.tight_layout()
        # plt.show()
        plt.savefig('HybridLasso_threshold.png', bbox_inches='tight')
        '''
    def predict(self, X_testing, Y_testing_tupla=None):
        Y_testing_prediction = self.estimator.predict(X_testing)
        Y_testing_classification = Y_testing_prediction > self.threshold
        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            self.metrics_testing['0'] = metrics_classification(
                Y_testing>0, Y_testing_classification)
        return Y_testing_classification

    def get_params(self, deep=True):
        return self.estimator.get_params(deep)

    def set_params(self, **params):
        return self.estimator.set_params(**params)

class HybridLassoL2:
    """
    FLARECAST wrapper for the scikit-learn logistic regression
    """

    def __init__(self, **parameters):
        global c_type
        try:
            HybridLasso_parameters = parameters.pop('HybridLassoL2')
            c_type = HybridLasso_parameters.pop('classification')
        except KeyError:
            HybridLasso_parameters = parameters

        self.classification = list(c_type.keys())[
            list(c_type.values()).index(True)]

        self.estimator = LassoCV(**HybridLasso_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        self.estimator.fit(X_training, Y_training)
        probability_prediction2 = self.estimator.predict(X_training)
        XL2 = X_training[:,self.estimator.coef_>0]
        clf = RidgeCV()
        clf.fit(XL2,Y_training)
        probability_prediction = clf.predict(XL2)

        # threshold estimate
        #self.threshold, self.metrics_training['0'], hss_vector = optimize_threshold(
        #    probability_prediction, Y_training, 'hss')
        self.threshold2, self.metrics_training['0'], tss_vector = optimize_threshold(
            probability_prediction2, Y_training, self.classification)
        self.threshold, self.metrics_training['0'], dummy = optimize_threshold(
            probability_prediction, Y_training, self.classification)
        self.metrics_training["feature importance"] = self.estimator.coef_.tolist(
        )
        '''
        ## PLOT
        fig, ax = plt.subplots()
        ax.plot( tss_vector, linewidth=2.5, linestyle="-", label="tss")
        ax.plot( hss_vector, linewidth=2.5, linestyle="-", label="hss")
        ax.set_xlabel('threshold', fontsize=20)
        # ax.set_ylabel(classification, fontsize=20)
        plt.legend(loc='upper right', frameon=False)
        ax.grid(True)
        fig.tight_layout()
        # plt.show()
        plt.savefig('HybridLasso_threshold.png', bbox_inches='tight')
        '''
    def predict(self, X_testing, Y_testing=None):
        Y_testing_prediction = self.estimator.predict(X_testing)
        Y_testing_classification = Y_testing_prediction > self.threshold
        if Y_testing is not None:
            self.metrics_testing['0'] = metrics_classification(
                Y_testing>0, Y_testing_classification)
        return Y_testing_classification

    def get_params(self, deep=True):
        return self.estimator.get_params(deep)

    def set_params(self, **params):
        return self.estimator.set_params(**params)



class SVR_CV:
    """
    FLARECAST wrapper for the scikit-learn Support Vector Regression
    """

    def __init__(self, **parameters):
        SVR_parameters = parameters.pop('SVR')
        self.opt_parameter = SVR_parameters.pop('dfl_optimization')
        self.estimator = SVR(**SVR_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}

    def predict(self, X_testing, Y_testing_tupla=None):
        Y_testing_prediction = self.estimator.predict(X_testing)
        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            self.metrics_testing['0'] = metrics_regression(
                Y_testing, Y_testing_prediction)
        return Y_testing_prediction

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        if self.opt_parameter:
            H_DFL.Hyper_DFL(self.estimator, X_training, Y_training)
        else:
            CV = 5
            best_error = 10**10
            best_C = 0
            # best_r2=0
            for exp_c in range(-5, 10):
                self.estimator.C = 2**exp_c
                kf = KFold(n_splits=CV)
                kf.get_n_splits(X_training)
                error_fold = []
                # r2_fold=[]
                for train_index, validation_index in kf.split(X_training):
                    X_train, X_validation = X_training[train_index], X_training[validation_index]
                    Y_train, Y_validation = Y_training[train_index], Y_training[validation_index]
                    self.estimator.fit(X_train, Y_train)
                    F_validation = self.estimator.predict(X_validation)
                    error = mean_squared_error(
                        Y_validation, F_validation, multioutput='raw_values')
                    error_fold.append(error)
                # r2_fold.append(r2_score(Y_validation, F_validation, multioutput='raw_values'))

                error_tot = numpy.mean(error_fold)
                if error_tot < best_error:
                    best_error = error_tot
                    best_C = 2**exp_c
                # best_r2=numpy.mean(r2_fold)

            # Final Training
            self.estimator.C = best_C
            self.estimator.fit(X_training, Y_training)
            self.metrics_training['0'] = metrics_regression(
                Y_training, self.estimator.predict(X_training))


class SVC_CV:
    """
    FLARECAST wrapper for the scikit-learn Support Vector Classification
    """

    def __init__(self, **parameters):
        #        SVC_parameters = parameters['SVC_CV']
        #        self.estimator = SVC(**SVC_parameters)
        #        self.metrics_training = {}
        #        self.metrics_testing = {}
        #

        global c_type
        try:
            SVC_parameters = parameters.pop('SVC')
            c_type = SVC_parameters.pop('classification')
        except KeyError:
            SVC_parameters = parameters

        self.classification = list(c_type.keys())[
            list(c_type.values()).index(True)]

        self.estimator = SVC(**SVC_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}

    def predict(self, X_testing, Y_testing_tupla=None):
        Y_testing_prediction = self.estimator.predict(X_testing)
        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            self.metrics_testing['0'] = metrics_classification(
                Y_testing, Y_testing_prediction)

        return Y_testing_prediction

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        
        CV = 5
        best_score = 0
        best_C = 0
        
        if Y_training_pre_std.sum()>10:
            for exp_c in range(-5, 10):
                self.estimator.C = 2**exp_c
                kf = KFold(n_splits=CV)
                kf.get_n_splits(X_training)
                score_fold = []
                for train_index, validation_index in kf.split(X_training):
                    X_train, X_validation = X_training[train_index], X_training[validation_index]
                    Y_train, Y_validation = Y_training[train_index], Y_training[validation_index]
                    self.estimator.fit(X_train, Y_train)
                    F_validation = self.estimator.predict(X_validation)
                    if self.classification == 'tss' or self.classification == 'hss':
                        score = metrics_classification(Y_validation, F_validation, print_skills=False)
                        score_fold.append(score[self.classification])
                    else:
                        score = accuracy_score(Y_validation, F_validation)
                        score_fold.append(score)
    
                score_tot = numpy.mean(score_fold)
                if score_tot > best_score:
                    best_score = score_tot
                    best_C = 2**exp_c

        # Final Training
        if best_C<=0:
            self.estimator.C = 1
        else:
            self.estimator.C = best_C

        self.estimator.fit(X_training, Y_training)
        self.prediction_training = {
            'source_data': Y_training,
            'prediction_data': self.estimator.predict(X_training)}
        self.metrics_training['0'] = metrics_classification(
            Y_training, self.estimator.predict(X_training))


class MLPClassifier_HM:
    """
    FLARECAST wrapper for the scikit-learn MLPClassifier
    """

    def __init__(self, **parameters):
        global c_type

        try:
            MLPClassifier_parameters = parameters.pop('MLPClassifier_HM')
            c_type = MLPClassifier_parameters.pop('classification')
        except KeyError:
            MLPClassifier_parameters = parameters

        self.classification = list(c_type.keys())[
            list(c_type.values()).index(True)]

        self.estimator = MLPClassifier(**MLPClassifier_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        #Xn, self._mean_, self._std_ = training_set_standardization(X)
        self.estimator.fit(X_training, Y_training)
        # predicts the probability of a flare
        probability_prediction = self.estimator.predict_proba(X_training)[:, 1]

        # threshold estimate
        # ['0'] to be uniform with multitask
        self.threshold, self.metrics_training['0'], dummy  = optimize_threshold(
            probability_prediction, Y_training, self.classification)

    def predict(self, X_testing, Y_testing_tupla=None):
        self.probability = self.estimator.predict_proba(X_testing)[:, 1]
        Y_testing_classification = self.probability > self.threshold
        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            self.metrics_testing['0'] = metrics_classification(
                Y_testing, Y_testing_classification)
        return Y_testing_classification


class MLPRegressor_HM:
    """
    FLARECAST wrapper for the scikit-learn MLPRegressor
    """

    def __init__(self, **parameters):
        MLPRegressor_parameters = parameters.pop('MLPRegressor_HM')
        self.opt_parameter = MLPRegressor_parameters.pop('dfl_optimization')
        self.estimator = MLPRegressor(**MLPRegressor_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}
        self.threshold = {}

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        if self.opt_parameter:
            H_DFL.Hyper_DFL(self.estimator, X_training, Y_training)
        else:
            self.estimator.fit(X_training, Y_training)
        Y_predicted = self.estimator.predict(X_training)
        self.metrics_training = metrics_regression(Y_training, Y_predicted)
        if Y_training.shape[1] == 1:
            self.threshold['0'], self.metrics_training['0'], xss_vector = optimize_threshold(
                Y_predicted, Y_training, 'tss')
        else:
            for i in range(Y_training.shape[1]):
                # threshold estimate
                self.threshold[str(i)], self.metrics_training[str(i)], xss_vector = optimize_threshold(
                    Y_predicted[:,i], Y_training[:,i], 'tss')

    def predict(self, X_testing, Y_testing_tupla=None):
        Y_testing_prediction = self.estimator.predict(X_testing)
        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            self.metrics_testing = metrics_regression(
                Y_testing, Y_testing_prediction)
            if Y_testing.shape[1] == 1:
                self.threshold, self.metrics_training['0'], xss_vector = optimize_threshold(
                    Y_testing_prediction, Y_testing, 'tss')
            else:
                for i in range(Y_testing.shape[1]):
                    # Y_testing_classification: if a predicted value larger than a fixed threshold -> true
                    Y_testing_classification = Y_testing_prediction[:,i] > self.threshold[str(i)]
                    # Y_testing_classification: if a true value is larger than a fixed threshold -> it happened
                    Y_testing_true = Y_testing[:,i] > self.threshold[str(i)]
                    if Y_testing is not None:
                        print(self.threshold[str(i)])
                        self.metrics_testing[str(i)] = metrics_classification(
                            Y_testing_true, Y_testing_classification)
        return Y_testing_prediction


class KMeans_HM:
    """
    FLARECAST wrapper for the home-made KMeans
    """

    def __init__(self, **parameters):
        KMeans_HM_parameters = parameters.pop('KMeansHM')
        c_type = KMeans_HM_parameters.pop('classification')
        self.classification = list(c_type.keys())[
            list(c_type.values()).index(True)]
        self.estimator = KMeans(**KMeans_HM_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        self.estimator.fit(X_training, Y_training)
        self.estimator.labels_, switch = check_prediction_clustering(
            Y_training, self.estimator.labels_)
        if switch == 1:
            cluster_centers = self.estimator.cluster_centers_.copy()
            self.estimator.cluster_centers_[0] = cluster_centers[1]
            self.estimator.cluster_centers_[1] = cluster_centers[0]
        self.metrics_training['0'] = metrics_classification(
            Y_training, self.estimator.labels_)

    def predict(self, X_testing, Y_testing_tupla=None):
        Y_testing_prediction = self.estimator.predict(X_testing)
        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            #Y_testing_prediction, switch = check_prediction_clustering(
            #    Y_testing, Y_testing_prediction)
            self.metrics_testing['0'] = metrics_classification(
                Y_testing, Y_testing_prediction)
        return Y_testing_prediction


class FKMeans_HM:
    """
    FLARECAST wrapper for the home-made FKMeans
    """

    def __init__(self, **parameters):
        FKMeans_HM_parameters = parameters.pop('FKMeansHM')
        c_type = FKMeans_HM_parameters.pop('classification')
        self.classification = list(c_type.keys())[
            list(c_type.values()).index(True)]
        self.estimator = FKMeans(**FKMeans_HM_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        self.estimator.fit(X_training, Y_training)
        self.estimator.labels_, switch = check_prediction_clustering(
            Y_training, self.estimator.labels_)
        if switch == 1:
            cluster_centers = self.estimator.cluster_centers_.copy()
            self.estimator.cluster_centers_[0] = cluster_centers[1]
            self.estimator.cluster_centers_[1] = cluster_centers[0]
        self.metrics_training['0'] = metrics_classification(
            Y_training, self.estimator.labels_)

    def predict(self, X_testing, Y_testing_tupla=None):
        Y_testing_prediction = self.estimator.predict(X_testing)
        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            #Y_testing_prediction, switch = check_prediction_clustering(
            #    Y_testing, Y_testing_prediction)
            self.metrics_testing['0'] = metrics_classification(
                Y_testing, Y_testing_prediction)
        return Y_testing_prediction


class PKMeans_HM:
    """
    FLARECAST wrapper for the home-made PKMeans
    """

    def __init__(self, **parameters):
        PKMeans_HM_parameters = parameters.pop('PKMeansHM')
        c_type = PKMeans_HM_parameters.pop('classification')
        self.classification = list(c_type.keys())[
            list(c_type.values()).index(True)]
        self.estimator = PKMeans(**PKMeans_HM_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        self.estimator.fit(X_training)
        self.estimator.labels_, switch = check_prediction_clustering(
            Y_training, self.estimator.labels_)
        if switch == 1:
            cluster_centers = self.estimator.cluster_centers_.copy()
            self.estimator.cluster_centers_[0] = cluster_centers[1]
            self.estimator.cluster_centers_[1] = cluster_centers[0]
        self.metrics_training['0'] = metrics_classification(
            Y_training, self.estimator.labels_)


    def predict(self, X_testing, Y_testing_tupla=None):
        Y_testing_prediction = self.estimator.predict(X_testing)
        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            #Y_testing_prediction, switch = check_prediction_clustering(
            #    Y_testing, Y_testing_prediction)
            self.metrics_testing['0'] = metrics_classification(
                Y_testing, Y_testing_prediction)
        return Y_testing_prediction


class SimAnnKMeans_HM:
    """
    FLARECAST wrapper for the Simulated annealing KMeans
    """

    def __init__(self, **parameters):
        SimAnnKMeans_HM_parameters = parameters.pop('SimAnnKMeansHM')
        c_type = SimAnnKMeans_HM_parameters.pop('classification')
        self.classification = list(c_type.keys())[
            list(c_type.values()).index(True)]
        self.estimator = SimAnnKMeans(**SimAnnKMeans_HM_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        self.estimator.fit(X_training, Y_training)
        self.estimator.labels_, switch = check_prediction_clustering(
            Y_training, self.estimator.labels_)
        if switch == 1:
            cluster_centers = self.estimator.cluster_centers_.copy()
            self.estimator.cluster_centers_[0] = cluster_centers[1]
            self.estimator.cluster_centers_[1] = cluster_centers[0]
        self.metrics_training['0'] = metrics_classification(
            Y_training, self.estimator.labels_)

    def predict(self, X_testing, Y_testing_tupla=None):
        Y_testing_prediction = self.estimator.predict(X_testing)
        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            #Y_testing_prediction, switch = check_prediction_clustering(
            #    Y_testing, Y_testing_prediction)
            self.metrics_testing['0'] = metrics_classification(
                Y_testing, Y_testing_prediction)
        return Y_testing_prediction


class SimAnnFKMeans_HM:
    """
    FLARECAST wrapper for the Simulated annealing FKMeans
    """

    def __init__(self, **parameters):
        SimAnnFKMeans_HM_parameters = parameters.pop('SimAnnFKMeansHM')
        c_type = SimAnnFKMeans_HM_parameters.pop('classification')
        self.classification = list(c_type.keys())[
            list(c_type.values()).index(True)]
        self.estimator = SimAnnFKMeans(**SimAnnFKMeans_HM_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        self.estimator.fit(X_training, Y_training)
        self.estimator.labels_, switch = check_prediction_clustering(
            Y_training, self.estimator.labels_)
        if switch == 1:
            cluster_centers = self.estimator.cluster_centers_.copy()
            self.estimator.cluster_centers_[0] = cluster_centers[1]
            self.estimator.cluster_centers_[1] = cluster_centers[0]
        self.metrics_training['0'] = metrics_classification(
            Y_training, self.estimator.labels_)

    def predict(self, X_testing, Y_testing_tupla=None):
        Y_testing_prediction = self.estimator.predict(X_testing)
        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            #Y_testing_prediction, switch = check_prediction_clustering(
            #    Y_testing, Y_testing_prediction)
            self.metrics_testing['0'] = metrics_classification(
                Y_testing, Y_testing_prediction)
        return Y_testing_prediction


class AdaptiveLasso_CV:
    """
    FLARECAST inplementation of the Adaptive Lasso
    """

    def __init__(self, **parameters):
        Lasso_parameters = parameters.pop('Lasso')
        LassoCV_parameters = parameters.pop('LassoCV')
        self.estimator = Lasso(**Lasso_parameters)
        self.estimator_CV = LassoCV(**LassoCV_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        # train the algorithm with a given training set (X,Y)
        self.estimator_CV.fit(X_training, Y_training)
        # extract the cross validated regularization parameter value
        lambda_opt = self.estimator_CV.alpha_
        # extract the corresponding solution and compute the adaptive weights
        beta = self.estimator_CV.coef_
        n = Y_training.shape[0]
        gamma = 2.1 # 2
        w = (numpy.absolute(beta) + 1. / n)**(-gamma)
        # create the Lasso cross validated object
        self.estimator.alpha = lambda_opt
        # train the algorithm with a rescaled X matrix
        self.estimator.fit(X_training / w, Y_training)
        # rescale the predictor weights
        self.estimator.coef_ = self.estimator.coef_ / w

        # TO DO !!!!!!!!!!!!
        #self.metrics_training = metrics_regression(
        #    Y_training, self.estimator.predict(X_training))
        self.metrics_training['0'] = metrics_regression(
            Y_training, self.estimator_CV.predict(X_training))

    def predict(self, X_testing, Y_testing_tupla=None):
        # TO DO !!!!!!!!!!!! not CV !!!!!!!!!!!!!!!!!!!
        Y_testing_prediction = self.estimator_CV.predict(X_testing)
        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            self.metrics_testing['0'] = metrics_regression(
                Y_testing, Y_testing_prediction)
        return Y_testing_prediction


class MultiTaskLasso_CV:
    """
    FLARECAST wrapper for the the scikit-learn Multi Task Lasso
    """

    def __init__(self, **parameters):
        MultiTaskLassoCV_parameters = parameters.pop('MultiTaskLassoCV')
        self.estimator = MultiTaskLassoCV(**MultiTaskLassoCV_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}
        self.threshold = {}

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        # train the algorithm with a given training set (X,Y)
        self.estimator.fit(X_training, Y_training)
        # probability_prediction = self.estimator.predict(X_training)
        Y_predicted = self.estimator.predict(X_training)
        self.metrics_training = metrics_regression(
            Y_training, Y_predicted)
        self.metrics_training["feature importance"] = self.estimator.coef_.tolist()
        # for each label create the classification skill score (0 vs > 0)
        for i in range(Y_training.shape[1]):
            # threshold estimate
            self.threshold[str(i)], self.metrics_training[str(i)], tss_vector = optimize_threshold(
            Y_predicted[:,i], Y_training_pre_std[:,i], 'tss')
            self.threshold[str(i)], self.metrics_training[str(i)], hss_vector = optimize_threshold(
            Y_predicted[:,i], Y_training_pre_std[:,i], 'hss')
            '''
            fig, ax = plt.subplots()
            ax.plot( tss_vector, linewidth=2.5, linestyle="-", label="tss")
            ax.plot( hss_vector, linewidth=2.5, linestyle="-", label="hss")
            ax.set_xlabel('threshold', fontsize=20)
            # ax.set_ylabel(classification, fontsize=20)
            plt.legend(loc='upper right', frameon=False)
            ax.grid(True)
            fig.tight_layout()
            # plt.show()
            plt.savefig('MTL_threshold_task_'+str(i)+'.png', bbox_inches='tight')
            '''

    def predict(self, X_testing, Y_testing_tupla=None):
        Y_testing_prediction = self.estimator.predict(X_testing)


        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            Y_testing_pre_std = Y_testing_tupla[1]
            self.metrics_testing = metrics_regression(
                Y_testing, Y_testing_prediction)

            for i in range(Y_testing.shape[1]):
                # Y_testing_classification: if a predicted value larger than a fixed threshold -> true
                Y_testing_classification = Y_testing_prediction[:,i] > self.threshold[str(i)]
                # Y_testing_classification: if a true value is larger than a fixed threshold -> it happened
                Y_testing_true = Y_testing_pre_std[:,i] > 0.
                if Y_testing is not None:
                    print(self.threshold[str(i)])
                    self.metrics_testing[str(i)] = metrics_classification(
                        Y_testing_true, Y_testing_classification)

        return Y_testing_prediction


class AdaptiveMultiTaskLasso_CV:
    """
    FLARECAST wrapper for the the scikit-learn Adaptive Multi Task Lasso
    """

    def __init__(self, **parameters):
        MultiTaskLassoCV_parameters = parameters.pop('MultiTaskLassoCV')
        self.estimator = MultiTaskLasso()
        self.estimatorCV = MultiTaskLassoCV(**MultiTaskLassoCV_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}
        self.threshold = {}

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        # train the algorithm with a given training set (X,Y)
        self.estimatorCV.fit(X_training, Y_training)
        # extract the cross validated regularization parameter value
        lambda_opt = self.estimatorCV.alpha_
        # extract the corresponding solution and compute the adaptive weights
        beta = self.estimatorCV.coef_
        n = Y_training.shape[0]
        # adaptive weights calculation
        gamma = 3
        w = (numpy.sqrt(sum(beta**2, 1)) + 1 / n)**(-gamma)
        # create the Multi Task Lasso cross validated object
        self.estimator.alpha = lambda_opt
        # train the algorithm with a rescaled X matrix (Zou's style 2006)
        self.estimator.fit(X_training / w, Y_training)
        # rescale the predictor weights
        self.estimator.coef_ = self.estimator.coef_ / numpy.transpose(w)
        self.metrics_training['0'] = metrics_regression(
            Y_training, self.estimator.predict(X_training))
        self.metrics_training["feature importance"] = self.estimator.coef_.transpose(
        ).tolist()
        Y_predicted = self.estimator.predict(X_training)
        for i in range(Y_training.shape[1]):
            # threshold estimate
            self.threshold[str(i)], self.metrics_training[str(i)], tss_vector = optimize_threshold(
            Y_predicted[:,i], Y_training_pre_std[:,i], 'tss')

    def predict(self, X_testing, Y_testing_tupla=None):
        Y_testing_prediction = self.estimator.predict(X_testing)
        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            self.metrics_testing['0'] = metrics_regression(
                Y_testing, Y_testing_prediction)
        return Y_testing_prediction


class MultiTaskPoissonLasso_CV:
    """
    FLARECAST wrapper for the the scikit-learn Multi Task Lasso
    """

    def __init__(self, **parameters):
        MultiTaskLassoCV_parameters = parameters.pop('MultiTaskLassoCV')
        self.estimator = MultiTaskLassoCV(**MultiTaskLassoCV_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        # train the algorithm with a given training set (X,Y)
        if numpy.any(Y_training < 0):
            print("Warning: Some values in Y training are negative")
        Y_training = numpy.where(Y_training > 0, Y_training, 0)
        mu = 1 / numpy.sqrt(Y_training + 1)
        self.estimator.fit(X_training, Y_training, mu=mu)
        self.metrics_training['0'] = metrics_regression(
            Y_training, self.estimator.predict(X_training))

    def predict(self, X_testing, Y_testing_tupla=None):
        Y_testing_prediction = self.estimator.predict(X_testing)
        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            self.metrics_testing['0'] = metrics_regression(
                Y_testing, Y_testing_prediction)
        return Y_testing_prediction


class AdaptiveMultiTaskPoissonLasso_CV:
    """
    FLARECAST wrapper for the the scikit-learn Adaptive Multi Task Lasso
    """

    def __init__(self, **parameters):
        MultiTaskLassoCV_parameters = parameters.pop('MultiTaskLassoCV')
        self.estimator = MultiTaskLasso()
        self.estimatorCV = MultiTaskLassoCV(**MultiTaskLassoCV_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        # train the algorithm with a given training set (X,Y)

        if numpy.any(Y_training < 0):
            print("Warning: Some values in Y training are negative")
        Y_training = numpy.where(Y_training > 0, Y_training, 0)
        mu = 1 / numpy.sqrt(Y_training + 1)
        self.estimatorCV.fit(X_training, Y_training, mu=mu.tolist())
        # extract the cross validated regularization parameter value
        lambda_opt = self.estimatorCV.alpha_
        # extract the corresponding solution and compute the adaptive weights
        beta = self.estimatorCV.coef_
        n = Y_training.shape[0]
        # adaptive weights calculation
        gamma = 3
        w = (numpy.sqrt(sum(beta**2, 1)) + 1 / n)**(-gamma)
        # create the Multi Task Lasso cross validated object
        self.estimator.alpha = lambda_opt
        # train the algorithm with a rescaled X matrix (Zou's style 2006)
        self.estimator.fit(X_training / w, Y_training, mu=mu)
        # rescale the predictor weights
        self.estimator.coef_ = self.estimator.coef_ / numpy.transpose(w)
        self.metrics_training['0'] = metrics_regression(
            Y_training, self.estimator.predict(X_training))
        self.metrics_training["feature importance"] = self.estimator.coef_.transpose(
        ).tolist()

    def predict(self, X_testing, Y_testing_tupla=None):
        Y_testing_prediction = self.estimator.predict(X_testing)
        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            self.metrics_testing['0'] = metrics_regression(
                Y_testing, Y_testing_prediction)
        return Y_testing_prediction


class RandomForest:
    """
    FLARECAST wrapper for the the scikit-learn Random Forest Classification
    """

    def __init__(self, **parameters):

        global c_type
        try:
            RandomForest_parameters = parameters.pop('RandomForest')
            c_type = RandomForest_parameters.pop('classification')
        except KeyError:
            RandomForest_parameters = parameters

        self.classification = list(c_type.keys())[
            list(c_type.values()).index(True)]
        self.estimator = RandomForestClassifier(**RandomForest_parameters)
        self.metrics_training = {}
        self.metrics_testing = {}

    def fit(self, X_training, Y_training_tupla):
        Y_training = Y_training_tupla[0]
        Y_training_pre_std = Y_training_tupla[1]
        #Xn, self._mean_, self._std_ = training_set_standardization(X)
        self.estimator.fit(X_training, Y_training)
        # predicts the probability of a flare
        probability_prediction = self.estimator.predict_proba(X_training)[:, 1]

        # threshold estimate
        # ['0'] to be uniform with multitask
        self.threshold, self.metrics_training['0'], dummy  = optimize_threshold(
            probability_prediction, Y_training, self.classification)
        self.metrics_training["feature importance"] = self.estimator.feature_importances_.tolist()[
            0]

    def predict(self, X_testing, Y_testing_tupla=None):
        self.probability = self.estimator.predict_proba(X_testing)[:, 1]
        Y_testing_classification = self.probability > self.threshold
        if Y_testing_tupla is not None:
            Y_testing = Y_testing_tupla[0]
            self.metrics_testing['0'] = metrics_classification(
                Y_testing, Y_testing_classification)
        return Y_testing_classification

    def get_params(self, deep=True):
        return self.estimator.get_params(deep)

    def set_params(self, **params):
        return self.estimator.set_params(**params)
