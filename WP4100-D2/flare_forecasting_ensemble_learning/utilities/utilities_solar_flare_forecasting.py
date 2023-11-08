#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:54:37 2022

@author: sabry
"""

import numpy as np
from keras.models import load_model
import pandas



from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix


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

    # acc = (a + d) / (a + b + c + d)
    # tpr = a / (a + b)
    # tnr = d / (d + c)
    # wtpr = a / (a + b) * (a + c) / (a + b + c + d) + d / (c + d) * (b + d) / (a + b + c + d)
    # pacc = a / (a + c)
    # nacc = d / (b + d)
    # wacc = a / (a + c) * (a + c) / (a + b + c + d) + d / (b + d) * (b + d) / (a + b + c + d)

    # if the cm has a row or a column equal to 0, we have bad tss
    if TP + FN == 0 or TN + FP == 0 or TP + FP == 0 or TN + FN == 0:
        tss = 0

    # , pod, acc, hss, tss, fnfp, tpr, tnr, pacc, nacc, wacc, wtpr)
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

    return {"mse": mse.tolist(), "r2": r2.tolist()}  


def optimize_threshold(probability_prediction, Y_training, classification):

    if classification == 'tss' or classification == 'hss':
        n_samples = 100#
        step = 1. / n_samples
        # to be set in the learning phase: algo.threshold
        # TODO after the learning step, compute the thresholding parameter
        # with tss or hss.
        xss = -1.
        xss_threshold = 0
        Y_best_predicted = np.zeros((Y_training.shape))
        xss_vector = np.zeros(n_samples)
        a = probability_prediction.max()
        b = probability_prediction.min()
        for threshold in range(1, n_samples):
            xss_threshold = step * threshold * np.abs(a - b) + b
            Y_predicted = probability_prediction > xss_threshold
            res = metrics_classification(Y_training > 0, Y_predicted, print_skills=False)
            xss_vector[threshold] = res[classification]
            if res[classification] > xss:
                xss = res[classification]
                Y_best_predicted = Y_predicted
                best_xss_threshold = xss_threshold

    metrics_training = metrics_classification(Y_training > 0, Y_best_predicted)  



    return best_xss_threshold, metrics_training, xss_vector


def training_set_standardization(X_training):
    # normalization / standardization
    mean_ = X_training.sum(axis=0) / X_training.shape[0]

    std_ = np.sqrt(
        (((X_training - mean_) ** 2.).sum(axis=0) / X_training.shape[0]))

    Xn_training = div0((X_training - mean_), std_)

    return Xn_training, mean_, std_

def testing_set_standardization(X_testing, mean_, std_):
    Xn_testing = div0((X_testing - mean_), std_)

    return Xn_testing

def div0(a, b):
    """
    ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]

    Parameters
    ----------
    a
    b

    Returns
    -------

    c

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c.tolist())] = 0  # -inf inf NaN
    return c


def compute_cm_tss_threshold(y, pred,threshold):
    # Compute confusion matrix and TSS given y_true and y_pred in categorical form
    pred_threshold = pred > threshold
    cm = confusion_matrix(y,pred_threshold)
    if cm.shape[0] == 1 and sum(y_true) == 0:
        a = 0.
        d = float(cm[0, 0])
        b = 0.
        c = 0.
    elif cm.shape[0] == 1 and sum(y_true) == y_true.shape[0]:
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
    
    if TP+FP+FN==0:
        CSI = 0
    else:
        CSI = TP/(TP+FP+FN)

    
    return cm, tss, hss, CSI
        

def compute_cm_tss(y, pred):
    # Compute confusion matrix and TSS given y_true and y_pred in categorical form
    cm = confusion_matrix(y,pred)
    if cm.shape[0] == 1 and sum(y_true) == 0:
        a = 0.
        d = float(cm[0, 0])
        b = 0.
        c = 0.
    elif cm.shape[0] == 1 and sum(y_true) == y_true.shape[0]:
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
    
    if TP+FP+FN==0:
        CSI = 0
    else:
        CSI = TP/(TP+FP+FN)

    
    return cm, tss, hss, CSI

def compute_weight_cm_tss(y, pred):
    # Compute confusion matrix and TSS given y_true and y_pred in categorical form
    #pred_threshold = pred > threshold
    TN,FP,FN,TP = weighted_confusion_matrix(y,pred)

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
    
    if TP+FP+FN==0:
        CSI=0
    else:
        CSI = TP/(TP+FP+FN)
    
    weighted_cm = np.zeros((2,2))
    weighted_cm[0,0]=TN
    weighted_cm[0,1]=FP
    weighted_cm[1,0]=FN
    weighted_cm[1,1]=TP
    
    return weighted_cm, tss, hss, CSI

def compute_weight_cm_tss_harp(y_real, y_pred,panel_label):
    harp = [panel_label.index[idx][0] for idx in range(panel_label.shape[0])]
    
    harp=np.unique(harp)
    #print(harp)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for ar in harp:
        #print(ar)
        idx_ar=[i for i in range(panel_label.shape[0]) if panel_label.index[i][0]==ar]
        y_real_ar = y_real[idx_ar]
        y_pred_ar = y_pred[idx_ar]
        #print('ar=',ar)
        TN_ar,FP_ar,FN_ar,TP_ar = weighted_confusion_matrix(y_real_ar, y_pred_ar)
        TP = TP + TP_ar
        FP = FP + FP_ar
        FN = FN + FN_ar
        TN = TN + TN_ar

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
        
    if TP+FP+FN==0:
        csi = 0
    else:
        csi = TP/(TP+FP+FN)
    
    
    weighted_cm = np.zeros((2,2))
    weighted_cm[0,0]=TN
    weighted_cm[0,1]=FP
    weighted_cm[1,0]=FN
    weighted_cm[1,1]=TP
    #, pod, acc, hss, tss, fnfp, tpr, tnr, pacc, nacc, wacc, wtpr)
    return weighted_cm, tss, hss, csi

def optimize_time_weighted_tss_solar_flares(probability_prediction, Y_training, panel_harp):
    '''
    harp = [panel_label.index[idx][0] for idx in range(panel_label.shape[0])]   
    harp=np.unique(harp)
    panel_harp = {}
    panel_harp['ar'] = {}
    panel_harp['ar']['idx'] = {}
    ar_list=[]
    idx_list=[]
    for ar in harp:
        ar_list.append(ar)
        idx_ar=[i for i in range(panel_label.shape[0]) if panel_train.index[i][0]==ar]
        idx_list.append(np.array(idx_ar))
    
    panel_harp['ar']=ar_list
    panel_harp['idx']=idx_list
    df_harp = pandas.DataFrame (panel_harp, columns = ['ar','idx'])
    print('end list!')
    '''
    n_samples = 100#
    step = 1. / n_samples
    
    xss = -1.
    xss_threshold = 0
    Y_best_predicted = np.zeros((Y_training.shape))
    tss_vector = np.zeros(n_samples)
    hss_vector = np.zeros(n_samples)
    xss_threshold_vector = np.zeros(n_samples)
    a = probability_prediction.max()
    b = probability_prediction.min()
    print('A:',a)
    print('B:',b)
    if a>b:
        for threshold in range(1, n_samples):
            xss_threshold = step * threshold * np.abs(a - b) + b
            xss_threshold_vector[threshold] = xss_threshold
            Y_predicted = probability_prediction > xss_threshold
            res = metrics_classification_weight_solar_flares_harp_panel(Y_training > 0, Y_predicted, panel_harp,print_skills=False)#panel_label,print_skills=False)
            tss_vector[threshold] = res['tss']
            hss_vector[threshold] = res['hss']
        #print(tss_vector)
        #print(hss_vector)
        max_tss=np.max(tss_vector)
        eps=1e-5
        if max_tss==0:
            max_tss=max_tss+eps
        print('MAX TSS:',max_tss)
    
        #best TSS
        idx_best_tss = np.where(tss_vector==np.max(tss_vector))  
        print('idx best tss=',idx_best_tss)
        best_xss_threshold_tss = xss_threshold_vector[idx_best_tss]
        if len(best_xss_threshold_tss)>1:
            best_xss_threshold_tss = best_xss_threshold_tss[0]
            Y_best_predicted_tss = probability_prediction > best_xss_threshold_tss
        else:
            Y_best_predicted_tss = probability_prediction > best_xss_threshold_tss
        print('best TSS')
        metrics_training_tss = metrics_classification_weight_solar_flares_harp_panel(Y_training > 0, Y_best_predicted_tss,panel_harp)
    else:
        print('Exit! A==B')
        best_xss_threshold_tss=-1000
        max_tss=-1000

    return best_xss_threshold_tss, max_tss



def metrics_classification_weight_solar_flares_harp_panel(y_real, y_pred, panel_harp,print_skills=True):

    cm, far, pod, acc, hss, tss, fnfp, csi = classification_skills_weight_solar_flares_harp_panel(y_real, y_pred,panel_harp)

    if print_skills:
        print ('confusion matrix')
        print (cm)
        print ('false alarm ratio       \t', far)
        print ('probability of detection\t', pod)
        print ('accuracy                \t', acc)
        print ('hss                     \t', hss)
        print ('tss                     \t', tss)
        print ('balance                 \t', fnfp)
        print ('csi                 \t', csi)

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
        "balance label": balance_label,
        "csi": csi}

def classification_skills_weight_solar_flares_harp_panel(y_real, y_pred,panel_harp):
    #harp = [panel_label.index[idx][0] for idx in range(panel_label.shape[0])]

    harp=panel_harp['ar']#np.unique(harp)
    harp=np.array(harp)
    #print('harp:',harp)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for k in range(len(harp)): #harp:
        #print('ar:',harp[k])
        #idx_ar=[i for i in range(len(panel_harp)) if panel_harp['ar'][i]==ar]
        #print('idx_ar',idx_ar)
        idx = panel_harp['idx'][k]#.values[0]
        #print('idx',idx)
        y_real_ar = y_real[idx]
        y_pred_ar = y_pred[idx]
        #print('shape y_real_ar:',y_real_ar.shape)
        #print('shape y_real_ar:',y_pred_ar.shape)
        TN_ar,FP_ar,FN_ar,TP_ar = weighted_confusion_matrix(y_real_ar, y_pred_ar)
        TP = TP + TP_ar
        FP = FP + FP_ar
        FN = FN + FN_ar
        TN = TN + TN_ar

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
        
    if TP+FP+FN==0:
        csi = 0
    else:
        csi = TP/(TP+FP+FN)
    
    
    weighted_cm = np.zeros((2,2))
    weighted_cm[0,0]=TN
    weighted_cm[0,1]=FP
    weighted_cm[1,0]=FN
    weighted_cm[1,1]=TP
    #, pod, acc, hss, tss, fnfp, tpr, tnr, pacc, nacc, wacc, wtpr)
    return weighted_cm, far, pod, acc, hss, tss, fnfp, csi

def classification_skills_weight_solar_flares(y_real, y_pred,panel_label):
    harp = [panel_label.index[idx][0] for idx in range(panel_label.shape[0])]

    harp=np.unique(harp)
    #print('harp:',harp)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for ar in harp:
        #print('ar:',ar)
        idx_ar=[i for i in range(panel_label.shape[0]) if panel_label.index[i][0]==ar]
        y_real_ar = y_real[idx_ar]
        y_pred_ar = y_pred[idx_ar]
        print('shape y_real_ar:',y_real_ar.shape)
        print('shape y_real_ar:',y_pred_ar.shape)
        TN_ar,FP_ar,FN_ar,TP_ar = weighted_confusion_matrix(y_real_ar, y_pred_ar)
        TP = TP + TP_ar
        FP = FP + FP_ar
        FN = FN + FN_ar
        TN = TN + TN_ar

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
        
    if TP+FP+FN==0:
        csi = 0
    else:
        csi = TP/(TP+FP+FN)
    
    
    weighted_cm = np.zeros((2,2))
    weighted_cm[0,0]=TN
    weighted_cm[0,1]=FP
    weighted_cm[1,0]=FN
    weighted_cm[1,1]=TP
    #, pod, acc, hss, tss, fnfp, tpr, tnr, pacc, nacc, wacc, wtpr)
    return weighted_cm, far, pod, acc, hss, tss, fnfp, csi



def weighted_confusion_matrix(y_true, y_pred):
    TP_values = np.logical_and(np.equal(y_true, True), np.equal(y_pred, True))
    idx_TP=np.where(TP_values==True)
    TP=len(idx_TP[0])
    
    TN_values = np.logical_and(np.equal(y_true, False), np.equal(y_pred, False))
    idx_TN=np.where(TN_values==True)
    TN=len(idx_TN[0])


    FP_values = np.logical_and(np.equal(y_true, False), np.equal(y_pred, True))
    idx_FP=np.where(FP_values==True)
    mask = [1./2.,1./3.,1./4.]
    FP=0
    window_hour=3
    if y_true.shape[0] >=6:
        #tutta la logica sotto
        for t in idx_FP[0]: #range(len(y_true)):
            if t >=  window_hour and t <= len(y_true)- window_hour-1:
                #window -4 +4
                y_true_window = y_true[t- window_hour:t+ window_hour+1]
                y_pred_window = y_pred[t- window_hour:t+ window_hour+1]
        
                if len(np.where(y_true_window==1)[0]) >= 1:
                    count_FP = 1-np.max(mask*y_true[t+1:t+ window_hour+1])
                else:
                    count_FP = 2
            elif t<window_hour:
                y_true_window = y_true[:t+ window_hour+1]
                y_pred_window = y_pred[:t+ window_hour+1]
                if len(np.where(y_true_window==1)[0]) >= 1:
                    count_FP = 1-np.max(mask*y_true[t+1:t+ window_hour+1])
                else:
                    count_FP = 2
            elif t > len(y_true)-window_hour-1:
                y_true_window = y_true[t- window_hour:]
                y_pred_window = y_pred[t- window_hour:]
                if len(np.where(y_true_window==1)[0]) >= 1:
                    if t < len(y_true)-1:
                        count_FP = 1-np.max(mask[0:len(y_true)-t-1]*y_true[t+1:])
                    elif t == len(y_true)-1:
                        count_FP = 1
                
                else:
                    count_FP = 2
            #print('COUNT_FP:',count_FP)
            FP=FP+count_FP
    
    if y_true.shape[0]<6:
        for t in idx_FP[0]: #range(len(y_true
            if t<window_hour:
                if t+window_hour+1<=np.shape(y_true)[0]:
                    y_true_window = y_true[:t+ window_hour+1]
                    y_pred_window = y_pred[:t+ window_hour+1]
                    if len(np.where(y_true_window==1)[0]) >= 1:
                        count_FP = 1-np.max(mask*y_true[t+1:t+ window_hour+1])
                    else:
                        count_FP = 2
                elif t+window_hour+1 >np.shape(y_true)[0]:
                    y_true_window = y_true
                    y_pred_window = y_pred
                    if len(np.where(y_true_window==1)[0]) >= 1:
                        if t < len(y_true)-1:
                            count_FP = 1-np.max(mask[0:len(y_true)-t-1]*y_true[t+1:])
                        elif t == len(y_true)-1:
                            count_FP = 1
                
                    else:
                        count_FP = 2
            elif t > len(y_true)-window_hour-1:
                if t- window_hour>=0:
                    y_true_window = y_true[t- window_hour:]
                    y_pred_window = y_pred[t- window_hour:]
                    if len(np.where(y_true_window==1)[0]) >= 1: #sabry
                        if t < len(y_true)-1:
                            count_FP = 1-np.max(mask[0:len(y_true)-t-1]*y_true[t+1:])
                        elif t == len(y_true)-1:
                            count_FP = 1
                
                    else:
                        count_FP = 2
                elif t- window_hour<0:
                    y_true_window = y_true
                    y_pred_window = y_pred
                    if len(np.where(y_true_window==1)[0]) >= 1:
                        if t < len(y_true)-1:
                            count_FP = 1-np.max(mask[0:len(y_true)-t-1]*y_true[t+1:])
                        elif t == len(y_true)-1:
                            count_FP = 1
                
                    else:
                        count_FP = 2
            #print('COUNT_FP:',count_FP)
            FP=FP+count_FP
                        
        
    FN_values = np.logical_and(np.equal(y_true, True), np.equal(y_pred, False))
    idx_FN=np.where(FN_values==True)
    FN=0
    mask_FN=[1./4.,1./3.,1./2.]
    if y_true.shape[0]>=6:
        #FAI TUTTO SOTTO
        for t in idx_FN[0]: #range(len(y_true)):

            if t >=  window_hour and t <= len(y_true)- window_hour-1:
             #window -4 +4
                y_true_window = y_true[t- window_hour:t+ window_hour+1]
                y_pred_window = y_pred[t- window_hour:t+ window_hour+1]
                if len(np.where(y_pred_window==1)[0]) >= 1:
                    count_FN = 1-np.max(mask_FN*y_pred[t- window_hour:t])
                else:
                    count_FN = 2
            elif t<window_hour:
                y_true_window = y_true[:t+ window_hour+1]
                y_pred_window = y_pred[:t+ window_hour+1]
                if len(np.where(y_pred_window==1)[0]) >= 1:
                    if t > 0:
                        count_FN = 1-np.max(mask_FN[window_hour-t:window_hour]*y_pred[:t])#[t-1:2]*y_pred[:t])
                    elif t ==0:
                        count_FN = 1
                else:
                    count_FN = 2
            elif t > len(y_true)- window_hour-1:
                y_true_window = y_true[t- window_hour:]
                y_pred_window = y_pred[t- window_hour:]
                if len(np.where(y_pred_window==1)[0]) >= 1:
                    count_FN = 1-np.max(mask_FN*y_pred[t- window_hour:t])
                else:
                    count_FN = 2
            #print('size>=6')
            #print('COUNT_FN:',count_FN)
            FN=FN+count_FN
                
                
    if y_true.shape[0]<6:
        for t in idx_FN[0]:
            if t<window_hour:
                if t+window_hour+1<=np.shape(y_true)[0]:
                    y_true_window = y_true[:t+ window_hour+1]
                    y_pred_window = y_pred[:t+ window_hour+1]
                    if len(np.where(y_pred_window==1)[0]) >= 1:
                        if t > 0:
                            count_FN = 1-np.max(mask_FN[window_hour-t:window_hour]*y_pred[:t])#mask_FN[t-1:2]*y_pred[:t])
                        elif t ==0:
                            count_FN = 1
                    else:
                        count_FN = 2
                elif t+window_hour+1 >np.shape(y_true)[0]:
                    y_true_window = y_true
                    y_pred_window = y_pred
                    if len(np.where(y_pred_window==1)[0]) >= 1:
                        if t > 0:
                            count_FN = 1-np.max(mask_FN[window_hour-t:window_hour]*y_pred[:t])#mask_FN[t-1:2]*y_pred[:t])
                        elif t == 0:
                            count_FN = 1
                    else:
                        count_FN = 2
                            
            elif t > len(y_true)-window_hour-1:
                if t- window_hour>=0:
                    y_true_window = y_true[t- window_hour:]
                    y_pred_window = y_pred[t- window_hour:]
                    if len(np.where(y_pred_window==1)[0]) >= 1:
                        count_FN = 1-np.max(mask_FN*y_pred[t- window_hour:t])
                    else:
                        count_FN = 2
                elif t- window_hour<0:
                    y_true_window = y_true
                    y_pred_window = y_pred
                    if len(np.where(y_pred_window==1)[0]) >= 1:
                        if t > 0:
                            count_FN = 1-np.max(mask_FN[window_hour-t:window_hour]*y_pred[:t])#[t-1:2]*y_pred[:t])
                        elif t == 0:
                            count_FN = 1
                    else:
                        count_FN = 2
            #print('size<6')
            #print('COUNT_FN:',count_FN)
            FN=FN+count_FN
            
            
    return TN, FP, FN, TP



def classification_skills_weight_solar_flares_threshold(y_real, y_predicted,panel_label,threshold):
    y_pred = y_predicted > threshold
    harp = [panel_label.index[idx][0] for idx in range(panel_label.shape[0])]
    
    harp=np.unique(harp)
    #print(harp)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for ar in harp:
        #print(ar)
        idx_ar=[i for i in range(panel_label.shape[0]) if panel_label.index[i][0]==ar]
        y_real_ar = y_real[idx_ar]
        y_pred_ar = y_pred[idx_ar]
        TN_ar,FP_ar,FN_ar,TP_ar = weighted_confusion_matrix(y_real_ar, y_pred_ar)
        TP = TP + TP_ar
        FP = FP + FP_ar
        FN = FN + FN_ar
        TN = TN + TN_ar

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
        
    if TP+FP+FN==0:
        csi = 0
    else:
        csi = TP/(TP+FP+FN)
    
    
    weighted_cm = np.zeros((2,2))
    weighted_cm[0,0]=TN
    weighted_cm[0,1]=FP
    weighted_cm[1,0]=FN
    weighted_cm[1,1]=TP
    #, pod, acc, hss, tss, fnfp, tpr, tnr, pacc, nacc, wacc, wtpr)
    return weighted_cm, tss, hss, csi

def optimize_tss(probability_prediction, Y_training):
    n_samples = 100
    step = 1. / n_samples
    
    xss = -1.
    xss_threshold = 0
    Y_best_predicted = np.zeros((Y_training.shape))
    tss_vector = np.zeros(n_samples)
    hss_vector = np.zeros(n_samples)
    xss_threshold_vector = np.zeros(n_samples)
    a = probability_prediction.max()
    b = probability_prediction.min()
    print('A:',a)
    print('B:',b)
    if a>b:
        for threshold in range(1, n_samples):
            xss_threshold = step * threshold * np.abs(a - b) + b
            xss_threshold_vector[threshold] = xss_threshold
            Y_predicted = probability_prediction > xss_threshold
            res = metrics_classification(Y_training > 0, Y_predicted, print_skills=False)
            tss_vector[threshold] = res['tss']
            hss_vector[threshold] = res['hss']
        #print(tss_vector)
        #print(hss_vector)
        max_tss=np.max(tss_vector)
        eps=1e-5
        if max_tss==0:
            max_tss=max_tss+eps
        print('MAX TSS:',max_tss)
    
        #best TSS
        idx_best_tss = np.where(tss_vector==np.max(tss_vector))  
        print('idx best tss=',idx_best_tss)
        best_xss_threshold_tss = xss_threshold_vector[idx_best_tss]
        if len(best_xss_threshold_tss)>1:
            best_xss_threshold_tss = best_xss_threshold_tss[0]
            Y_best_predicted_tss = probability_prediction > best_xss_threshold_tss
        else:
            Y_best_predicted_tss = probability_prediction > best_xss_threshold_tss
        print('best TSS')
        metrics_training_tss = metrics_classification(Y_training > 0, Y_best_predicted_tss)
    else:
        print('Exit! A==B')
        best_xss_threshold_tss=-1000
        max_tss=-1000.
    
    return best_xss_threshold_tss, max_tss


folder = 'prediction/checkpoints/'

def choose_thresholds_tss_wtss(folder,list_epochs,file_name,X_train_std,y_train, X_val_std,y_val,df_train,panel_23features_val):

    opt_threshold=np.zeros(100)
    opt_threshold_weight=np.zeros(100)
    tss=np.zeros(100)
    tss_weight=np.zeros(100)
    tss_val_weight_array=np.zeros(100)
    tss_val_array=np.zeros(100)
    
    k=0
    for file in list_epochs:
        print(file)
        model = load_model(file,compile=False)
        pred_train = model.predict(X_train_std)
        pred_val = model.predict(X_val_std)
        pred_prob = pred_train.reshape(1,len(pred_train))
        pred_prob = pred_prob[0]
        pred_prob_val = pred_val.reshape(1,len(pred_val))
        pred_prob_val = pred_prob_val[0]
        
        #OPTIMIZE new TSS with weighted matrix
        print('start optimization wtss----')
        [threshold_weight,max_tss_weight] = optimize_time_weighted_tss_solar_flares(pred_prob,y_train,df_train)#panel_train
        print('start optimization tss----')
        [threshold,max_tss] = optimize_tss(pred_prob,y_train)
        
        opt_threshold[k]=threshold
        opt_threshold_weight[k]=threshold_weight
        
        tss[k]=max_tss
        tss_weight[k]=max_tss_weight
        
        
        
        
    
        cm_val, tss_val, hss_val, csi_val = compute_cm_tss_threshold(y_val, pred_prob_val,threshold)
        tss_val_array[k]=tss_val
        
    
        wcm_val, wtss_val, whss_val, wcsi_val = classification_skills_weight_solar_flares_threshold(y_val, pred_prob_val,panel_23features_val,threshold_weight)
        tss_val_weight_array[k]=wtss_val
        
        k = k + 1
    
    return opt_threshold, opt_threshold_weight, tss, tss_weight, tss_val_array, tss_val_weight_array


def construct_panel_info_train(panel_23features_train_from_2012_09_15_to_2015_10_02):
    harp = [panel_23features_train_from_2012_09_15_to_2015_10_02.index[idx][0] for idx in range(panel_23features_train_from_2012_09_15_to_2015_10_02.shape[0])]
    harp=np.unique(harp)
    panel_harp = {}
    panel_harp['ar'] = {}
    panel_harp['ar']['idx'] = {}
    ar_list=[]
    idx_list=[]
    for ar in harp:
        ar_list.append(ar)
        idx_ar=[i for i in range(panel_23features_train_from_2012_09_15_to_2015_10_02.shape[0]) if panel_23features_train_from_2012_09_15_to_2015_10_02.index[i][0]==ar]
        idx_list.append(np.array(idx_ar))
        
    panel_harp['ar']=ar_list
    panel_harp['idx']=idx_list
    
    df_train = pandas.DataFrame (panel_harp, columns = ['ar','idx'])
    
    return df_train



def predict_ensemble(tss_val_array,perc,list_epochs,X_test_std,opt_threshold):
    
    pred_0_1_test_list=[]
    alpha=np.max(tss_val_array)*perc
    
    
    idx=np.where(np.array(tss_val_array)>alpha)
    idx=idx[0]
    print('#epochs involved in the ensemble prediction: ', len(idx))
    for i in idx:
        file=list_epochs[i]
        #print(file)
        model = load_model(file,compile=False)
        pred_test = model.predict(X_test_std)
        pred_prob_test = pred_test.reshape(1,len(pred_test))
        pred_prob_test = pred_prob_test[0]
        
        pred_0_1_test = pred_prob_test > opt_threshold[i]
        pred_0_1_test_list.append(pred_0_1_test)
    
    
    pred_0_1_arr=np.array(pred_0_1_test_list)*1
   
    
    pred_median_pred_0_1=np.median(pred_0_1_arr,axis=0)
   
    idx_to_discard=np.where(pred_median_pred_0_1==0.5)[0]
    pred_median_pred_0_1[idx_to_discard]=1
    
    return pred_median_pred_0_1
    

def read_data_construct_sets(folder_data, class_flare):
    
    panel_2012_2016_correct=pandas.read_pickle(folder_data+'panel_features_2012_2016_correct_'+class_flare+'_00.pkl')
    panel_label_2012_2016_correct=pandas.read_pickle(folder_data+'panel_label_2012_2016_correct_'+class_flare+'_00.pkl')
    
    #read panel with 23 features to save features
    panel_2017=pandas.read_pickle(folder_data+'panel_features_2017_from_august_correct.pkl')
    
    #
    # read panel feature from 2012 to 2017 to construct the test set
    panel_2012_2017=pandas.read_pickle(folder_data+'panel_features_2012_2017_correct_'+class_flare+'_00.pkl')
    # read panel label from 2012 to 2017 
    panel_label_2012_2017=pandas.read_pickle(folder_data+'panel_label_2012_2017_correct_'+class_flare+'_00.pkl')
    panel_test_2017_until_harp5634=panel_2012_2017.iloc[4790:5052]
    # cut with 23 features
    panel_test_2017_until_harp5634=panel_test_2017_until_harp5634[panel_2017.columns]
    panel_label_test_2017_until_harp5634=panel_label_2012_2017.iloc[4790:5052]
    
    #***
    #PANEL TRAIN
    panel_train = panel_2012_2016_correct[panel_2017.columns]
    
    #
    #
    #PANEL CONTAINING TRAINING AND VALIDATION
    panel_train_val_from_2012_09_15_to_2016_11_01=panel_train[0:3840+896]
    #***
    panel_23features_train_val_from_2012_09_15_to_2016_11_01=panel_train_val_from_2012_09_15_to_2016_11_01[panel_2017.columns]
    
    #Construct training and validation
    X_train_val=np.array(panel_23features_train_val_from_2012_09_15_to_2016_11_01, dtype=float)
    
    Y_train=panel_label_2012_2016_correct.values 
    
    #Construct test
    X_test=np.array(panel_test_2017_until_harp5634,dtype=float)
    Y_test=panel_label_test_2017_until_harp5634.values
    
    #Standardization
    train_val_X_std , mm, sd = training_set_standardization(X_train_val)
    X_test_std = testing_set_standardization(X_test, mm, sd)
    
    if class_flare == 'abovec' or class_flare == 'abovem':
        X_train_std=train_val_X_std[0:3840]
        X_val_std=train_val_X_std[3840:3840+896]
        Y_train_new = Y_train[0:3840]
        Y_val = Y_train[3840:3840+896]
        panel_train_from_2012_09_15_to_2015_10_02=panel_train[0:3840]
        panel_val_from_2015_09_29_to_2015_11_01=panel_train[3840:3840+896]
    elif class_flare == 'abovex':
        X_train_std=train_val_X_std[0:2560]
        X_val_std=train_val_X_std[2563:2563+896]
        Y_train_new = Y_train[0:2560]
        Y_val = Y_train[2563:2563+896]
        panel_train_from_2012_09_15_to_2015_10_02=panel_train[0:2560]
        panel_val_from_2015_09_29_to_2015_11_01=panel_train[2563:2563+896]
    
    print('#samples in training: ', Y_train_new.shape[0])
    print('#yes in training: ',Y_train_new.sum())
    print('#samples in validation: ',Y_val.shape[0])
    print('#yes in validation: ', Y_val.sum())
    
    panel_23features_val_from_2015_09_29_to_2015_11_01=panel_val_from_2015_09_29_to_2015_11_01[panel_2017.columns]
    
    
    panel_23features_train_from_2012_09_15_to_2015_10_02=panel_train_from_2012_09_15_to_2015_10_02[panel_2017.columns]
    
    df_train=construct_panel_info_train(panel_23features_train_from_2012_09_15_to_2015_10_02)
    
    return X_train_std,Y_train_new, X_val_std, Y_val, df_train, panel_23features_val_from_2015_09_29_to_2015_11_01, X_test_std, Y_test


def select_best_patience_on_val(folder_path,class_flare,X_train_std,Y_train_new, df_train, X_val_std, Y_val,panel_23features_val_from_2015_09_29_to_2015_11_01):
    num_patience=5
    tss_opt_tss = numpy.zeros(num_patience)
    wtss_opt_wtss = numpy.zeros(num_patience)
    threshold_opt_tss = numpy.zeros(num_patience)
    threshold_opt_wtss = numpy.zeros(num_patience)

    for i in range(1,6):
        model = Sequential()
    
        model.add(Dense(64, input_dim=23,  kernel_regularizer=l2(0.01), activation='relu'))
        model.add(Dense(48,  kernel_regularizer=l2(0.01), activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(24,  activation='relu'))
        model.add(Dense(16,  activation='relu'))
        model.add(Dense(8,   activation='relu'))
        model.add(Dense(4,  activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(lr=1e-3)
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        patience=i*10
    
        model.load_weights(folder_path+'early_stop'+str(patience)+'_nn7593param_'+class_flare+'_00_std_reg2layers_train_from_2012_09_15_to_2015_10_02_val_from_2015_09_29_to_2015_11_01.hdf5')
    
        #
        #class M
        y_pred_train_early=model.predict(X_train_std)
        pred_prob_train_early = y_pred_train_early.reshape(1,len(y_pred_train_early))
        pred_prob_train_early = pred_prob_train_early[0]
    
        [threshold_weight,max_tss_weight] = optimize_time_weighted_tss_solar_flares(pred_prob_train_early,Y_train_new,df_train,panel_23features_val_from_2015_09_29_to_2015_11_01)#panel_train
        print(threshold_weight)
        threshold_opt_wtss[i-1]=threshold_weight
        print('start optimization tss----')
        [threshold,max_tss] = optimize_tss(pred_prob_train_early,Y_train_new)
        print(threshold)
        threshold_opt_tss[i-1]=threshold
        pred_prob_val = model.predict(X_val_std)
    
        y_pred_val_weight = pred_prob_val>threshold_weight
        y_pred_val_weight = y_pred_val_weight*1
        y_pred_val_weight = y_pred_val_weight[:,0]
    
        y_pred_val = pred_prob_val>threshold
        y_pred_val = y_pred_val*1
        y_pred_val = y_pred_val[:,0]
        cm_val, tss_val, hss_val, csi_val = compute_cm_tss(Y_val, y_pred_val_weight)#_weight
        print('Skill scores (otimization weighted tss)')
        print(cm_val)
        print('tss = ','{:0.4f}'.format(tss_val))
        print('hss = ','{:0.4f}'.format(hss_val))
        print('csi = ','{:0.4f}'.format(csi_val))
        cm_val, tss_val, hss_val, csi_val = compute_cm_tss(Y_val, y_pred_val)#_weight
        print('Skill scores (optimization tss)')
        print(cm_val)
        print('tss = ','{:0.4f}'.format(tss_val))
        print('hss = ','{:0.4f}'.format(hss_val))
        print('csi = ','{:0.4f}'.format(csi_val))
        tss_opt_tss[i-1]=tss_val
    
        wcm_val, wtss_val, whss_val, wcsi_val = compute_weight_cm_tss_harp(Y_val, y_pred_val_weight,panel_23features_val_from_2015_09_29_to_2015_11_01)#_weight
        print('Weighted Skill scores (optimization weighted tss)')
        print(wcm_val)
        print('wtss = ','{:0.4f}'.format(wtss_val))
        print('whss = ','{:0.4f}'.format(whss_val))
        print('wcsi = ','{:0.4f}'.format(wcsi_val))
        wtss_opt_wtss[i-1]=wtss_val
    
        wcm_val, wtss_val, whss_val, wcsi_val = compute_weight_cm_tss_harp(Y_val, y_pred_val,panel_23features_val_from_2015_09_29_to_2015_11_01)#_weight
        print('Weighted Skill scores (optimization tss)')
        print(wcm_val)
        print('wtss = ','{:0.4f}'.format(wtss_val))
        print('whss = ','{:0.4f}'.format(whss_val))
        print('wcsi = ','{:0.4f}'.format(wcsi_val))
    
    
    return tss_opt_tss,wtss_opt_wtss, threshold_opt_tss, threshold_opt_wtss 
