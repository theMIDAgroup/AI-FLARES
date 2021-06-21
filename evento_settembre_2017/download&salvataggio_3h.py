# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:46:50 2017

@author: benvenuto
"""

import sys
import numpy
import json
import flarecast_engine as fc
from flarecast_engine import training_set_standardization
import pandas
from sklearn.feature_selection import RFE
import pickle

# if you modify definitions.py you have to relaod the module
try:
    # python 2
    fc = reload(fc)
except BaseException:
    # python 3
    import importlib
#    fc =   importlib.realod(fc)

def preprocessing_training(db, X, Y):
    Xs = X
    Ys = Y
    if db.algorithm_info['parameters']['preprocessing']['standardization_feature']:
        Xs, _mean_, _std_ = training_set_standardization(X)
    if db.algorithm_info['parameters']['preprocessing']['standardization_label']:
        Ys, _Ymean_, _Ystd_ = training_set_standardization(Y)
    else:
        _Ymean_ = 0
        _Ystd_ = 0

    return Xs, Ys, _mean_, _std_, _Ymean_, _Ystd_



if __name__ == '__main__':

    sys.setrecursionlimit(5000)

    # LEGGI TUTTE LE FEATURE DEL POINT IN TIME X NEW
    # e imposta quali feature leggere per fare il training per questo evento
    # ATTENZIONE : CFG_X_NEW E' UN FILE DI TIPO CFG_TRAIN
    # IMPORTANTE : NON DEVE STANDARDIZZARE, RISCHIO IMPLOSIONE IN build_arrays ###

    # evento_settembre_2017

    flare_class_list = ( 'abovec','abovex','abovem',)
    window_list = ('6','24','12', )
    issuing_time_list = ( '00','18','06', '12',)
    folder_data = 'data_bmix/'
    folder_res = 'res/res_bmix_hybrid/'
    fh = 'withfh'
    kind = '_Bmix_all_'

    cfg_name = 'Bmix_cadence3h'
    fc_dataset = fc.access_db(cfg_name + '-train.json', offline=False)
    fc_dataset.load_dataset_3h(cfg_name + '-train', offline=False)

