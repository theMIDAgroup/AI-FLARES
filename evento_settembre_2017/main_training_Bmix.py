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

    flare_class_list = ('abovex'),# ('abovec', 'abovem', 'abovex')
    window_list = ( '12',)# '24')#'6'
    issuing_time_list = ('00', '12')#06', '12', '18')
    folder_data = 'data_blos/'
    folder_res = 'res/res_blos/'


    for it_list in flare_class_list:
        for it_issuing in issuing_time_list:
            for it_window in window_list:

                table_features_fulldata = []
                table_ranking_fulldata = []
                file_df_fulldata = folder_res + '/feature_importance_HybridLasso_' + it_list + '_' + it_window + '_0_' + it_issuing + '_Bmix_all-train.pkl'
                file_ranking_fulldata = folder_res + '/ranking_HybridLasso_' + it_list + '_' + it_window + '_0_' + it_issuing + '_Bmix_all-train.pkl'
                file_df_fulldata_csv = folder_res + '/feature_importance_HybridLasso_' + it_list + '_' + it_window + '_0_' + it_issuing + '_Bmix_all-train.csv'
                file_ranking_fulldata_csv = folder_res + '/ranking_HybridLasso_' + it_list + '_' + it_window + '_0_' + it_issuing + '_Bmix_all-train.csv'

                cfg_name = 'HybridLasso_' + it_list + '_' + it_window + '_0_' + it_issuing + '_Bmix_all-train.json'


                fc_dataset = fc.access_db(cfg_name, offline=True)

                aux_X_pkl = folder_data + '/X_training_' + cfg_name[0:len(cfg_name) - 5] + '.pkl'
                aux_Y_pkl = folder_data + '/Y_training_' + cfg_name[0:len(cfg_name) - 5] + '.pkl'
                df_X = pickle.load(open(aux_X_pkl))
                df_Y = pickle.load(open(aux_Y_pkl))

                active_features = df_X.columns

                algo = fc.algorithm(fc_dataset)

                X_training = numpy.array(df_X)
                Y_training = numpy.array(df_Y)

                Xn_training, Yn_training, _mean_, _std_, _Ymean_, _Ystd_ = \
                    preprocessing_training(fc_dataset, X_training, Y_training)
                algo.estimator.fit(Xn_training, Yn_training)

                table_features_fulldata.append([Xn_training.shape[0],Yn_training.sum(),
                                            algo.estimator.metrics_training['0']['tss'],
                                            algo.estimator.metrics_training['0']['hss'],
                                            algo.estimator.metrics_training['0']['acc'],
                                            algo.estimator.metrics_training['0']['far'],
                                            algo.estimator.metrics_training['0']['fnfp'],
                                            algo.estimator.metrics_training['0']['pod'],
                                            algo.estimator.metrics_training['0']['balance label']] \
                                           + algo.estimator.metrics_training['feature importance'])

                # ranking
                selector = RFE(algo.estimator.estimator, n_features_to_select=1)
                selector.fit(Xn_training, Yn_training)
                table_ranking_fulldata.append(selector.ranking_)



                label_column = ['#point-in-time', 'num_label=1', 'tss', 'hss', 'accuracy', 'far', 'fnfp', 'pod',
                        'balance label'] + active_features.tolist()
                df_fulldata = pandas.DataFrame(table_features_fulldata, columns=label_column)
                df_fulldata.to_pickle(file_df_fulldata)
                df_fulldata.to_csv(file_df_fulldata_csv, sep='\t', float_format='%10.2f')

                df_ranking = pandas.DataFrame(table_ranking_fulldata, columns=active_features.tolist())
                df_ranking.to_pickle(file_ranking_fulldata)
                df_ranking.to_csv(file_ranking_fulldata_csv, sep='\t', float_format='%10.2f')
                numpy.save('active_features.npy', active_features)
