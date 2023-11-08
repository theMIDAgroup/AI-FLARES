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

    flare_class_list = ('abovec','abovem', 'abovex')#
    window_list = ('6',)#'12', '24',)
    issuing_time_list = ('00','06', '12', '18')
    folder_data = 'data_blos/'
    folder_res = 'res/res_blos_hybrid/'
    fh = 'withfh'
    fh_exit = 'withfh'
    kind = '_Blos_all_'
    for it_list in flare_class_list:
        for it_issuing in issuing_time_list:
            for it_window in window_list:

                table_features_fulldata = []
                table_ranking_fulldata = []
                table_predict_features_fulldata = []
                file_df_fulldata = folder_res + '/feature_importance_HybridLasso_' + it_list + '_' + it_window + '_0_' + it_issuing +   kind + fh_exit + '-train.pkl'
                file_ranking_fulldata = folder_res + '/ranking_HybridLasso_' + it_list + '_' + it_window + '_0_' + it_issuing +   kind + fh_exit + '-train.pkl'
                file_df_predict_fulldata = folder_res + '/feature_importance_HybridLasso_' + it_list + '_' + it_window + '_0_' + it_issuing +   kind + fh_exit + '-predict.pkl'
                file_df_fulldata_csv = folder_res + '/feature_importance_HybridLasso_' + it_list + '_' + it_window + '_0_' + it_issuing +   kind + fh_exit + '-train.csv'
                file_ranking_fulldata_csv = folder_res + '/ranking_HybridLasso_' + it_list + '_' + it_window + '_0_' + it_issuing +   kind + fh_exit + '-train.csv'
                file_df_predict_fulldata_csv = folder_res + '/feature_importance_HybridLasso_' + it_list + '_' + it_window + '_0_' + it_issuing +   kind + fh_exit + '-predict.csv'

                cfg_name = 'HybridLasso_' + it_list + '_' + it_window + '_0_'  \
                           + it_issuing +   kind + fh
                cfg_name_test = 'HybridLasso_' + it_list + '_' + it_window + '_0_' \
                                    + it_issuing +   kind + fh

                fc_dataset = fc.access_db(cfg_name + '-train.json' , offline=False)
                fc_dataset.load_dataset( cfg_name + '-train', offline=True)

                if fh_exit == 'nofh':
                    if len(fc_dataset.feature_dirty_df.columns.to_list())==24:
                        idx = [19, 20]
                        fc_dataset.feature_dirty_df = fc_dataset.feature_dirty_df.drop(columns=['flare_index_past','flare_past'])
                        fc_dataset.feature_df = fc_dataset.feature_df.drop(columns=['flare_index_past','flare_past'])
                        fc_dataset.X = numpy.delete(fc_dataset.X, [19, 20], axis = 1)
                        fc_dataset.Xs = numpy.delete(fc_dataset.Xs, [19, 20], axis = 1)
                        fc_dataset._mean_ = numpy.delete(fc_dataset._mean_, [19, 20])
                        fc_dataset._std_ = numpy.delete(fc_dataset._std_, [19, 20])

                algo = fc.algorithm(fc_dataset)
                algo.train_db(fc_dataset)

                fc_dataset.write_model(algo, offline=True)

                fc_dataset_test = fc.access_db(cfg_name_test + '-predict.json', offline=True)
                fc_dataset_test.load_dataset(cfg_name_test+'-predict', offline=False)
                if fh_exit == 'nofh':
                    if len(fc_dataset.feature_dirty_df.columns.to_list()) == 24:
                        fc_dataset_test.feature_dirty_df = fc_dataset_test.feature_dirty_df.drop(columns=['flare_index_past', 'flare_past'])
                        fc_dataset_test.feature_df = fc_dataset_test.feature_df.drop(columns=['flare_index_past', 'flare_past'])
                        fc_dataset_test.X = numpy.delete(fc_dataset_test.X, [19, 20], axis=1)
                        fc_dataset_test.Xs = numpy.delete(fc_dataset_test.Xs, [19, 20], axis=1)

                algo_predict = fc_dataset_test.read_model(offline=True)

                algo_predict.predict_db(fc_dataset_test)
                fc_dataset_test.write_prediction(algo_predict,offline=True)

                #train

                table_features_fulldata.append([fc_dataset.feature_df.shape[0],
                                                fc_dataset.label_df['flaring'].sum(),
                                                algo_predict.estimator.metrics_training['0']['tss'],
                                                algo_predict.estimator.metrics_training['0']['hss'],
                                                algo_predict.estimator.metrics_training['0']['acc'],
                                                algo_predict.estimator.metrics_training['0']['far'],
                                                algo_predict.estimator.metrics_training['0']['fnfp'],
                                                algo_predict.estimator.metrics_training['0']['pod'],
                                                algo_predict.estimator.metrics_training['0']['balance label']] \
                                           + algo_predict.estimator.metrics_training['feature importance'])

                # ranking
                selector = RFE(algo.estimator.estimator, n_features_to_select=1)
                selector.fit(fc_dataset.feature_df, fc_dataset.label_df)
                table_ranking_fulldata.append(selector.ranking_)

                #predict
                table_predict_features_fulldata.append([fc_dataset_test.feature_df.shape[0],
                                                fc_dataset_test.label_df['flaring'].sum(),
                                                algo_predict.estimator.metrics_testing['0']['tss'],
                                                algo_predict.estimator.metrics_testing['0']['hss'],
                                                algo_predict.estimator.metrics_testing['0']['acc'],
                                                algo_predict.estimator.metrics_testing['0']['far'],
                                                algo_predict.estimator.metrics_testing['0']['fnfp'],
                                                algo_predict.estimator.metrics_testing['0']['pod'],
                                                algo_predict.estimator.metrics_testing['0']['balance label']] )

                label_column = ['#point-in-time', 'num_label=1', 'tss', 'hss', 'accuracy', 'far', 'fnfp', 'pod',
                                'balance label'] + fc_dataset.feature_df.columns.to_list()
                df_fulldata = pandas.DataFrame(table_features_fulldata, columns=label_column)
                df_fulldata.to_pickle(file_df_fulldata)
                df_fulldata.to_csv(file_df_fulldata_csv, sep='\t', float_format='%10.2f')

                df_ranking = pandas.DataFrame(table_ranking_fulldata, columns=fc_dataset.feature_df.columns)
                df_ranking.to_pickle(file_ranking_fulldata)
                df_ranking.to_csv(file_ranking_fulldata_csv, sep='\t', float_format='%10.2f')
                numpy.save('active_features.npy', fc_dataset.feature_df.columns.to_list())

                label_column_test = ['#point-in-time', 'num_label=1', 'tss', 'hss', 'accuracy', 'far', 'fnfp', 'pod',
                                'balance label']
                df_predict_fulldata_test = pandas.DataFrame(table_predict_features_fulldata, columns=label_column_test)
                df_predict_fulldata_test.to_pickle(file_df_predict_fulldata)
                df_predict_fulldata_test.to_csv(file_df_predict_fulldata_csv, sep='\t', float_format='%10.2f')

