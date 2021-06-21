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

    flare_class_list = ('abovex',)#'abovec', 'abovem',
    window_list = ('6','12','24')
    issuing_time_list = ('00', '06', '12', '18')
    folder_data = 'data/'
    folder_data_output = 'data_bmix/'

    for it_list in flare_class_list:
        for it_issuing in issuing_time_list:
            X_test_aux = pandas.read_pickle(
                'data/dati_evento_settembre_2017.pkl')
            Y_test_aux = json.load(open(
                'data/flare_association_dati_evento_settembre_2017.json', 'r'))

            for it_window in window_list:

                # cfg_name_test = 'HybridLasso_' + it_list + '_' + it_window + '_0_' \
                #                 + it_issuing + '_Bmix_all-test.json'
                #
                # fc_dataset_test = fc.access_db(cfg_name_test, offline=True)
                #
                # # legge il point in time sul server
                # fc_dataset_test.load_dataset('dati_evento_settembre_2017.pkl', \
                #         'flare_association_dati_evento_settembre_2017.json',offline=True)
                #
                #
                # df_X_test = pandas.DataFrame()
                # df_Y_test = pandas.DataFrame()
                #
                # df_X_test = pandas.concat([df_X_test, fc_dataset_test.feature_df], axis=0)
                # df_Y_test = pandas.concat([df_Y_test, fc_dataset_test.label_df], axis=0)
                #
                # df_X_test.to_pickle(
                #     folder_data_output + '/X_test_' + cfg_name_test[0:len(cfg_name_test) - 5] + '.pkl')
                #
                # df_Y_test.to_pickle(
                #     folder_data_output + '/Y_test_' + cfg_name_test[0:len(cfg_name_test) - 5] + '.pkl')


                cfg_name_training = 'HybridLasso_' + it_list + '_' + it_window +'_0_' \
                           + it_issuing + '_Bmix_all-train.json'
                fc_dataset_training = fc.access_db(cfg_name_training, offline=True)

                # legge il point in time sul server
                fc_dataset_training.load_dataset('training_2012_2016.pkl', \
                            'flare_association_training_2012_2016.json', offline=True)
                df_X_training = pandas.DataFrame()
                df_Y_training = pandas.DataFrame()

                df_X_training = pandas.concat([df_X_training, fc_dataset_training.feature_df], axis=0)
                df_Y_training = pandas.concat([df_Y_training, fc_dataset_training.label_df], axis=0)



                df_X_training.to_pickle(
                    folder_data_output + '/X_training_' + cfg_name_training[0:len(cfg_name_training)-5] + '.pkl')

                df_Y_training.to_pickle(
                    folder_data_output + '/Y_training_' + cfg_name_training[0:len(cfg_name_training)-5] + '.pkl')
