
from sklearn.feature_selection import RFECV, RFE
import flarecast_engine as fc
import numpy as np
import pandas
from flarecast_engine import training_set_standardization, testing_set_standardization

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)


def warn(*args, **kwargs):
    pass

warnings.warn = warn

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


def preprocessing_testing(db, X, Y,_mean_,_std_,_Ymean_,_Ystd_):
    Xs = X
    Ys = Y


    if db.algorithm_info['parameters']['preprocessing']['standardization_feature']:
        Xs = testing_set_standardization(X, _mean_, _std_)

    if db.algorithm_info['parameters']['preprocessing']['standardization_label']:
        Ys = testing_set_standardization(Y, _Ymean_, _Ystd_)

    return Xs, Ys




if __name__ == '__main__':


    topo = 1


    n_run = 100
    flare_class =  ('abovec','abovem',)#'abovec'
    issuing = ('00','06','12','18')#'00'

    start_time = '20120914'
    end_time = '20160430'
    folder_bootstrap = 'bootstrap/'


    algorithm_list =('HybridLasso',)# ('RandomForest','HybridLasso', )#('RandomForest','HybridLasso', 'HybridLogit', )#('SVC-CV',)# 'SVC-CV',('HybridLasso', 'HybridLogit', 'RandomForest')
    algo_str_list = ('HybridLasso',)#('RandomForest','HybridLasso', )#('RandomForest','HybridLasso', 'HybridLogit', )#('SVC-CV',)#  ,'SVC'('HybridLasso', 'HybridLogit', 'RandomForest')


    #ising_energy_blos/ising_energy
    #ising_energy_br/ising_energy
    #ising_energy_part_blos/ising_energy_part
    #ising_energy_part_br/ising_energy_part

    ff = 'D_topo'
    ff2use = ['topo']#, #'ising_energy_part_br/ising_energy_part',
             # 'ising_energy_blos/ising_energy']#, 'ising_energy_part_blos/ising_energy_part']



    for it_class in flare_class:
        for it_issuing in issuing:
            properties = 'Bmix_' + it_issuing + 'h'
            folder_res = folder_bootstrap + '/results_solo_ising/results_' + it_class + '_' + properties
            AR_list = np.loadtxt('data/AR_list_' + properties + '_' + start_time + '_' + end_time + '.txt')
            label = 'flaring'
            n_AR = len(AR_list)
            for algorithm in algorithm_list:
                file_df_fulldata = folder_res + '/feature_importance_' +it_class+ '_'+properties +'_'+start_time+'_'+end_time+'_'+algorithm +'_' + ff + '.pkl'
                file_ranking_fulldata = folder_res + '/ranking_'+it_class+ '_'+properties +'_'+start_time+'_'+end_time+'_'+algorithm +'_' + ff + '.pkl'
                file_df_predict_fulldata = folder_res + '/skill_scores_testing_' +it_class+ '_'+properties +'_'+start_time+'_'+end_time+'_'+algorithm +'_' + ff + '.pkl'

                file_df_fulldata_csv = folder_res + '/feature_importance_' +it_class+ '_'+properties +'_'+start_time+'_'+end_time+'_'+algorithm +'_' +  ff + '.csv'
                file_ranking_fulldata_csv = folder_res + '/ranking_'  +it_class+ '_'+properties +'_'+start_time+'_'+end_time+'_'+algorithm +'_' +  ff + '.csv'
                file_df_predict_fulldata_csv = folder_res + '/skill_scores_testing_'  +it_class+ '_'+properties +'_'+start_time+'_'+end_time+'_'+algorithm  +'_'+ ff + '.csv'

                table_features_fulldata = []
                table_ranking_fulldata = []
                table_predict_features_fulldata = []

                db = fc.access_db('%s-train.json' % algorithm)

                algo = fc.algorithm(db)

                for it_run in np.arange(n_run):
                    print(it_run)
                    df_X_training = pandas.read_pickle(folder_bootstrap + '/data/X_training_'+properties +'_'+start_time+'_'+end_time+'_'+ str(it_run) + '.pkl')
                    df_Y_training = pandas.read_pickle(folder_bootstrap + '/data/Y_training_' + it_class + '_'+properties +'_'+start_time+'_'+end_time+'_' + str(it_run) + '.pkl')

                    df_X_testing = pandas.read_pickle(folder_bootstrap + '/data/X_testing_'+properties +'_'+start_time+'_'+end_time+'_'+ str(it_run) + '.pkl')
                    df_Y_testing = pandas.read_pickle(folder_bootstrap + '/data/Y_testing_' + it_class + '_'+properties +'_'+start_time+'_'+end_time+'_' + str(it_run) + '.pkl')

                    if topo == 1:
                        df_X_training = df_X_training[ff2use]
                        df_X_testing = df_X_testing[ff2use]

                    X_training = np.array(df_X_training)

                    #if properties == 'Bmix_06h' and flare_class == 'abovec':
                    #    Y_training = np.array(df_Y_training[0])
                    #else:
                    #    Y_training = np.array(df_Y_training[label])
                    Y_training = np.array(df_Y_training[label])

                    active_features = df_X_training.columns

                    X_testing = np.array(df_X_testing)
                    #if properties == 'Bmix_06h' and flare_class == 'abovec':
                    #    Y_testing = np.array(df_Y_testing[0])
                    #else:
                    Y_testing = np.array(df_Y_testing[label])

                    # training
                    Xn_training, Yn_training, _mean_, _std_, _Ymean_, _Ystd_ = preprocessing_training(db,X_training,Y_training)
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

                    # testing
                    Xn_testing, Yn_testing = preprocessing_testing(db, X_testing, Y_testing, _mean_, _std_, _Ymean_, _Ystd_)
                    predict = algo.estimator.predict(Xn_testing, Yn_testing)

                    table_predict_features_fulldata.append([Xn_testing.shape[0], Yn_testing.sum(),
                                                            algo.estimator.metrics_testing['0']['tss'],
                                                            algo.estimator.metrics_testing['0']['hss'],
                                                            algo.estimator.metrics_testing['0']['acc'],
                                                            algo.estimator.metrics_testing['0']['far'],
                                                            algo.estimator.metrics_testing['0']['fnfp'],
                                                            algo.estimator.metrics_testing['0']['pod'],
                                                            algo.estimator.metrics_testing['0']['balance label']])


                label_column = ['#point-in-time','num_label=1','tss', 'hss', 'accuracy', 'far', 'fnfp', 'pod', 'balance label'] + active_features.tolist()
                df_fulldata = pandas.DataFrame(table_features_fulldata, columns=label_column)
                df_fulldata.to_pickle(file_df_fulldata)
                df_fulldata.to_csv(file_df_fulldata_csv,sep='\t', float_format='%10.2f')

                df_ranking = pandas.DataFrame(table_ranking_fulldata, columns=active_features.tolist())
                df_ranking.to_pickle(file_ranking_fulldata)
                df_ranking.to_csv(file_ranking_fulldata_csv,sep='\t', float_format='%10.2f')
                np.save('active_features.npy', active_features)

                label_column_testing = ['#point-in-time','num_label=1','tss', 'hss', 'accuracy', 'far', 'fnfp', 'pod', 'balance label']
                df_predict_fulldata = pandas.DataFrame(table_predict_features_fulldata, columns=label_column_testing)
                df_predict_fulldata.to_pickle(file_df_predict_fulldata)
                df_predict_fulldata.to_csv(file_df_predict_fulldata_csv,sep='\t', float_format='%10.2f')



