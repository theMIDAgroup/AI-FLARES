
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

    it_init = 0
    it_final = 19
    step_feature = 1

    n_feature = abs(it_final-it_init)

    topo = 0


    n_run = 100
    flare_class =  ('abovem',)#'abovec'
    issuing = ('18','06','12','18')#'00'

    start_time = '20120914'
    end_time = '20160430'
    folder_bootstrap = 'bootstrap'


    algorithm_list =('HybridLasso',)# ('RandomForest','HybridLasso', )#('RandomForest','HybridLasso', 'HybridLogit', )#('SVC-CV',)# 'SVC-CV',('HybridLasso', 'HybridLogit', 'RandomForest')
    algo_str_list = ('HybridLasso',)#('RandomForest','HybridLasso', )#('RandomForest','HybridLasso', 'HybridLogit', )#('SVC-CV',)#  ,'SVC'('HybridLasso', 'HybridLogit', 'RandomForest')




    for it_class in flare_class:
        for it_issuing in issuing:
            properties = 'Bmix_' + it_issuing + 'h'
            folder_res = folder_bootstrap + '/fXf_results_' + it_class + '_' + properties
            AR_list = np.loadtxt('data/AR_list_' + properties + '_' + start_time + '_' + end_time + '.txt')
            label = 'flaring'
            n_AR = len(AR_list)


            
            for algorithm in algorithm_list:

                indici_features_full = np.loadtxt(
                    'data/indici_ordinati_per_media_' + it_class + '_' + properties + '_' + start_time + '_' + end_time + '_' + algorithm + '.txt',
                    dtype=int)

                if step_feature == -1:
                    indici_features_full = np.flip(indici_features_full, 0)

                db = fc.access_db('%s-train.json' % algorithm)

                algo = fc.algorithm(db)

                for it_run in np.arange(n_run):
                    print(it_run)
                    df_X_training = pandas.read_pickle(folder_bootstrap + '/data/X_training_'+properties +'_'+start_time+'_'+end_time+'_'+ str(it_run) + '.pkl')
                    df_Y_training = pandas.read_pickle(folder_bootstrap + '/data/Y_training_' + it_class + '_'+properties +'_'+start_time+'_'+end_time+'_' + str(it_run) + '.pkl')

                    df_X_testing = pandas.read_pickle(folder_bootstrap + '/data/X_testing_'+properties +'_'+start_time+'_'+end_time+'_'+ str(it_run) + '.pkl')
                    df_Y_testing = pandas.read_pickle(folder_bootstrap + '/data/Y_testing_' + it_class + '_'+properties +'_'+start_time+'_'+end_time+'_' + str(it_run) + '.pkl')

                    file_df_fulldata = folder_res + '/fXf_feature_importance_' + it_class + '_' + properties + '_' + start_time + '_' + end_time +'_' + str(it_run) + '_' + algorithm + '.pkl'
                    file_df_predict_fulldata = folder_res + '/fXf_skill_scores_testing_' + it_class + '_' + properties + '_' + start_time + '_' + end_time +'_' + str(it_run) +'_' + algorithm + '.pkl'

                    file_df_fulldata_csv = folder_res + '/fXf_feature_importance_' + it_class + '_' + properties + '_' + start_time + '_' + end_time + '_'+ str(it_run) +  '_' + algorithm + '.csv'
                    file_df_predict_fulldata_csv = folder_res + '/fXf_skill_scores_testing_' + it_class + '_' + properties + '_' + start_time + '_' + end_time +'_' + str(it_run) + '_' + algorithm + '.csv'

                    table_features_fulldata = []
                    table_predict_features_fulldata = []


                    if topo == 1:
                        df_X_training = df_X_training[['R_topo','topo']]
                        df_X_testing = df_X_testing[['R_topo', 'topo']]

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



                    for it_f in np.arange(n_feature):
                        print('n predittori ' + str(it_f))

                        indici_aux = indici_features_full[0:it_f + 1]
                        X_training_fXf = X_training[:, indici_aux]
                        X_testing_fXf = X_testing[:, indici_aux]

                        active_features_fXf = active_features[indici_aux]
                        # preprecessing training
                        Xn_training, Yn_training, _mean_, _std_, _Ymean_, _Ystd_ = preprocessing_training(db,
                                                                                                          X_training_fXf,
                                                                                                          Y_training)

                        # fase di fit
                        algo.estimator.fit(Xn_training, Yn_training)

                        table_features_fulldata.append([Xn_training.shape[0], Yn_training.sum(),
                                                        algo.estimator.metrics_training['0']['tss'],
                                                        algo.estimator.metrics_training['0']['hss'],
                                                        algo.estimator.metrics_training['0']['acc'],
                                                        algo.estimator.metrics_training['0']['far'],
                                                        algo.estimator.metrics_training['0']['fnfp'],
                                                        algo.estimator.metrics_training['0']['pod'],
                                                        algo.estimator.metrics_training['0']['balance label']] \
                                                       + algo.estimator.metrics_training['feature importance'])

                        # testing
                        Xn_testing, Yn_testing = preprocessing_testing(db, X_testing_fXf, Y_testing, _mean_, _std_, _Ymean_,
                                                                       _Ystd_)
                        predict = algo.estimator.predict(Xn_testing, Yn_testing)

                        table_predict_features_fulldata.append([Xn_testing.shape[0], Yn_testing.sum(),
                                                                algo.estimator.metrics_testing['0']['tss'],
                                                                algo.estimator.metrics_testing['0']['hss'],
                                                                algo.estimator.metrics_testing['0']['acc'],
                                                                algo.estimator.metrics_testing['0']['far'],
                                                                algo.estimator.metrics_testing['0']['fnfp'],
                                                                algo.estimator.metrics_testing['0']['pod'],
                                                                algo.estimator.metrics_testing['0']['balance label']])

                    label_column = ['#point-in-time','num_label=1','tss', 'hss', 'accuracy', 'far', 'fnfp', 'pod', 'balance label'] + active_features_fXf.tolist()
                    df_fulldata = pandas.DataFrame(table_features_fulldata, columns=label_column)
                    df_fulldata.to_pickle(file_df_fulldata)
                    df_fulldata.to_csv(file_df_fulldata_csv,sep='\t', float_format='%10.2f')

                    label_column_testing = ['#point-in-time','num_label=1','tss', 'hss', 'accuracy', 'far', 'fnfp', 'pod', 'balance label']
                    df_predict_fulldata = pandas.DataFrame(table_predict_features_fulldata, columns=label_column_testing)
                    df_predict_fulldata.to_pickle(file_df_predict_fulldata)
                    df_predict_fulldata.to_csv(file_df_predict_fulldata_csv,sep='\t', float_format='%10.2f')



