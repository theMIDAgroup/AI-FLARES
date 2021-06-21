
import numpy as np
import pandas


if __name__ == '__main__':

    new_dataset = False

    folder_bootstrap = 'bootstrap/'

    n_run = 100
    flare_class = 'abovem'
    properties = 'Bmix_00h'
    start_time = '20120914'
    end_time = '20160430'



    AR_list = np.loadtxt('data/AR_list_'+properties+'_'+start_time+'_'+end_time+'.txt')
    AR_path = 'data/AR_full_'+properties+'_'+start_time+'_'+end_time
    label = 'flaring'
    n_AR = len(AR_list)


    n_AR_to_train = int(n_AR * 0.66)
    n_AR_to_test = n_AR - n_AR_to_train


    to_train = np.loadtxt('data/index_for_training_set_'+ properties+'_'+start_time+'_'+end_time+'.txt')
    to_test = np.loadtxt('data/index_for_test_set_'+ properties+'_'+start_time+'_'+end_time+'.txt')


    for it_run in np.arange(n_run):
        print(it_run)
        df_X_training = pandas.DataFrame()
        df_Y_training = pandas.DataFrame()

        df_X_testing = pandas.DataFrame()
        df_Y_testing = pandas.DataFrame()

        for it_train in to_train[it_run,:]:
            file_AR_df_feature = AR_path+'/AR_' + str(it_train) + '_feature_'+properties+'_'+start_time+'_'+end_time+'.pkl'
            file_AR_df_label = AR_path+'/AR_' + str(it_train) + '_label_'+flare_class+'_'+properties+'_'+start_time+'_'+end_time+'.pkl'
            df_feature = pandas.read_pickle(file_AR_df_feature)
            df_label = pandas.read_pickle(file_AR_df_label)

            df_X_training = pandas.concat([df_X_training, df_feature], axis=0)
            df_Y_training = pandas.concat([df_Y_training, df_label], axis=0)

        for it_test in to_test[it_run,:]:
            file_AR_df_feature = AR_path+'/AR_' + str(it_test) + '_feature_'+properties+'_'+start_time+'_'+end_time+'.pkl'
            file_AR_df_label = AR_path+'/AR_' + str(it_test) + '_label_' + flare_class + '_'+properties+'_'+start_time+'_'+end_time+'.pkl'
            df_feature = pandas.read_pickle(file_AR_df_feature)
            df_label = pandas.read_pickle(file_AR_df_label)

            df_X_testing = pandas.concat([df_X_testing, df_feature], axis=0)
            df_Y_testing = pandas.concat([df_Y_testing, df_label], axis=0)


        df_X_training.to_pickle(folder_bootstrap + '/data/X_training_'+properties+'_'+start_time+'_'+end_time+'_'+ str(it_run) + '.pkl')
        df_X_testing.to_pickle(folder_bootstrap + '/data/X_testing_'+properties+'_'+start_time+'_'+end_time+'_' + str(it_run) + '.pkl')

        df_Y_training.to_pickle(
                folder_bootstrap + '/data/Y_training_'+ flare_class + '_'+properties+'_'+start_time+'_'+end_time+'_' + str(it_run) + '.pkl')
        df_Y_testing.to_pickle(folder_bootstrap + '/data/Y_testing_'
                               + flare_class + '_'+properties+'_'+start_time+'_'+end_time+'_' + str(it_run) + '.pkl')