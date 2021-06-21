import pandas
import numpy as np

def drop_row(X_1h,Y_1h,ora):

    aux = []
    for it in np.arange(X_1h.shape[0]):
        if X_1h.index[it][1][11:13] == ora:
            aux.append(it)

    X_24h = X_1h.ix[aux]
    Y_24h = Y_1h.ix[aux]
    return X_24h, Y_24h

if __name__ == '__main__':


    write_df_full = True
    extract_AR = False
    write_df_ora = False

    flare_class = 'abovem'
    properties = 'Bmix_18h'
    start_time = '20120914'
    end_time = '20160430'
    file_X = 'data/feature_'+properties+'_'+start_time+'_'+end_time+'_clean.pkl'
    file_Y = 'data/label_'+ flare_class +'_'+properties+'_'+start_time+'_'+end_time+'_clean.pkl'

    df_feature_full = pandas.read_pickle(file_X)
    df_label_full = pandas.read_pickle(file_Y)


    AR_list = np.loadtxt('data/AR_list_'+properties+'_'+start_time+'_'+end_time+'.txt')




    for it in AR_list:
        file_AR_df_feature = 'data/AR_full_'+properties+'_'+start_time+'_'+end_time+'/AR_'+str(it)+'_feature_'+properties+'_'+start_time+'_'+end_time+'.pkl'
        file_AR_df_label = 'data/AR_full_'+properties+'_'+start_time+'_'+end_time+'/AR_' + str(it) + '_label_'+flare_class +'_'+properties+'_'+start_time+'_'+end_time+'.pkl'

        df_AR_feature = pandas.DataFrame()
        df_AR_label = pandas.DataFrame()

        df_AR_feature = pandas.concat([df_AR_feature, df_feature_full.ix[it]],axis=0)
        df_AR_label = pandas.concat([df_AR_label, df_label_full.ix[it]], axis=0)

        df_AR_feature.to_pickle(file_AR_df_feature)
        df_AR_label.to_pickle(file_AR_df_label)







