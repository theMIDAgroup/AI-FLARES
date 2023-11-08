import flarecast_engine as fc

flare_class_list = ('abovem', 'abovec',)
window_list = ('24',)
issuing_time_list = ('00', '06', '12', '18')
fh = 'withfh'

for it_list in flare_class_list:
    for it_issuing in issuing_time_list:
        for it_window in window_list:
            cfg_name = 'HybridLasso_' + it_list + '_' + it_window + '_0_' \
                       + it_issuing + '_Bmix_all_' + fh + '-train.json'
            cfg_name_test = 'HybridLasso_' + it_list + '_' + it_window + '_0_' \
                            + it_issuing + '_Bmix_all_' + fh + '-predict.json'

            # train
            fc_dataset = fc.access_db(cfg_name, offline=True)
            fc_dataset.load_dataset_creazione_pkl(offline=True)
            fc_dataset.feature_df.to_pickle('data/feature_Bmix_' + it_issuing +
                                            'h_20120914_20160430_clean.pkl')
            fc_dataset.label_df.to_pickle('data/label_' + it_list +
                                          '_Bmix_' + it_issuing +
                                          'h_20120914_20160430_clean.pkl')

