import flarecast_engine as fc
import pickle, json

XX = pickle.load(open('data/Bmix_cadence3h-train.pkl'))
YY = json.load(open('data/flare_association_Bmix_cadence3h-train.json'))


flare_class_list =('abovec','abovem','abovex')#( 'abovec','abovem','abovex')
window_list = ('6','12','24',)
issuing_time_list = ('00', '06', '12', '18')
fh = 'withfh'

caso = 'predict'

for it_list in flare_class_list:
    for it_issuing in issuing_time_list:
        for it_window in window_list:
            cfg_name = 'HybridLasso_' + it_list + '_' + it_window + '_0_' \
                       + it_issuing + '_Bmix_all_' + fh


            fc_dataset = fc.access_db(cfg_name + '-'+caso+'.json', offline=True)
            fc_dataset.load_dataset_3h('Bmix_cadence3h-' + caso, offline=True)

            fc_dataset.feature_df.to_pickle('data/'+cfg_name+'-' + caso + '.pkl')
            fc_dataset.label_df.to_pickle('data/flare_association_'+cfg_name+'-' + caso + '.pkl')
            fc_dataset.fc_id_df.to_pickle('data/fc_id_'+cfg_name+'-' + caso + '.pkl')
            fc_dataset.discarded_fc_id_df.to_pickle('data/discarded_fc_id_'+cfg_name+'-' + caso + '.pkl')


