import pandas
import json
import numpy
from Py_flarecast_learning_algorithms import classification_skills

flare_class_list = ('abovec', 'abovem', 'abovex')  #
issuing_time_list = ('00', '12', '18')
column_names = ["flaring", "fc_id", "predict"]
for it_list in flare_class_list:
    gt_class = pandas.DataFrame(columns = column_names)
    for it_issuing in issuing_time_list:
        flare_association = pandas.read_pickle('data/flare_association_HybridLasso_' + \
                                it_list + '_24_0_' + \
                                it_issuing + '_Bmix_all_withfh-predict.pkl')
        fc_id = pandas.read_pickle('data/fc_id_HybridLasso_' + \
                                it_list + '_24_0_' + \
                                it_issuing + '_Bmix_all_withfh-predict.pkl')
        predict = json.load(open('creazione_figura/predictions_old_ok/'
                                 'HybridLasso_' + \
                                it_list + '_6_0_' + \
                                it_issuing + '_Blos_all_nofh.json','r'))
        gt = flare_association.ix[5634]
        gt['fc_id'] = fc_id.ix[5634]
        esito = numpy.zeros((gt.shape[0],1))
        for it_id in range(gt.shape[0]):
            id  = gt['fc_id'][it_id]
            for it_p in range(len(predict['prediction_data'])):
                if predict['prediction_data'][it_p]['source_data:0'] == id:
                    esito[it_id,0] = predict['prediction_data'][it_p]['data:result']
        gt['predict'] = esito


        gt.to_csv('res/ground_truth_old/gt_6_0_' + it_list + '_' + it_issuing + '.csv', sep='\t')
        print('\n')
        print(it_list, ' ', it_issuing)
        print(classification_skills(numpy.asanyarray(gt['flaring'], dtype=int),
                                    numpy.asanyarray(gt['predict'])))

        gt_class = gt_class.append(gt)

    skill_scores = classification_skills(numpy.asanyarray(gt_class['flaring'],dtype=int), numpy.asanyarray(gt_class['predict']))
    cm = skill_scores[0]
    a = float(cm[1][1])
    d = float(cm[0][0])
    b = float(cm[0][1])
    c = float(cm[1][0])
    TP = a
    TN = d
    FP = b
    FN = c

    print('\n')
    print(it_list)
    print(skill_scores)
    gt_class.to_csv('res/ground_truth_old/gt_6_0_' + it_list + '.csv', sep='\t')