import numpy
import pickle
import pandas as pd

issuing_time_list = ('00', '06', '12', '18')
window_list = ('6','12','24')
df = numpy.zeros([4,15])
c = 0
for issuing in issuing_time_list:
    for window in window_list:
        for it_train in ('train','test'):
            X_str = open('data/X_HybridLasso_abovex_'+window+'_0_'+issuing+'_Bmix_all-'+it_train+'.pkl')
            M_str = open('data/X_HybridLasso_abovem_'+window+'_0_'+issuing+'_Bmix_all-'+it_train+'.pkl')
            C_str = open('data/X_HybridLasso_abovec_'+window+'_0_'+issuing+'_Bmix_all-'+it_train+'.pkl')

            X = pickle.load(X_str)
            M = pickle.load(M_str)
            C = pickle.load(C_str)

            df[c, 0:9] = numpy.asanyarray([issuing, window, it_train,\
                len(X),len(M),len(C), \
                    sum(sum(numpy.asanyarray(X-M))), sum(sum(numpy.asanyarray(X-C))),\
                                       sum(sum(numpy.asanyarray(M-C)))])


            XL_str = open('data/Y_HybridLasso_abovex_'+window+'_0_'+issuing+'_Bmix_all-'+it_train+'.pkl')
            ML_str = open('data/Y_HybridLasso_abovem_'+window+'_0_'+issuing+'_Bmix_all-'+it_train+'.pkl')
            CL_str = open('data/Y_HybridLasso_abovec_'+window+'_0_'+issuing+'_Bmix_all-'+it_train+'.pkl')

            XL = pickle.load(XL_str)
            ML = pickle.load(ML_str)
            CL = pickle.load(CL_str)

            df[c, 6:12] = numpy.asanyarray([len(XL), len(ML), len(CL),\
                sum(sum(numpy.asanyarray(XL))),\
                                        sum(sum(numpy.asanyarray(ML))), \
                                        sum(sum(numpy.asanyarray(CL)))])
            c = c+1

pd.DataFrame(df).to_csv("check_data.csv", sep = '\t', index=False, header=False)