import pickle
import pandas
import numpy

df_topo = pandas.read_csv('data/indici_RD.csv', delimiter=',')

df = pickle.load(open('data/features_2012_2015.pkl'))

df_training = df.iloc[0:24095]
output = open('data/features_2012_2014.pkl', 'wb')
pickle.dump(df_training,output)


df_testing = df.iloc[24095:df.shape[0]]
output = open('data/features_2015_2016.pkl', 'wb')
pickle.dump(df_testing,output)

R_topo = numpy.asarray(df_topo["R_topo"])
topo = numpy.asarray(df_topo["topo"])
df["R_topo"] = R_topo
df['topo'] = topo

df_training = df.iloc[0:24095]
output = open('data/features_with_topo_2012_2014.pkl', 'wb')
pickle.dump(df_training,output)


df_testing = df.iloc[24095:df.shape[0]]
output = open('data/features_with_topo_2015_2016.pkl', 'wb')
pickle.dump(df_testing,output)