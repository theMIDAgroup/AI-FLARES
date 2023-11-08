# -*- coding: utf-8 -*-
'''
Created on Thu Dec 17 10:32:30 2015

@author: Federico Benvenuto & Annalisa Perasso
'''
import requests
import flatdict
import json
import numpy
import pandas
import logging
import pickle
import base64
import warnings
#import pandas.tools.util as tools
import sys
import os
from sys import exit

# PATCH to linear model
# Use the skl_dev version of scikit-learn for Weigthed (Poisson) Multi
# Task Regression
if os.path.isdir('./scikit-learn/build/'):
    if os.listdir('./scikit-learn/build/'):
        sys.path.insert(0, './scikit-learn')
else:
    warnings.warn(
        "Use the scikit-learn standard version: "
        "(Adaptive) Multi Task Poisson Lasso algorithm is not available",
        UserWarning,
        stacklevel=2)

# python setup.py build_ext --inplace

from operator import itemgetter
from itertools import compress
# from ConfigParserUtils import optionParser, separateDictionary
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegressionCV, \
    ElasticNetCV, ARDRegression, BayesianRidge, \
    HuberRegressor, Lars, LassoLars, LogisticRegression, OrthogonalMatchingPursuit, \
    SGDClassifier, SGDRegressor, TheilSenRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, LinearSVC
from sknn.mlp import Classifier as MLPerceptron
from sknn.mlp import Layer
import datetime
#from R_flarecast_learning_algorithms import R_nn, R_svm, R_rf, R_lm, R_probit, R_logit, R_svc, R_lda
#from R_flarecast_learning_algorithms import r_plot_ROC, r_plot_RD, r_plot_SSP
# from R_flarecast_learning_algorithms import training_set_standardization, testing_set_standardization
from Py_flarecast_learning_algorithms import HybridLogit, HybridLasso, SVR_CV, SVC_CV, MLPClassifier_HM, \
    MLPRegressor_HM, AdaptiveLasso_CV, HybridLassoL2
# from Py_flarecast_learning_algorithms import training_set_scaling, training_set_descaling,training_set_destandardization,testing_set_scaling

# from Py_flarecast_learning_algorithms import training_set_standardization, testing_set_standardization
from Py_flarecast_learning_algorithms import KMeans_HM, FKMeans_HM, SimAnnKMeans_HM, SimAnnFKMeans_HM, PKMeans_HM, \
    RandomForest
from Py_flarecast_learning_algorithms import MultiTaskLasso_CV, AdaptiveMultiTaskLasso_CV
from Py_flarecast_learning_algorithms import MultiTaskPoissonLasso_CV, AdaptiveMultiTaskPoissonLasso_CV

# from pandas import core as pan
from itertools import compress

FORMAT = '%(log_prefix)s %(levelname)s: %(asctime)s %(funcName)s at line %(lineno)d: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.level = logging.DEBUG

def drop_row(feature_dirty_df_pre_issuing,hour):
    aux = []
    hour_minus1 = str(int(hour)-1)
    if len(hour_minus1)==1:
        hour_minus1 = '0'+ hour_minus1
    for it in numpy.arange(feature_dirty_df_pre_issuing.shape[0]):
        if feature_dirty_df_pre_issuing.index[it][1][11:13] == hour or \
            feature_dirty_df_pre_issuing.index[it][1][11:13] ==   hour_minus1:
            aux.append(it)
    feature_dirty_df = feature_dirty_df_pre_issuing.ix[aux]
    return feature_dirty_df

def training_set_scaling(X_training):
    max_ = numpy.amax(X_training, axis=0)
    min_ = numpy.amin(X_training, axis=0)
    Xn_training = (X_training - min_) / (max_ - min_)
    return Xn_training, max_, min_


def training_set_descaling(X_training, max_, min_):
    return X_training * (max_ - min_) + min_


def training_set_standardization(X_training):
    # normalization / standardization
    mean_ = X_training.sum(axis=0) / X_training.shape[0]
    # Q = ((X_training - mean_) ** 2.).sum(axis=0) / X_training.shape[0]
    # std_ = numpy.array([numpy.sqrt(Q[i]) for i in range(Q.shape[0])])

    std_ = numpy.sqrt(
        (((X_training - mean_) ** 2.).sum(axis=0) / X_training.shape[0]))
    # Xn_training = scale(X_training)

    Xn_training = div0((X_training - mean_), std_)

    return Xn_training, mean_, std_


def training_set_destandardization(X_training, mean_, std_):
    if type(X_training) ==dict:
        for i in range(len(X_training)):
            X_training[str(i)] = X_training[str(i)]*std_ + mean_
        return X_training
    else:
        return X_training * std_ + mean_


def testing_set_standardization(X_testing, mean_, std_):
    Xn_testing = div0((X_testing - mean_), std_)

    return Xn_testing


def testing_set_scaling(X_training, max_, min_):
    """

    Parameters
    ----------
    X_training
    max_
    min_

    Returns
    -------

    """
    Xn_training = (X_training - min_) / (max_ - min_)
    return Xn_training


def div0(a, b):
    """
    ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]

    Parameters
    ----------
    a
    b

    Returns
    -------

    c

    """
    with numpy.errstate(divide='ignore', invalid='ignore'):
        c = numpy.true_divide(a, b)
        c[~ numpy.isfinite(c.tolist())] = 0  # -inf inf NaN
    return c


def download_range(
        service_url,
        dataset,
        questionable,
        start_questionable,
        end_questionable,
        start,
        end,
        cadence='6h',
        step=datetime.timedelta(
            days=1),
        **params):
    """
    service_url:    URL to get to the service. This is all the part before '/ui', e.g.
                    'http://cluster-r730-1:8002'
                    'http://api.flarecast.eu/property'
                    'http://localhost:8002'
                    Type: string
    dataset:        The dataset to download from
                    Type: string
    start, end:     Total start and end time of the data to download
                    Type: datetime
    step:           Time range of a single download slice
                    The total range (start - end) will be splitted up in smaller time ranges
                    with the size of 'step' and then every time range will be downloaded separately
                    Type: timedelta
    params:         Keyword argument, will be passed as query parameters to the http request url:
                    Examples:
                    property_type="sfunction_blos,sfunction_br"
                    nar=3120

    returns:        List with all entries, like you would download the whole time range in one request
                    Type: List of dicts
    """
    all_data = []

    while start < end:
        # params['property_type'] = params['property_type'] + ',flare_association'

        response = None
        end_step = min(start + step, end)
        ''' manage the questionable dataset. If questionable keyword is set to true, switch to questionable data '''
        running_dataset = dataset   # default dataset
        if questionable: # if we have to consider also a questionable dataset ... we have four cases
            ''' 4 cases
                SQ          EQ
            S       E                     case 1
            S                  E          case 2
                    S    E                case 3
                    S          E          case 4
            '''
            if start < start_questionable and end_step >= start_questionable:   # case 1, 2
                end_step = start_questionable
            if start >= start_questionable and start < end_questionable:        # case 3, 4
                running_dataset = questionable
                if end_step > end_questionable:                                # case 4
                    end_step = end_questionable
        params["time_start"] = "between(%s,%s)" % (
            start.isoformat(),
            end_step.isoformat()
        )
        try:
            response = requests.get(
                "%s/region/%s/list?&cadence=%s&exclude_higher_cadences=false&region_fields=*" %
                (service_url, running_dataset, cadence), params=params)
        except requests.exceptions.BaseHTTPError as ex:
            logger.error(
                "exception while downloading: " %
                ex, extra={
                    'log_prefix': 'FLARECAST-LOG'})

        if response is not None and response.status_code == 200:
            all_data.extend(response.json()["data"])
            logger.info(
                'from %s to %s: done' %
                (start, end), extra={
                    'log_prefix': 'FLARECAST-LOG'})
        else:
            resp_msg = response.json() if response is not None else ""
            logger.error(
                "error while downloading time range (%s - %s): %s" %
                (start,
                 start +
                 step,
                 resp_msg),
                extra={
                    'log_prefix': 'FLARECAST-LOG'})
        start = end_step

    return all_data


def download_range_ORIGINALE_E_PURE_SACRO(
        service_url,
        dataset,
        questionable,
        start_questionable,
        end_questionable,
        start,
        end,
        cadence='6h',
        step=datetime.timedelta(
            days=1),
        **params):
    """
    service_url:    URL to get to the service. This is all the part before '/ui', e.g.
                    'http://cluster-r730-1:8002'
                    'http://api.flarecast.eu/property'
                    'http://localhost:8002'
                    Type: string
    dataset:        The dataset to download from
                    Type: string
    start, end:     Total start and end time of the data to download
                    Type: datetime
    step:           Time range of a single download slice
                    The total range (start - end) will be splitted up in smaller time ranges
                    with the size of 'step' and then every time range will be downloaded separately
                    Type: timedelta
    params:         Keyword argument, will be passed as query parameters to the http request url:
                    Examples:
                    property_type="sfunction_blos,sfunction_br"
                    nar=3120

    returns:        List with all entries, like you would download the whole time range in one request
                    Type: List of dicts
    """
    all_data = []

    while start < end:
        # params['property_type'] = params['property_type'] + ',flare_association'

        response = None
        end_step = min(start + step, end)
        try:
            params["time_start"] = "between(%s,%s)" % (
                start.isoformat(),
                end_step.isoformat()
            ),
            if questionable != False and start>=start_questionable and end>=end_questionable:
                dataset = questionable
            else:
                dataset = dataset

            response = requests.get(
                "%s/region/%s/list?&cadence=%s&exclude_higher_cadences=true&region_fields=*" %
                (service_url, dataset, cadence), params=params)
        except requests.exceptions.BaseHTTPError as ex:
            logger.error(
                "exception while downloading: " %
                ex, extra={
                    'log_prefix': 'FLARECAST-LOG'})

        if response is not None and response.status_code == 200:
            all_data.extend(response.json()["data"])
            logger.info(
                'from %s to %s: done' %
                (start, end), extra={
                    'log_prefix': 'FLARECAST-LOG'})
        else:
            resp_msg = response.json() if response is not None else ""
            logger.error(
                "error while downloading time range (%s - %s): %s" %
                (start,
                 start +
                 step,
                 resp_msg),
                extra={
                    'log_prefix': 'FLARECAST-LOG'})
        start += step

    return all_data


class access_db:  # Read only(erase the write), reads the records from the database

    def __init__(self, file_name, offline=False):

        # create a file handler
        handler = logging.FileHandler('log/%s.log' % file_name)
        # handler.setLevel(logging.INFO)
        logger = logging.getLogger(__name__)
        # add the handlers to the logger
        logger.addHandler(handler)
        # initialize json file - very ugly coding: FLARECAT-LOG is carved into
        # this file
        logger.info(
            'Start reading json file: %s' %
            file_name, extra={
                'log_prefix': 'FLARECAST-LOG'})
        self.read_json('cfg/%s' % file_name, offline=offline)
        # self.read_json(file_name)
        logger.info('End reading json file: %s' % file_name, extra=self.prefix)

    def read_json(self, json_file, offline=False):
        # read json file
        with open(json_file) as params_file:
            data = json.load(params_file)
            # display just lines containing the FLARECAST-LOG prefix
            self.prefix = {'log_prefix': data["services"]['log_prefix']}
            logger.debug('json file data: %s' % data, extra=self.prefix)
            try:
                if data['algorithm']['phase'] == 'training':
                    self.read_training(data)
                elif data['algorithm']['phase'] == 'execution':
                    self.read_testing(data, offline=offline)
            except KeyError:
                self.read_training(data)

    def read_training(self, data):
        # read address
        self.write_addr = data['services']['property']['write']
        self.read_addr = data['services']['property']['read']
        self.cadence = data['dataset']['cadence']
        self.run_id = data['runtime']['run_id']
        self.token_type = data['runtime'].get('token_type', None)
        self.access_token = data['runtime'].get('access_token', None)
        self.address = 'http://%s:%s' % (
            self.read_addr['host'], self.read_addr['port'])

        self.prediction_address = 'http://%s:%s' % (
            data['services']['prediction']['read']['host'], data['services']['prediction']['read']['port'])
        # read which dataset
        # list(d.keys())[list(d.values()).index(True)]
        self.dataset = data["dataset"]["name"]
        # try for questionable dataset
        self.questionable = False
        self.dstart_questionable = '2016-04-13T00:00:00Z'
        self.dend_questionable = '2017-09-07T00:00:00Z'
        try:
            self.questionable = data["dataset"]["questionable"]["name"]
            self.dstart_questionable = data['dataset']['questionable']['start_time']
            self.dend_questionable = data['dataset']['questionable']['end_time']
        except BaseException:
            pass
        # read training set
        self.dstart = data['dataset']['time_interval']['start_time']
        self.dend = data['dataset']['time_interval']['end_time']
        # read flare type
        self.flare_class = data['flare']['class']
        self.flare_class_max = data['flare'].get('class_max', 'X9.9')
        self.flare_window = data['flare']['window']
        self.latency = data['flare'].get('latency', 0)
        self.issuing = data['flare'].get('issuing', '00')
        self.algorithm_info = data['algorithm']
        self.phase = data['algorithm']["phase"]
        self.config_name = data['algorithm']["config_name"]
        self.algo_descr = data['algorithm']['description']
        self.flare_history_window = data.get('flare_history_window', 24)

        # add self.issuing_cadence variable
        if self.issuing == '00':
            self.issuing_cadence = '24h'
        elif self.issuing == '12':
            self.issuing_cadence = '12h'
        elif self.issuing == '06' or self.issuing == '18':
            self.issuing_cadence = '6h'
        elif self.issuing == '03' or self.issuing == '21' \
                or self.issuing == '09' or self.issuing == '15':
            self.issuing_cadence = '3h'
        else:
            warnings.warn("check the issuing variable value",
                          UserWarning, stacklevel=2)
            exit(50)

        d = data["dataset"]['type']
        self.datatype = list(d.keys())[list(d.values()).index(True)]

        # add spatial information to features according to json
        # try:
        #     self.add_time_features = data['time_features']
        # except:
        self.add_time_features = False

        # add spatial information to features according to json
        self.add_spatial_features = []
        if data['spatial_features'] is True:
            if data['lat_hg'] is True:
                self.add_spatial_features.append('lat_hg')
            if data['long_hg'] is True:
                self.add_spatial_features.append('long_hg')
        else:
            self.add_spatial_features = False
        # read which labels
        self.labels = []
        datal = data['labels']
        for entry in datal:
            if datal[entry]:
                self.labels.append(entry)

        d2 = flatdict.FlatDict(data['properties'], delimiter='/')
        self.vector_features = flatdict.FlatDict({}, delimiter='/')
        for key,item in d2.items():
            if type(item) ==int:
                self.vector_features[key] = item

        self.properties = list(compress(d2.keys(), d2.values()))

        # for entry in self.algorithm_info:
        #    if self.algorithm_info[entry]:
        #        self.algo_name = entry
        self.algo_name = self.algorithm_info['config_name']

        # add flare history list if configured in the json file
        self.add_flare_history = []
        try:
            flare_history_features = data['flare_history_features']
            for entry in flare_history_features:
                if isinstance(flare_history_features[entry], bool):
                    if flare_history_features[entry]:
                        self.add_flare_history.append(entry)
        except BaseException:
            pass

        # Property string. It is not used anymore in the query
        self.properties_string = (','.join(self.properties))


        self.features = list(
            compress(
                self.properties, [
                    'flare_association' not in s for s in self.properties]))

        # info stored in the trained model output
        self.json_data = {'dataset': {'name': self.dataset,
                                      'time_interval': {'start_time': self.dstart,
                                                        'end_time': self.dend},
                                      'type': self.datatype,
                                      'cadence': self.cadence},
                          'features': {  # 'list': self.features,
            # 'time_features':self.add_time_features,
            'spatial_features': self.add_spatial_features},  # TODO add past flares
            'labels': {'list': self.labels,
                       'flare_type': {'class': self.flare_class,
                                      'class_max': self.flare_class_max,
                                      'window': self.flare_window,
                                      'latency': self.latency,
                                      'issuing':self.issuing}},
            'algorithm': {'params': self.algorithm_info['parameters'],
                          'name': self.algo_name},
            #'properties_string': self.properties_string,
            'properties': self.properties,
            'flare_history': self.add_flare_history,
            'flare_history_window' : self.flare_history_window,
            'vector_features' : self.vector_features.as_dict()
        }

    def read_testing(self, data, offline=False):
        self.phase = data['algorithm']["phase"]
        self.config_name = data['algorithm']["config_name"]
        self.algo_descr = data['algorithm']['description']
        self.write_addr = data['services']['property']['write']
        self.read_addr = data['services']['property']['read']
        self.run_id = data['runtime']['run_id']
        self.token_type = data['runtime'].get('token_type', None)
        self.access_token = data['runtime'].get('access_token', None)
        # read testing dataset
        self.data_testing = data["dataset"]
        self.dstart = self.data_testing['time_interval']['start_time']
        self.dend = self.data_testing['time_interval']['end_time']
        # try for questionable dataset
        self.questionable = False
        self.dstart_questionable = '2016-04-13T00:00:00Z'
        self.dend_questionable = '2017-09-07T00:00:00Z'
        try:
            self.questionable = data["dataset"]["questionable"]["name"]
            self.dstart_questionable = data['dataset']['questionable']['start_time']
            self.dend_questionable = data['dataset']['questionable']['end_time']
        except BaseException:
            pass
        # extract datatype
        tmp = self.data_testing['type']
        self.datatype = list(compress(tmp.keys(), tmp.values()))[0]

        self.prediction_address = 'http://%s:%s' % (data['services']['prediction']['read']['host'],
                                                    data['services']['prediction']['read']['port'])

        # load self.data_training containing the information on the used
        # training set
        self.algo = self.read_model(offline=offline)
        self.algo_type = self.data_training['algorithm']['type']
        self.properties = self.data_training['properties']
        self.address = 'http://%s:%s' % (
            self.read_addr['host'], self.read_addr['port'])
        self.cadence = self.data_training['dataset']['cadence']
        self.dataset = self.data_training['dataset']['name']
        # read flare type
        self.flare_class = self.data_training['labels']['flare_type']['class']
        self.flare_class_max = self.data_training['labels']['flare_type']['class_max']
        self.flare_window = self.data_training['labels']['flare_type']['window']
        self.latency = self.data_training['labels']['flare_type']['latency']
        self.issuing = self.data_training['labels']['flare_type']['issuing']

        if self.issuing == '00':
            self.issuing_cadence = '24h'
        elif self.issuing == '12':
            self.issuing_cadence = '12h'
        elif self.issuing == '06' or self.issuing == '18':
            self.issuing_cadence = '6h'
        elif self.issuing == '03' or self.issuing == '09' \
                or self.issuing == '15' or self.issuing == '21':
            self.issuing_cadence = '3h'
        else:
            warnings.warn("check the issuing variable value",
                          UserWarning, stacklevel=2)
            exit(50)
        # add spatial information to features according to json
        try:
            self.add_time_features = self.data_training['features']['time_features']
        except BaseException:
            self.add_time_features = False
        # add spatial information to features according to json
        try:
            self.add_spatial_features = self.data_training['features']['spatial_features']
        except BaseException:
            self.add_spatial_features = False
        # read which labels
        self.labels = self.data_training['labels']['list']
        #self.properties_string = self.data_training['properties_string']
        #self.features = self.data_training['features']['list']
        #self.features = list(
        #    compress(
        #        self.properties, [
        #            'flare_association' not in s for s in self.properties]))
        self.features = self.data_training['active_features']
        self.vector_features = flatdict.FlatDict(self.data_training['vector_features'], delimiter='/')
        self.add_flare_history = self.data_training['flare_history']
        self.flare_history_window = self.data_training['flare_history_window']
        self.algorithm_info = {
            'parameters': {
                'preprocessing': self.data_training['algorithm']['params']['preprocessing']}}

    def load_dataset_3h(self, name, offline=False):
        if not offline:
            # read features (diirectly stored in dataframes (l.0 level 0, i.e.
            # dirty data) )
            # read flare_association (each one is a list of dictionaries)
            self.read_properties()
            # save self.features_dirty_df
            # save self.flare_association
            self.write_property_files(name)
        else:
            self.read_property_files(name)
        # Build the label dataframe and complete the feature dataframe ()
        self.build_dataset()
        # Clean point in time (l.1 level 1, i.e. clean data)
        self.load()
        # create X,Y and Xs and Ys
        self.build_arrays()

    def load_dataset_pickle(self, name, offline=False):

        self.feature_df = pandas.read_pickle(
            'data/' + name + '.pkl')
        self.label_df = pandas.read_pickle(
            'data/flare_association_' + name + '.pkl')
        self.fc_id_df = pandas.read_pickle(
            'data/fc_id_' + name + '.pkl')
        self.discarded_fc_id_df = pandas.read_pickle(
            'data/discarded_fc_id_' + name + '.pkl')
        #self.build_dataset()
        # Clean point in time (l.1 level 1, i.e. clean data)
        #self.load()
        # create X,Y and Xs and Ys
        self.active_features = self.feature_df.columns

        self.build_arrays()



    def load_dataset(self, name,offline=False):
        if not offline:
            # read features (diirectly stored in dataframes (l.0 level 0, i.e.
            # dirty data) )
            # read flare_association (each one is a list of dictionaries)
            self.read_properties()
            # save self.features_dirty_df
            # save self.flare_association
            self.write_property_files(name)
        else:
            self.read_property_files(name)
        # Build the label dataframe and complete the feature dataframe ()
        self.build_dataset()
        # Clean point in time (l.1 level 1, i.e. clean data)
        self.load()
        # create X,Y and Xs and Ys
        self.build_arrays()

    def write_property_files(self,name):
        if not os.path.exists('./data'):
            os.makedirs('./data')

        if self.phase == 'training':
            self.feature_dirty_df.to_pickle('data/'+name+'.pkl')
            # self.feature_dirty_df.to_pickle('data/features_2012_2016.pkl')
            # numpy.save('data/features_2013_2014.npy', self.feature_dirty_df)
            json.dump(
                self.flare_association,
                open(
                    'data/flare_association_' +name + # str(self.flare_class) +
                    '.json',
                    'w'))
        # 'data/flare_association_2012_2016' +  # str(self.flare_class) +

        elif self.phase == 'execution':
            self.feature_dirty_df.to_pickle('data/'+name+'.pkl')
            # self.feature_dirty_df.to_pickle('data/features_2017_09_event.pkl')
            # numpy.save('data/features_2015.npy', self.feature_dirty_df)
            json.dump(
                self.flare_association,
                open(
                    'data/flare_association_' +name + # str(self.flare_class) +
                    '.json',
                    'w'))

    # 'data/flare_association_2017_09_event' +  # str(self.flare_class) +

    def read_property_files(self, name):
        if self.phase == 'training':
            self.feature_dirty_df = pandas.read_pickle(
                'data/'+ name +'.pkl')
            # self.feature_dirty_df = numpy.load('data/features_2013_2014.npy').item()
            self.flare_association = json.load(open(
                'data/flare_association_'+ name +'.json','r'))
        elif self.phase == 'execution':
            self.feature_dirty_df = pandas.read_pickle(
                'data/'+ name +'.pkl')
            # self.feature_dirty_df = numpy.load('data/features_2015.npy').item()
            self.flare_association = json.load(open(
                'data/flare_association_'+ name +'.json','r'))

        '''
            just for 2013 - 2014 and 2015 as testing set

        Returns
        -------

        X_2013 = numpy.loadtxt('data/X_2013_new.txt')
        X_2014 = numpy.loadtxt('data/X_2014_new.txt')
        X_2013_2014 = numpy.concatenate((X_2013,X_2014))
        feature_names = numpy.load('data/features_2013.npy')
        self.feature_df = pandas.DataFrame(X_2013_2014,columns=feature_names)

        Y_2013 = numpy.loadtxt('data/Y_'+flare_class+'_2013_new.txt')
        Y_2014 = numpy.loadtxt('data/Y_'+flare_class+'_2014_new.txt')
        Y_2013_2014 = numpy.concatenate((Y_2013,Y_2014))
        self.label_df = pandas.DataFrame(Y_2013_2014,columns=[flare_class])
        '''

    def read_properties(self):
        """

        Parameters
        ----------
        self.dstart
        self.dend
        self.features
        self.prefix
        self.address
        self.dataset
        self.cadence

        Returns
        -------
        self.feature_dirty_df
        self.flare_association

        """

        start = datetime.datetime.strptime(self.dstart, '%Y-%m-%dT%H:%M:%SZ')
        end = datetime.datetime.strptime(self.dend, '%Y-%m-%dT%H:%M:%SZ')
        start_questionable = datetime.datetime.strptime(self.dstart_questionable, '%Y-%m-%dT%H:%M:%SZ')
        end_questionable = datetime.datetime.strptime(self.dend_questionable, '%Y-%m-%dT%H:%M:%SZ')
        logger.info('Start reading %d features : from %s to %s' %
                    (len(self.features), start, end), extra=self.prefix)

        #########################################
        # READ FEATURES
        #########################################
        self.feature_dirty_df = pandas.DataFrame()
        n_features = 50
        si = 0
        ei = min(n_features, len(self.features))
        count = 0

        while si <= len(self.features):
            logger.info('Start reading features from %d to %d' %
                        (si + 1, min(ei, len(self.features))), extra=self.prefix)
            feature_string = (','.join(self.features[si:ei]))

            tmp_data = download_range(
                self.address,
                self.dataset,
                self.questionable,
                start_questionable,
                end_questionable,
                start,
                end,
                cadence=self.issuing_cadence,
                property_type=feature_string)
            logger.info('Reading features from %d to %d: done' %
                        (si + 1, min(ei, len(self.features))), extra=self.prefix)
            count = 0
            for entry in tmp_data:
                entry_flat = flatdict.FlatDict(entry,delimiter='/')
                for key, item in self.vector_features.items():
                    if key in entry_flat['data']:
                        entry_flat['data'][key] = entry_flat['data'][key][item]
                        tmp_data[count] = entry_flat.as_dict()
                count = count +1


            tmp_feature_df = self.dict_to_df(tmp_data)
            # remove point-in-time with wrong issuing
            # tmp_feature_df = drop_row(tmp_feature_df, self.issuing) TOLTOOOOOOOOOO

            self.feature_dirty_df = pandas.concat(
                [self.feature_dirty_df, tmp_feature_df], axis=1)
            count = count + 1
            si = ei
            ei += n_features
        #########################################

        #########################################
        # READ FLARE ASSOCIATION
        # self.read_flare_association()
        logger.info(
            'Start reading flare association: from %s to %s' %
            (start, end), extra=self.prefix)
        self.flare_association = download_range(
            self.address,
            self.dataset,
            self.questionable,
            start_questionable,
            end_questionable,
            start,
            end,
            cadence=self.issuing_cadence,
            property_type='flare_association')
        #self.flare_association = drop_row(tmp_flare_association, self.issuing)
        logger.info(
            'Reading flare association: from %s to %s : done' %
            (start, end), extra=self.prefix)


    def build_dataset(self):

        # tmp_feature_df = drop_row(tmp_feature_df, self.issuing) TOLTOOOOOOOOOO
        self.feature_dirty_df = drop_row(self.feature_dirty_df, self.issuing)

        #########################################
        # ADD FC_ID
        #########################################
        tmp_additional_fc_id = self.dict_to_fc_id(self.flare_association)
        self.additional_fc_id  = drop_row(tmp_additional_fc_id, self.issuing)
        self.feature_dirty_df = pandas.concat(
            [self.feature_dirty_df, self.additional_fc_id], axis=1)

        #########################################
        # LABEL DF
        #########################################
        tmp_label_dirty_df = self.dict_to_dl(self.flare_association)
        self.label_dirty_df = drop_row(tmp_label_dirty_df, self.issuing)

        compact_df = pandas.concat(
            [self.feature_dirty_df, self.label_dirty_df], axis=1)

        # clean data frame

        sum_label_values = compact_df[self.label_dirty_df.keys()].sum(
            axis=1)  # sum axis 1 per il multitask

        try:
            not_nan_on_rows = (~sum_label_values.isnull()).values
            self.feature_dirty_df = self.feature_dirty_df.ix[not_nan_on_rows[:, 0]]
        except IndexError:
            sum_label_values = compact_df[self.label_dirty_df.keys()]
            not_nan_on_rows = (~sum_label_values.isnull()).values
            self.feature_dirty_df = self.feature_dirty_df.ix[not_nan_on_rows[:, 0]]

        #########################################
        # ADD TO FEATURE DF
        #########################################

        if self.add_spatial_features or self.add_flare_history:
            tmp_additional_info_df = self.dict_to_additional_info_df(
                self.flare_association)
            self.additional_info_df = drop_row( tmp_additional_info_df, self.issuing)
            self.feature_dirty_df = pandas.concat(
                [self.feature_dirty_df, self.additional_info_df], axis=1)

    def dict_to_dl(self, tmp_data):
        """
        Create a panel data stored as a 3-dimensional matrix. Here both the
        properties and the labels are included with the methods append_properties
        and append_labels (you can find them below).

        Returns
        -------

        """
        logger.info('Start creating label dataframe', extra=self.prefix)

        # initialization
        feature_dict = {}
        label_dict = {}
        i = 0
        for entry in tmp_data:
            try:  # check if harp number exists
                harp = entry['meta']['harp']
                nar = entry['meta']['nar']
                check_data = entry['data']['flare_association']
            except KeyError:
                continue
            if check_data is None:
                continue
            if nar is None:  # if nar is null in the json (None for python)
                continue
            date = entry['time_start']

            # add properties
            i = i + 1

            #self.append_features_dataframe(feature_dict, harp, date, entry)
            self.append_labels_dataframe(label_dict, harp, date, entry)

        # feature_reform = {(innerKey,outerKey): values for outerKey, innerDict in feature_dict.iteritems() for
        #                  innerKey, values in innerDict.iteritems()}
        try:  # Python 3 compatibility
            label_reform = {(innerKey,
                             outerKey): values for outerKey,
                            innerDict in label_dict.iteritems() for innerKey,
                            values in innerDict.iteritems()}
        except AttributeError:
            label_reform = {(innerKey,
                             outerKey): values for outerKey,
                            innerDict in label_dict.items() for innerKey,
                            values in innerDict.items()}

        # feature_dirty_df = pandas.DataFrame(feature_reform).transpose()
        label_dirty_df = pandas.DataFrame(label_reform).transpose()

        logger.info('End creating label dataframe', extra=self.prefix)

        return label_dirty_df

    def dict_to_df(self, tmp_data):
        """
        Create a panel data stored as a 3-dimensional matrix. Here both the
        properties and the labels are included with the methods append_properties
        and append_labels (you can find them below).

        Returns
        -------

        """

        logger.info('Start creating feature dataframe', extra=self.prefix)

        # initialization
        feature_dict = {}
        label_dict = {}
        i = 0
        for entry in tmp_data:
            try:  # check if harp number exists
                harp = entry['meta']['harp']
                nar = entry['meta']['nar']
                #check_data = entry['data']['flare_association']
            except KeyError:
                continue
            # if check_data is None:
            #    continue
            if nar is None:  # if nar is null in the json (None for python)
                continue
            date = entry['time_start']

            # add properties
            i = i + 1

            self.append_features_dataframe(feature_dict, harp, date, entry)
        try:  # Python 3 compatibility TODO: mettere if python 2 o 3 !!!
            feature_reform = {(innerKey,
                               outerKey): values for outerKey,
                              innerDict in feature_dict.iteritems() for innerKey,
                              values in innerDict.iteritems()}
        except AttributeError:
            feature_reform = {(innerKey,
                               outerKey): values for outerKey,
                              innerDict in feature_dict.items() for innerKey,
                              values in innerDict.items()}

        feature_dirty_df = pandas.DataFrame(feature_reform).transpose()

        logger.info('End creating feature dataframe', extra=self.prefix)

        return feature_dirty_df

    def dict_to_fc_id(self, tmp_data):
        """
         Create a panel data stored as a 3-dimensional matrix. Here  the
         fc_id are included with the methods append_properties
         and append_labels (you can find them below).

         Returns
         -------

         """

        logger.info(
            'Start adding fc_id to feature dataframe',
            extra=self.prefix)

        # initialization
        feature_dict = {}
        i = 0
        for entry in tmp_data:
            try:  # check if harp number exists
                harp = entry['meta']['harp']
                nar = entry['meta']['nar']
                # check_data = entry['data']['flare_association']
            except KeyError:
                continue
            # if check_data is None:
            #    continue
            if nar is None:  # if nar is null in the json (None for python)
                continue
            date = entry['time_start']

            # add properties
            i = i + 1

            self.append_fc_id_to_features(feature_dict, harp, date, entry)


        try:  # Python 3 compatibility TODO: mettere if python 2 o 3 !!!
            feature_reform = {(innerKey,
                               outerKey): values for outerKey,
                              innerDict in feature_dict.iteritems() for innerKey,
                              values in innerDict.iteritems()}
        except AttributeError:
            feature_reform = {(innerKey,
                               outerKey): values for outerKey,
                              innerDict in feature_dict.items() for innerKey,
                              values in innerDict.items()}

        feature_dirty_df = pandas.DataFrame(feature_reform).transpose()

        logger.info('End adding fc_id to feature dataframe', extra=self.prefix)

        return feature_dirty_df

    def dict_to_additional_info_df(self, tmp_data):
        """
        Create a panel data stored as a 3-dimensional matrix. Here both the
        properties and the labels are included with the methods append_properties
        and append_labels (you can find them below).

        Returns
        -------

        """
        logger.info(
            'Start adding info to feature dataframe',
            extra=self.prefix)

        # initialization
        feature_dict = {}
        label_dict = {}
        i = 0
        for entry in tmp_data:
            try:  # check if harp number exists
                harp = entry['meta']['harp']
                nar = entry['meta']['nar']
                check_data = entry['data']['flare_association']
            except KeyError:
                continue
            if check_data is None:
                continue
            if nar is None:  # if nar is null in the json (None for python)
                continue
            date = entry['time_start']

            # add properties
            i = i + 1

            if self.add_flare_history:
                self.append_past_info_to_features(
                    feature_dict, harp, date, entry)
            if self.add_spatial_features:
                self.append_spatial_info_to_features(
                    feature_dict, harp, date, entry)
        try:  # Python 3 compatibility
            feature_reform = {(innerKey,
                               outerKey): values for outerKey,
                              innerDict in feature_dict.iteritems() for innerKey,
                              values in innerDict.iteritems()}
        except AttributeError:
            feature_reform = {(innerKey,
                               outerKey): values for outerKey,
                              innerDict in feature_dict.items() for innerKey,
                              values in innerDict.items()}
        feature_dirty_df = pandas.DataFrame(feature_reform).transpose()

        logger.info('End adding info to feature dataframe', extra=self.prefix)

        return feature_dirty_df

    def load(self):
        if self.datatype == 'point-in-time':
            return self.point_in_time()
        if self.datatype == 'longitudinal':
            return self.point_in_time()
        if self.datatype == 'time-series':
            return self.time_series()

    def time_series(self):
        return

    def build_arrays(self):

        self.X = numpy.array(self.feature_df, dtype=float)
        self.Y = numpy.array(self.label_df, dtype=float)

        self.Xs = self.X
        self.Ys = self.Y

        if self.phase == 'training':
            # standardization or scaling
            try:
                if self.algorithm_info['parameters']['preprocessing']['standardization_feature']:
                    self.Xs, self._mean_, self._std_ = training_set_standardization(
                        self.X)
                    if self.algorithm_info['parameters']['preprocessing']['scaling_feature']:
                        warnings.warn(
                            "Both standardization and scaling on the features are active , standardization has been used",
                            UserWarning,
                            stacklevel=2)
                elif self.algorithm_info['parameters']['preprocessing']['scaling']:
                    self.Xs, self._max_, self._min_ = training_set_scaling(self.X)

                if self.algorithm_info['parameters']['preprocessing']['standardization_label']:
                    self.Ys, self._Ymean_, self._Ystd_ = training_set_standardization(
                        self.Y)
                    if self.algorithm_info['parameters']['preprocessing']['scaling_feature']:
                        warnings.warn(
                            "Both standardization and scaling on the labels are active, standardization has been used",
                            UserWarning,
                            stacklevel=2)
                elif self.algorithm_info['parameters']['preprocessing']['scaling']:
                    self.Ys, self._Ymax_, self._Ymin_ = training_set_scaling(
                        self. Y)
            except KeyError:
                pass


    def point_in_time(self):
        """
        Create the dataframes X (which includes the properties) and Y (which includes the labels).

        Returns
        -------

        """
        # create a panel from the raw dict data structure
        logger.info('Feature panel size : %s', str(
            self.feature_dirty_df.shape), extra=self.prefix)
        logger.info('Label panel size : %s', str(
            self.label_dirty_df.shape), extra=self.prefix)

        # crete a multi-index dataframe for representing point-in-time data
        self.clean_dataframe()  # CLEAN

        logger.info('Feature dataframe size : %s', str(
            self.feature_df.shape), extra=self.prefix)
        logger.info('Label dataframe size : %s', str(
            self.label_df.shape), extra=self.prefix)


    def clean_dataframe(self):
        """
        Clean the feature and label dataframes.
        Cleaning means:
         1. remove nan columns (they correspond to features which have not calculated for the given time range)
         2. remove rows with at least a nan or a string (sometimes some feature calculation fails)
            2a. Count how many times it happens
         3. remove zero rows (maybe it does not happen anymore)
         4. REMOVE COLUMNS CONTAINING THE SAME VALUES
         5. SAVE WHICH FEATURES REMAIN AFTER CHECKING (THE SAME FEATURE WILL BE USED IN THE PREDICTION PHASE)

        This method is used in the class access_db

        Returns
        -------

        """
        logger.info('Start data cleaning', extra=self.prefix)

        # 0. PREPARE THE STRUCTURES (DATAFRAMES) TO BE CLEANED
        df = self.feature_dirty_df
        dl = self.label_dirty_df
        df_fc_id = pandas.DataFrame(df['fc_id'])
        df_discarded_fc_id = pandas.DataFrame([],columns = self.feature_dirty_df.columns)#self.feature_dirty_df.copy()

        # 1. REMOVE NAN COLUMNS
        not_nan_cols = df.columns[~(df.isnull().all(axis=0))]
        logger.debug(
            'Not all nan properties : \n%s',
            not_nan_cols,
            extra=self.prefix)
        df = df[not_nan_cols]
        df_discarded_fc_id = df_discarded_fc_id[not_nan_cols]
        # dataframe analysis

        # 2. REMOVE ROWS WITH AT LEAST A STRING TYPE
        # 2. REMOVE ROWS WITH AT LEAST A NAN
        if self.phase == 'execution':
            df = df.convert_objects(convert_numeric=True)
            df.fillna(0,inplace=True)

        """ BAD PATCH """
        not_nan_on_rows = ((~df.isnull()).all(axis=1)).values
        df_discarded_fc_id = pandas.concat((df_discarded_fc_id,df.ix[not_nan_on_rows == False]),axis=0)
        df = df.ix[not_nan_on_rows]
        # equivalent to :  df = df.dropna(0,'any') # remove rows with any NaN
        # values
        dl = dl.ix[not_nan_on_rows]
        df_fc_id = df_fc_id.ix[not_nan_on_rows]

        # filtered out rows containing strings or things that are not float
        aux_df_copy = df.copy()
        del aux_df_copy['fc_id']

        not_float_on_rows = ((aux_df_copy.applymap(numpy.isreal).all(axis=1))).values
        df_discarded_fc_id = pandas.concat((df_discarded_fc_id,df.ix[not_float_on_rows == False]),axis=0)

        df = df.ix[not_float_on_rows]
        dl = dl.ix[not_float_on_rows]
        df_fc_id = df_fc_id.ix[not_float_on_rows]

        self.nan_for_each_feature = df.isnull().sum()
        logger.debug(
            'NaN list for each feature : \n%s',
            self.nan_for_each_feature,
            extra=self.prefix)
        """ BAD PATCH """



        # 3. REMOVE ZERO ROWS
        not_0_rows = ~numpy.all(df == 0, axis=1)
        df_discarded_fc_id = pandas.concat((df_discarded_fc_id,df.ix[not_0_rows == False]),axis=0)

        df = df.ix[not_0_rows]
        dl = dl.ix[not_0_rows]
        df_fc_id = df_fc_id.ix[not_0_rows==True]

        del df['fc_id']

        # 4. REMOVE COLUMNS CONTAINING THE SAME VALUES
        if self.phase == 'training':
            aux_df = numpy.array(df, dtype=float)
            aux_label = []
            for it_key in range(aux_df.shape[1]):
                aux_std = numpy.std(aux_df[:, it_key])
                if numpy.isnan(aux_std) or aux_std == 0.0:
                    aux_label.append(df.columns[it_key])

            df = df.drop(aux_label, axis=1)

        # PATCH: MAYBE NO LONGER NEEDED AS
        # "filtered out rows containing strings or things that are not float"
        # IN CLEAN_DATAFRAME
        for item in df.keys():
            if item == 'gs_slf/g_s':
                #aux_gs_slf = df[item].tolist()
                for it in range(len(df)):
                    if df[item].iloc[it] == "Infinity" or df[item].iloc[it] == "-Infinity":
                        df[item].iloc[[it]] = 0.

        if self.datatype == 'longitudinal':
            df = self.add_longitudinal(df)

        self.active_features = df.columns
        self.feature_df = df
        self.label_df = dl
        self.fc_id_df = df_fc_id
        self.discarded_fc_id_df = df_discarded_fc_id

        # SAVE WHICH FEATURES REMAIN AFTER CHECKING (THE SAME FEATURE WILL BE USED IN THE PREDICTION PHASE)
        if self.phase == 'training':
            self.json_data['active_features'] = self.active_features.tolist()

            #for key, item in self.vector_features.items():
            #    self.json_data['vector_features'][key] = item
            # self.json_data['vector_features'] = self.vector_features.as_dict()
        if self.phase == 'execution':
            list_missing_features = []
            for it in numpy.arange(len(self.features)):
                if (self.active_features == self.features[it]).sum()==0:
                    list_missing_features.append(self.features[it])
            if list_missing_features != []:
                message_missing_features = ', '
                warnings.warn("different set of properties w.r.t. training set: " + message_missing_features.join(str(e) for e in list_missing_features),
                              UserWarning, stacklevel=2)
                exit(100)

        logger.info('End data cleaning', extra=self.prefix)

        # FEATURE EXTRACTION ALGORITHM VERIFICATION
        # number of FEAs run over HMI data in the required period = not_nan_columns
        # now, let's examine column by column
        vdf = self.feature_dirty_df
        # for each column
            # how many NaN, strings type (Infinity)
            # how many ZEROS


    def clean_dataframe_ORIGINALE_E_SACRO(self):
        """
        Clean the feature and label dataframes.
        Cleaning means:
         1. remove nan columns (they correspond to features which have not calculated for the given time range)
         2. remove rows with at least a nan or a string (sometimes some feature calculation fails)
            2a. Count how many times it happens
         3. remove zero rows (maybe it does not happen anymore)
         4. REMOVE COLUMNS CONTAINING THE SAME VALUES
         5. SAVE WHICH FEATURES REMAIN AFTER CHECKING (THE SAME FEATURE WILL BE USED IN THE PREDICTION PHASE)

        This method is used in the class access_db

        Returns
        -------

        """
        logger.info('Start data cleaning', extra=self.prefix)

        # 0. PREPARE THE STRUCTURES (DATAFRAMES) TO BE CLEANED
        df = self.feature_dirty_df
        dl = self.label_dirty_df
        df_fc_id = pandas.DataFrame(df['fc_id'])
        df_discarded_fc_id = pandas.DataFrame([],columns = self.feature_dirty_df.columns)#self.feature_dirty_df.copy()

        # 1. REMOVE NAN COLUMNS
        not_nan_cols = df.columns[~(df.isnull().all(axis=0))]
        logger.debug(
            'Not all nan properties : \n%s',
            not_nan_cols,
            extra=self.prefix)

        # 2. REMOVE ROWS WITH AT LEAST A NAN
        df = df[not_nan_cols]
        df_discarded_fc_id = df_discarded_fc_id[not_nan_cols]
        # dataframe analysis
        not_nan_on_rows = ((~df.isnull()).all(axis=1)).values
        df_discarded_fc_id = pandas.concat((df_discarded_fc_id,df.ix[not_nan_on_rows == False]),axis=0)
        df = df.ix[not_nan_on_rows]
        # equivalent to :  df = df.dropna(0,'any') # remove rows with any NaN
        # values
        dl = dl.ix[not_nan_on_rows]
        df_fc_id = df_fc_id.ix[not_nan_on_rows]

        # 2. REMOVE ROWS WITH AT LEAST A STRING TYPE
        # filtered out rows containing strings or things that are not float
        aux_df_copy = df.copy()
        del aux_df_copy['fc_id']

        not_float_on_rows = ((aux_df_copy.applymap(numpy.isreal).all(axis=1))).values
        df_discarded_fc_id = pandas.concat((df_discarded_fc_id,df.ix[not_float_on_rows == False]),axis=0)

        df = df.ix[not_float_on_rows]
        dl = dl.ix[not_float_on_rows]
        df_fc_id = df_fc_id.ix[not_float_on_rows]


        self.nan_for_each_feature = df.isnull().sum()
        logger.debug(
            'NaN list for each feature : \n%s',
            self.nan_for_each_feature,
            extra=self.prefix)

        # 3. REMOVE ZERO ROWS
        not_0_rows = ~numpy.all(df == 0, axis=1)
        df_discarded_fc_id = pandas.concat((df_discarded_fc_id,df.ix[not_0_rows == False]),axis=0)

        df = df.ix[not_0_rows]
        dl = dl.ix[not_0_rows]
        df_fc_id = df_fc_id.ix[not_0_rows==True]

        del df['fc_id']

        # 4. REMOVE COLUMNS CONTAINING THE SAME VALUES
        if self.phase == 'training':
            aux_df = numpy.array(df, dtype=float)
            aux_label = []
            for it_key in range(aux_df.shape[1]):
                aux_std = numpy.std(aux_df[:, it_key])
                if numpy.isnan(aux_std) or aux_std == 0.0:
                    aux_label.append(df.columns[it_key])

            df = df.drop(aux_label, axis=1)

        # PATCH: MAYBE NO LONGER NEEDED AS
        # "filtered out rows containing strings or things that are not float"
        # IN CLEAN_DATAFRAME
        for item in df.keys():
            if item == 'gs_slf/g_s':
                #aux_gs_slf = df[item].tolist()
                for it in range(len(df)):
                    if df[item].iloc[it] == "Infinity" or df[item].iloc[it] == "-Infinity":
                        df[item].iloc[[it]] = 0.

        if self.datatype == 'longitudinal':
            df = self.add_longitudinal(df)

        self.active_features = df.columns
        self.feature_df = df
        self.label_df = dl
        self.fc_id_df = df_fc_id
        self.discarded_fc_id_df = df_discarded_fc_id

        # SAVE WHICH FEATURES REMAIN AFTER CHECKING (THE SAME FEATURE WILL BE USED IN THE PREDICTION PHASE)
        if self.phase == 'training':
            self.json_data['active_features'] = self.active_features.tolist()

            #for key, item in self.vector_features.items():
            #    self.json_data['vector_features'][key] = item
            # self.json_data['vector_features'] = self.vector_features.as_dict()
        if self.phase == 'execution':
            list_missing_features = []
            for it in numpy.arange(len(self.features)):
                if (self.active_features == self.features[it]).sum()==0:
                    list_missing_features.append(self.features[it])
            if list_missing_features != []:
                message_missing_features = ', '
                warnings.warn("different set of properties w.r.t. training set: " + message_missing_features.join(str(e) for e in list_missing_features),
                              UserWarning, stacklevel=2)
                exit(100)

        logger.info('End data cleaning', extra=self.prefix)

        # FEATURE EXTRACTION ALGORITHM VERIFICATION
        # number of FEAs run over HMI data in the required period = not_nan_columns
        # now, let's examine column by column
        vdf = self.feature_dirty_df
        # for each column
            # how many NaN, strings type (Infinity)
            # how many ZEROS


    def add_longitudinal(self, df):

        df['harp'] = df.index.get_level_values(level=0)

        df['time1'] = 0
        j = 0
        for i in range(len(df['harp']) - 1):
            if df['harp'].iloc[i + 1] == df['harp'].iloc[i]:
                j = j + 1
            else:
                j = 0
            df['time1'].iloc[[i + 1]] = j

        '''
        temp_pan = pan.swapaxes(0, 2, copy=True)
        harp = temp_pan.axes[0]

        # dummy_column= df[df.axes[1][0:2]]
        # dummy_column.name='time1'#'['time1','harp']

        dummy_column = pandas.DataFrame(numpy.zeros((len(df), 2)), index=df.index, columns=['time1', 'harp'])

        df = pandas.concat([dummy_column, df], axis=1)

        df1 = pandas.DataFrame()

        for i in range(len(harp)):
            temp = df.loc(axis=0)[:, harp[i]]
            count = 0
            for j in range(len(temp.axes[0])):
                temp['time1'][j] = count
                temp['harp'][j] = harp[i]
                count = count + 1
            df1 = pandas.concat([df1, temp], axis=0)

        return df1
        '''

    def append_fc_id_to_features(self, dict_, harp, date, entry):
        if date not in dict_:
            dict_[date] = {}
        if harp not in dict_[date]:
            dict_[date][harp] = {}

        dict_[date][harp]['fc_id'] = entry['fc_id']

    def append_past_info_to_features(self, dict_, harp, date, entry):

        if date not in dict_:
            dict_[date] = {}
        if harp not in dict_[date]:
            dict_[date][harp] = {}

        add_flare_info = {'flare_past': self.flare_past,
                          'flare_index_past': self.flare_index_past}
        for flare_info in self.add_flare_history:
            dict_[date][harp][flare_info] = add_flare_info[flare_info](
                harp, date, entry)

    def append_spatial_info_to_features(self, dict_, harp, date, entry):

        if date not in dict_:
            dict_[date] = {}
        if harp not in dict_[date]:
            dict_[date][harp] = {}

        for space_info in self.add_spatial_features:
            dict_[date][harp][space_info] = entry[space_info]

    def append_features_dataframe(self, dict_, harp, date, entry):
        # Add properties to the the panel dict_
        # A dictionary object that allows
        flat = flatdict.FlatDict(entry['data'], delimiter='/')
        # for single level, delimited
        # value pair mapping of nested dictionaries.
        if date not in dict_:
            dict_[date] = {}
        if harp not in dict_[date]:
            dict_[date][harp] = {}

        for i_property in flat.keys():
            try:
                # return features (if it exists) collapsed into one dimension.
                #value_ = flat.get(self.features[i])
                value_ = flat[i_property]
            except KeyError:
                value_ = float('nan')
            dict_[date][harp][i_property] = value_

    def append_labels_dataframe(self, dict_, harp, date, entry):
        # Add labels to the dictionary (see also the next six methods)
        add_label = {'flare_index': self.flare_index,
                     'n_flare': self.n_flare,
                     'flaring': self.flaring,  # is there a flare?
                     'imminence': self.imminence,  # peak time
                     'flaring_ptime': self.flaring_ptime,  # peak time
                     'flaring_stime': self.flaring_stime,  # starting time
                     'flaring_etime': self.flaring_etime,  # ending time
                     'largest_flare': self.largest_flare,
                     'duration_flare': self.duration_flare,
                     'next_flare_class':self.next_flare_class}

        if date not in dict_:
            dict_[date] = {}
        if harp not in dict_[date]:
            dict_[date][harp] = {}

        for single_label in self.labels:
            dict_[date][harp][single_label] = add_label[single_label](
                harp, date, entry)

    # Note that flare_list can be a list or a dictionary. Therefore, for each of the
    # six following functions we need to treat the flare_list in different ways.
    # For each of the following six functions we consider a flare if is at least of
    # class flare_class (which is set in read_json) and if it happends in the
    # flare window (which is also set in read_json).

    def flaring(self, harp, date, entry):
        # check if there is at least a flare with magnitude in the interval
        # [flare_class, flare_clss_max] and peak time in flare_window.
        # Return 1 if there is such kind of flare,
        # otherwise 0.
        try:
            flare_list = entry["data"]["flare_association"]
        except KeyError:
            there_is_a_flare = float('NaN')
            return there_is_a_flare

        there_is_a_flare = 0  # inizialization
        if isinstance(flare_list, list):
            for flare in flare_list:
                # print(flare["f_mag"],convert(flare["f_mag"]))
                if (convert(flare["f_mag"]) >= convert(self.flare_class)) \
                    and (convert(flare["f_mag"]) <= convert(self.flare_class_max)) \
                        and (flare["f_stime_tau"] <= (self.flare_window + self.latency) * 3600) \
                        and (flare["f_stime_tau"] >= 0 + self.latency * 3600):
                    there_is_a_flare = 1
                    break
        elif isinstance(flare_list, dict):
            if (
                convert(
                    flare_list["f_mag"]) >= convert(
                    self.flare_class)) \
                    and (convert(flare_list["f_mag"]) <= convert(self.flare_class_max)) \
                    and (flare_list["f_stime_tau"] <= (self.flare_window + self.latency) * 3600) \
                    and (flare_list["f_stime_tau"] >= 0 + self.latency * 3600):
                there_is_a_flare = 1

        return there_is_a_flare

    def n_flare(self, harp, date, entry):
        # compute the number of flares which have at least magnitude "flare_class" and
        # have the peak time in the flare_window.
        flare_list = entry["data"]["flare_association"]
        n_flare = 0  # inizialization

        if isinstance(flare_list, list):
            for flare in flare_list:
                if (convert(flare["f_mag"]) >= convert(self.flare_class)) \
                        and (convert(flare["f_mag"]) <= convert(self.flare_class_max)) \
                        and (flare["f_stime_tau"] <= (self.flare_window + self.latency) * 3600) \
                        and (flare["f_stime_tau"] >= 0 + self.latency*3600):
                    n_flare += 1
        elif isinstance(flare_list, dict):
            if (
                convert(
                    flare_list["f_mag"]) >= convert(
                    self.flare_class)) \
                    and (convert(flare_list["f_mag"]) <= convert(self.flare_class_max)) \
                    and (flare_list["f_stime_tau"] <= (self.flare_window + self.latency) * 3600) \
                    and (flare_list["f_stime_tau"] >= 0 + self.latency * 3600):
                n_flare = 1

        return n_flare

    def flare_index(self, harp, date, entry):
        # compute the flare_index
        flare_list = entry["data"]["flare_association"]
        flare_index = 0  # inizialization
        # convert the flare_window (which is in hours) in days
        delta_t = self.flare_window / 24.

        if isinstance(flare_list, list):
            for flare in flare_list:
                if (convert(flare["f_mag"]) >= convert(self.flare_class)) \
                        and (convert(flare["f_mag"]) <= convert(self.flare_class_max)) \
                        and (flare["f_stime_tau"] <= (self.flare_window + self.latency) * 3600) \
                        and (flare["f_stime_tau"] >= 0 + self.latency * 3600):
                    flare_index = flare_index + convert(flare["f_mag"])
            flare_index = flare_index / delta_t
        elif isinstance(flare_list, dict):
            if (
                convert(
                    flare_list["f_mag"]) >= convert(
                        self.flare_class)) \
                    and (convert(flare_list["f_mag"]) <= convert(self.flare_class_max)) \
                    and (flare_list["f_stime_tau"] <= (self.flare_window + self.latency) * 3600) \
                    and (flare_list["f_stime_tau"] >= 0 + self.latency * 3600):
                flare_index = convert(flare_list["f_mag"]) / delta_t

        return flare_index

    def largest_flare(self, harp, date, entry):
        # compute the largest flare
        flare_list = entry["data"]["flare_association"]
        largest_flare = 0  # inizialization

        if isinstance(flare_list, list):
            for flare in flare_list:
                if (convert(flare["f_mag"]) >= convert(self.flare_class)) \
                    and (convert(flare["f_mag"]) <= convert(self.flare_class_max)) \
                        and (flare["f_stime_tau"] <= (self.flare_window + self.latency) * 3600) \
                        and (flare["f_stime_tau"] >= 0 + self.latency * 3600):
                    if convert(flare["f_mag"]) > largest_flare:
                        largest_flare = convert(flare["f_mag"])
        elif isinstance(flare_list, dict):
            if (
                convert(
                    flare_list["f_mag"]) >= convert(
                        self.flare_class)) \
                    and (convert(flare_list["f_mag"]) <= convert(self.flare_class_max)) \
                    and (flare_list["f_stime_tau"] <= (self.flare_window + self.latency) * 3600) \
                    and (flare_list["f_stime_tau"] >= 0 + self.latency * 3600):
                largest_flare = convert(flare_list["f_mag"])

        return largest_flare


    def next_flare_class(self, harp, date, entry):
        # compute the first occurring flare
        flare_list = entry["data"]["flare_association"]
        first_flare_time = self.flare_window * 3600  # inizialization
        next_flare_class = 0  # inizialization

        if isinstance(flare_list, list):
            for flare in flare_list:
                if (convert(flare["f_mag"]) >= convert(self.flare_class)) \
                    and (convert(flare["f_mag"]) <= convert(self.flare_class_max)) \
                        and (flare["f_stime_tau"] <= (self.flare_window + self.latency) * 3600) \
                        and (flare["f_stime_tau"] >= 0 + self.latency * 3600):
                    if flare["f_stime_tau"] < first_flare_time:
                        next_flare_class = convert(flare["f_mag"])
        elif isinstance(flare_list, dict):
            if (
                convert(
                    flare_list["f_mag"]) >= convert(
                        self.flare_class)) \
                    and (convert(flare_list["f_mag"]) <= convert(self.flare_class_max)) \
                    and (flare_list["f_stime_tau"] <= (self.flare_window + self.latency) * 3600) \
                    and (flare_list["f_stime_tau"] >= 0 + self.latency * 3600):
                next_flare_class = convert(flare_list["f_mag"])

        return next_flare_class


    def duration_flare(self, harp, date, entry):
        # compute the duration of the largest flare
        flare_list = entry["data"]["flare_association"]
        duration_flare = 0  # inizializations
        largest_flare = 0

        if isinstance(flare_list, list):
            for flare in flare_list:
                if (convert(flare["f_mag"]) >= convert(self.flare_class)) \
                        and (convert(flare["f_mag"]) <= convert(self.flare_class_max)) \
                        and (flare["f_stime_tau"] <= (self.flare_window + self.latency) * 3600) \
                        and (flare["f_stime_tau"] >= 0 + self.latency*3600):
                    if convert(flare["f_mag"]) > largest_flare:
                        largest_flare = convert(flare["f_mag"])
                        duration_flare = flare["f_etime_tau"] - \
                            flare["f_stime_tau"]
        elif isinstance(flare_list, dict):
            if (
                convert(
                    flare_list["f_mag"]) >= convert(
                        self.flare_class)) \
                    and (convert(flare_list["f_mag"]) <= convert(self.flare_class_max)) \
                    and (flare_list["f_stime_tau"] <= (self.flare_window + self.latency) * 3600) \
                    and (flare_list["f_stime_tau"] >= 0 + self.latency * 3600):
                duration_flare = flare_list["f_etime_tau"] - \
                    flare_list["f_stime_tau"]

        return duration_flare

    def flaring_stime(self, harp, date, entry):
        # compute the largest starting time of the flares that have at least
        # magnitude "flare_class" and have the starting time in the
        # flare_window.
        flare_list = entry["data"]["flare_association"]
        flaring_stime = 0  # inizialization
        largest_flare = 0

        if isinstance(flare_list, list):
            for flare in flare_list:
                if (convert(flare["f_mag"]) >= convert(self.flare_class)) \
                        and (convert(flare["f_mag"]) <= convert(self.flare_class_max)) \
                        and (flare["f_stime_tau"] <= (self.flare_window + self.latency) * 3600) \
                        and (flare["f_stime_tau"] >= 0 + self.latency * 3600):
                    if convert(flare["f_mag"]) > largest_flare:
                        largest_flare = convert(flare["f_mag"])
                        flaring_stime = flare["f_stime_tau"]
        elif isinstance(flare_list, dict):
            if (
                convert(
                    flare_list["f_mag"]) >= convert(
                        self.flare_class)) \
                    and (convert(flare_list["f_mag"]) <= convert(self.flare_class_max)) \
                    and (flare_list["f_stime_tau"] <= (self.flare_window + self.latency) * 3600) \
                    and (flare_list["f_stime_tau"] >= 0 + self.latency * 3600):
                flaring_stime = flare_list["f_stime_tau"]

        # print(flaring_ptime)

        return flaring_stime

    def flaring_ptime(self, harp, date, entry):
        # compute the largest peak time of the flares which have at least
        # magnitude "flare_class" and have the peak time in the flare_window.
        flare_list = entry["data"]["flare_association"]
        flaring_ptime = 0  # inizialization
        largest_flare = 0

        if isinstance(flare_list, list):
            for flare in flare_list:
                if (convert(flare["f_mag"]) >= convert(self.flare_class)) \
                        and (convert(flare["f_mag"]) <= convert(self.flare_class_max)) \
                        and (flare["f_ptime_tau"] <= (self.flare_window + self.latency) * 3600) \
                        and (flare["f_ptime_tau"] >= 0 + self.latency * 3600):
                    if convert(flare["f_mag"]) > largest_flare:
                        largest_flare = convert(flare["f_mag"])
                        flaring_ptime = flare["f_ptime_tau"]
        elif isinstance(flare_list, dict):
            if (
                convert(
                    flare_list["f_mag"]) >= convert(
                    self.flare_class)) \
                    and (convert(flare_list["f_mag"]) <= convert(self.flare_class_max)) \
                    and (flare_list["f_ptime_tau"] <= (self.flare_window + self.latency) * 3600) \
                    and (flare_list["f_ptime_tau"] >= 0 + self.latency * 3600):
                flaring_ptime = flare_list["f_ptime_tau"]
        return flaring_ptime

    def flaring_etime(self, harp, date, entry):
        # compute the largest ending time of the flares that have at least
        # magnitude "flare_class" and have the ending time in the flare_window.
        flare_list = entry["data"]["flare_association"]
        flaring_etime = 0  # inizialization
        largest_flare = 0

        if isinstance(flare_list, list):
            for flare in flare_list:
                if (convert(flare["f_mag"]) >= convert(self.flare_class)) \
                        and (convert(flare["f_mag"]) <= convert(self.flare_class_max)) \
                        and (flare["f_etime_tau"] <= (self.flare_window + self.latency) * 3600) \
                        and (flare["f_etime_tau"] >= 0 + self.latency * 3600):
                    if convert(flare["f_mag"]) > largest_flare:
                        largest_flare = convert(flare["f_mag"])
                        flaring_etime = flare["f_etime_tau"]
        elif isinstance(flare_list, dict):
            if (
                convert(
                    flare_list["f_mag"]) >= convert(
                    self.flare_class)) \
                    and (convert(flare_list["f_mag"]) <= convert(self.flare_class_max)) \
                    and (flare_list["f_etime_tau"] <= (self.flare_window + self.latency) * 3600) \
                    and (flare_list["f_etime_tau"] >= 0 + self.latency * 3600):
                flaring_etime = flare_list["f_etime_tau"]
        return flaring_etime

    def imminence(self, harp, date, entry):
        ptime = self.flaring_ptime(harp, date, entry)
        ret = 0
        if ptime > 0:
            ret = 1.0 / ( (ptime / 3600.) + 1.0)
        return ret

    def flare_past(self, harp, date, entry):
        flare_list = entry["data"]["flare_association"]
        flare_past = 0  # inizialization

        if isinstance(flare_list, list):
            for flare in flare_list:
                # print(flare["f_mag"],convert(flare["f_mag"]))
                # if there is a flare  and it happened in the past
                if (convert(flare["f_mag"]) > 0) \
                        and (flare["f_stime_tau"] <= 0) \
                        and (flare["f_etime_tau"] <=-1*self.flare_history_window*3600):
                    flare_past = 1
                    break
        elif isinstance(flare_list, dict):
            if (convert(flare_list["f_mag"]) > 0) \
                    and (flare_list["f_stime_tau"] <= 0) \
                        and (flare_list["f_etime_tau"] <=-1*self.flare_history_window*3600):
                flare_past = 1

        return flare_past

    def flare_index_past(self, harp, date, entry):
        flare_list = entry["data"]["flare_association"]
        flare_index_past = 0  # inizialization
        # delta_t=self.flare_window/24 # convert the flare_window (which is in
        # hours) in days

        if isinstance(flare_list, list):
            for flare in flare_list:
                max_stime = 0
                if (convert(flare["f_mag"]) > 0) \
                        and (flare["f_stime_tau"] <= 0)\
                        and (flare["f_etime_tau"] <=-1*self.flare_history_window*3600):
                    flare_index_past = flare_index_past + \
                        convert(flare["f_mag"])
                    if abs(flare["f_stime_tau"]) > max_stime:
                        max_stime = abs(flare["f_stime_tau"]) / 3600
            if max_stime > 0:
                flare_index_past = flare_index_past / max_stime
                # print(flare_index_past)
        elif isinstance(flare_list, dict):
            if (convert(flare_list["f_mag"]) > 0) \
                    and (flare_list["f_stime_tau"] <= 0)\
                    and (flare_list["f_etime_tau"] <=-1*self.flare_history_window*3600):
                flare_index_past = convert(
                    flare_list["f_mag"]) / abs(flare_list["f_stime_tau"])
                # print(flare_index_past)

        return flare_index_past

    def build_headers(self):
        if self.token_type is None or self.access_token is None:
            return {}
        else:
            return {
                "Authorization": "%s %s" % (self.token_type, self.access_token)
            }

    def write_model(self, algo, offline=False):

        logger.info('Start writing model', extra=self.prefix)
        self.json_data['active_features'] = self.active_features.tolist()
        s1 = base64.b64encode(pickle.dumps(algo, 2)).decode('ascii')
        # s1=pickle.dumps(algo,0).decode('ISO-8859-1')
        #        s1=s.decode("latin_1", "strict")
        d1 = {'data': s1, 'info': self.json_data}
        # we add  the corresponding names to the feature importance values in
        # the metrics dictionary
        if algo.name.startswith('R_nn') is False:
            if 'feature importance' in algo.estimator.metrics_training:
                # features_values = algo.estimator.metrics_training['feature importance']
                if hasattr(algo.estimator.estimator,'coef_'):
                    features_values = algo.estimator.estimator.coef_.transpose().tolist()
                elif hasattr(algo.estimator.estimator,'feature_importances_'):
                    features_values = algo.estimator.estimator.feature_importances_.transpose().tolist()
                algo.estimator.metrics_training['feature importance'] = dict(
                    zip(self.active_features, features_values))
        if 'r2' in algo.estimator.metrics_training:
            rse_values = algo.estimator.metrics_training['r2']
            algo.estimator.metrics_training['r2'] = dict(
                zip(self.labels, rse_values))
        if 'mse' in algo.estimator.metrics_training:
            rse_values = algo.estimator.metrics_training['mse']
            algo.estimator.metrics_training['mse'] = dict(
                zip(self.labels, rse_values))

        # insert the training_metrics into the algorithm results
        if self.algo_name.startswith('R_') is False:
            if len(self.label_df.columns) == 1:
                algo.estimator.metrics_training[self.label_df.columns[0]] = algo.estimator.metrics_training.pop('0')
                # FIX: [DV] algo.estimator.threshold is a scalar and therefore not indexable
                #algo.estimator.metrics_training[self.label_df.columns[0]]['threshold'] = \
                #    training_set_destandardization(algo.estimator.threshold['0'], self._Ymean_, self._Ystd_)[0]
                if hasattr(algo.estimator.metrics_training[self.label_df.columns[0]],'threshold'):
                    algo.estimator.metrics_training[self.label_df.columns[0]]['threshold'] = \
                        training_set_destandardization(algo.estimator.threshold, self._Ymean_, self._Ystd_)[0]
            else:
                for i in range(len(self.label_df.columns)):
                    # self.labels[i] "flaring", ... , "flare_duration"
                    algo.estimator.metrics_training[self.label_df.columns[i]] = algo.estimator.metrics_training.pop(str(i))
                    if hasattr(algo.estimator.metrics_training[self.label_df.columns[0]], 'threshold'):
                        algo.estimator.metrics_training[self.label_df.columns[i]]['threshold'] = \
                            training_set_destandardization(algo.estimator.threshold[str(i)], self._Ymean_[i], self._Ystd_[i])

        # TODO: check if needed
        algo.source_data_training = self.fc_id_df.to_dict(orient='record')
        algo.source_data_discarded_training = pandas.DataFrame(self.discarded_fc_id_df['fc_id']).to_dict(orient='record')



        post_data = {"algorithm_run_id": self.run_id,
                     # mt},
                     "config_data": {"configuration": d1, "metrics_training": algo.estimator.metrics_training,
                                     "source_data_fc_id": algo.source_data_training,
                                     "source_data_discarded_fc_id": algo.source_data_discarded_training},
                     "description": self.algo_descr}

        if offline:
            json.dump(post_data,open('./data/trained-models/'+self.algo_descr+'.model.json','w'))
        else:
            self.send_query(post_data)
        logger.info('End writing model', extra=self.prefix)

        # with open('model.json','w') as data_file:
        #    json.dump(post_data,data_file,indent=1)

    def send_query(self, post_data):
        logger.info('Send post data query', extra=self.prefix)
        prediction_string = '%s/algoconfig/%s' % (
            self.prediction_address, self.config_name)
        print prediction_string
        response = requests.post(
            prediction_string,
            json=post_data,
            headers=self.build_headers())
        response.raise_for_status()
        # response=requests.post(prediction_string, data=json.dumps(post_data),headers={"content-type": "application/json"}).json()
        return response.json()

    def read_model(self,offline=False):
        if offline:
            ret = json.load(open('./data/trained-models/'+self.config_name+'.model.json','r'))
        else:
            model_string = '%s/algoconfig/data?algorithm_config_name=%s&algorithm_config_version=latest' % (
                self.prediction_address, self.config_name)
            logger.info(
                'Reading the model from the following address : %s' %
                model_string, extra=self.prefix)
            response = requests.get(model_string).json()
            ret = response['data'][0]
        dict_ = ret['config_data']['configuration']['data']
        s2 = base64.b64decode(dict_)
        self.data_training = ret['config_data']['configuration']['info']
        self.data_training_fc_id = ret['config_data']['source_data_fc_id']
        return pickle.loads(s2)


    def write_prediction(self, algo, offline=False):

        prediction_dict = algo.prediction

        logger.info('Prediction skill scores', extra=self.prefix)
        # display total scores
        #try: # prevent from algorithms with 1 regression output
        for metrics in [algo.estimator.metrics_training, algo.estimator.metrics_testing]:
            # TODO: [DV] can we process "feature_importance" for algo.estimator.metrics_testing?
            # TODO: [DV] self._Ymean_ and self._Ystd_ are not set!
            if algo.name.startswith('R_nn') is False:
                if 'feature importance' in metrics:
                    # features_values = metrics['feature importance']
                    if hasattr(algo.estimator.estimator, 'coef_'):
                        features_values = algo.estimator.estimator.coef_.transpose().tolist()
                    elif hasattr(algo.estimator.estimator, 'feature_importances_'):
                        features_values = algo.estimator.estimator.feature_importances_.transpose().tolist()
                        metrics['feature importance'] = dict(zip(self.active_features, features_values))
            if 'r2' in metrics:
                rse_values = metrics['r2']
                metrics['r2'] = dict(
                    zip(self.labels, rse_values))
            if 'mse' in metrics:
                rse_values = metrics['mse']
                metrics['mse'] = dict(
                    zip(self.labels, rse_values))

            if  self.algo_descr.startswith('R_') is False:
                if len(self.label_df.columns) == 1:
                    #print(metrics)
                    print(metrics['0'])
                    metrics[self.label_df.columns[0]] = metrics['0']
                    if hasattr(algo,'_Ymean_') and  hasattr(algo,'_Ystd_') and hasattr(algo.estimator,'threshold'):
                        aux_destandardization = training_set_destandardization(algo.estimator.threshold, algo._Ymean_, algo._Ystd_)
                        if type(aux_destandardization) ==dict:
                            metrics[self.label_df.columns[0]]['threshold'] = aux_destandardization['0'].tolist()
                        else:
                            metrics[self.label_df.columns[0]]['threshold'] = aux_destandardization[0]
                    if hasattr(algo,'_Ymin_') and  hasattr(algo,'_Ymax_') and hasattr(algo.estimator,'threshold'):
                        metrics[self.label_df.columns[0]]['threshold'] = \
                            training_set_descaling(algo.estimator.threshold, algo._Ymin_, algo._Ymax_)[0]
                else:
                    for i in range(len(self.labels)):
                        metrics[self.label_df.columns[i]] = metrics.pop(str(i))
                    if hasattr(algo,'_Ymean_') and  hasattr(algo,'_Ystd_'):
                        metrics[self.label_df.columns[i]]['threshold'] = \
                            training_set_destandardization(algo.estimator.threshold[str(i)], algo._Ymean_[i], algo._Ystd_[i])
                    if hasattr(algo,'_Ymin_') and  hasattr(algo,'_Ymax_'):
                        metrics[self.label_df.columns[i]]['threshold'] = \
                            training_set_descaling(algo.estimator.threshold[str(i)], algo._Ymin_[i], algo._Ymax_[i])

        if offline:
            logger.info('Start writing results locally', extra=self.prefix)
            post_data = {'training_set':algo.estimator.metrics_training,
                         'testing_set':algo.estimator.metrics_testing}
            json.dump(post_data,open('./results/'+self.config_name+'_'+str(self.flare_class)+'_'+str(self.flare_window)+'h'+str(self.labels)+'.json','w'))

        # TODO: [DV] do we need the following lines of code?
        #aux_source_data = []
        #aux_source_data.append(
        #    ', '.join(
        #        d['fc_id'] for d in prediction_dict['source_data']))

        out = prediction_dict['prediction_data']['data']

        if hasattr(algo,'prediction_proba'):
            prediction_dict_proba = algo.prediction_proba
            out = out + prediction_dict_proba['prediction_data']['data']

        predictionset = {
            "algorithm_config": self.config_name,
            "algorithm_run_id": self.run_id,
            "prediction_data": out,
            "meta":{}#algo.estimator.metrics_testing["0"]
        }

        response = self.prepare_write_response(predictionset,offline)


        if offline:
            return
        else:
            return response.json()


    def prepare_write_response(self,predictionset,offline,proba = False):

        logger.info('Start writing prediction', extra=self.prefix)
        if offline:
            str_id = ''
            if proba == True:
                str_id = '_probability'
            json.dump(predictionset, open(
                './data/predictions/' + self.config_name + str_id + '.json', 'w'))
        else:
            prediction_string = '%s/predictionset' % self.prediction_address
            response = requests.post(
                prediction_string,
                json=predictionset,
                headers=self.build_headers())
            response.raise_for_status()
            view_prediction = "%s/prediction/list?algorithm_run_id=%s" % (
                self.prediction_address, self.run_id)
            logger.info(
                'Click here to view the prediction : %s' %
                view_prediction, extra=self.prefix)

        logger.info('End writing prediction', extra=self.prefix)

        if offline:
            return

        return response


class algorithm:
    def __init__(self, fc_dataset):
        # take the FLARECAST_LOG prefix from the db_access class
        self.prefix = fc_dataset.prefix

        # VESTIGIAL
        # select which algorithm is to be used in the params.json file list
        # for entry in fc_dataset.algorithm_info:
        #     if fc_dataset.algorithm_info[entry]:
        #         self.name = entry
        #self.name = 'HybridLogit_abovec_24_0_00_Blos_all_nofh'
        self.name = fc_dataset.algo_name

        # store the parameters
        self.parameters = fc_dataset.algorithm_info['parameters']

        # copy the standardization info in the algo object
        if hasattr(fc_dataset,'_mean_'):
            self._mean_ = fc_dataset._mean_
        if hasattr(fc_dataset,'_std_'):
            self._std_ = fc_dataset._std_
        if hasattr(fc_dataset,'_Ymean_'):
            self._Ymean_ = fc_dataset._Ymean_
        if hasattr(fc_dataset,'_Ystd_'):
            self._Ystd_ = fc_dataset._Ystd_

        # copy the normalization info in the algo object
        if hasattr(fc_dataset,'_min_'):
            self._min_ = fc_dataset._min_
        if hasattr(fc_dataset,'_max_'):
            self._max_ = fc_dataset._max_
        if hasattr(fc_dataset,'_Ymin_'):
            self._Ymin_ = fc_dataset._Ymin_
        if hasattr(fc_dataset,'_Ymax_'):
            self._Ymax_ = fc_dataset._Ymax_


        # List of the possible Machine Learning algorithm:
        ml_algorithm_classification = {'KMeans': KMeans_HM,
                        'FKMeans': FKMeans_HM,
                        'PKMeans': PKMeans_HM,
                        'SimAnn-KMeans': SimAnnKMeans_HM,
                        'SimAnn-FKMeans': SimAnnFKMeans_HM,
                        'SVC-CV': SVC_CV,
                        'RandomForest': RandomForest,
                        'MLPClassifier': MLPClassifier_HM,
                        'HybridLasso': HybridLasso,  # LassoCV,
                        'HybridLassoL2': HybridLassoL2,  # LassoCV,
                        'HybridLogit': HybridLogit}#,  # LogisticRegressionCV,
                        #'R_nn': R_nn,
                        #'R_svc': R_svc,
                        #'R_rf': R_rf,
                        #'R_lda': R_lda,
                        #'R_probit': R_probit,
                        #'R_logit': R_logit
                        #}

        ml_algorithm_regression = {'AdaptiveLasso-CV': AdaptiveLasso_CV,
                        'MLPRegressor': MLPRegressor_HM,
                        'MultiTaskLasso': MultiTaskLasso_CV,
                        'AdaptiveMultiTaskLasso': AdaptiveMultiTaskLasso_CV,
                        'MultiTaskPoissonLasso': MultiTaskPoissonLasso_CV,
                        'AdaptiveMultiTaskPoissonLasso': AdaptiveMultiTaskPoissonLasso_CV,
                        'SVR-CV': SVR_CV
                        }

        ml_algorithm = None

        if any([algo_name in self.name for algo_name in ml_algorithm_classification.keys()]):
            fc_dataset.algo_type = 'classification'
            ml_algorithm = next(
                (func for bname, func in ml_algorithm_classification.items() if self.name.startswith(bname)),
                None
            )
        if any([algo_name in self.name for algo_name in ml_algorithm_regression.keys()]):
            fc_dataset.algo_type = 'regression'
            ml_algorithm = next(
                (func for bname, func in ml_algorithm_regression.items() if self.name.startswith(bname)),
                None
            )

        fc_dataset.json_data['algorithm']['type'] = fc_dataset.algo_type #update the json_data structure

        if ml_algorithm is None:
            raise LookupError("Error: The algorithm '%s' is currently not supported!" % self.name)

        # Automatic execution of a Machine Learning algorithm
        # following the prescription on the configuration file
        self.estimator = ml_algorithm(**self.parameters)
        #self.name = 'I_am_algorithm_name'

    def save_panel(self, fc_dataset, year_str):
        file_X = 'X_' + year_str + '.pkl'
        file_Y = 'Y_' + year_str + '.pkl'
        file_X_panel = 'X_' + year_str + '_panel.pkl'
        file_Y_panel = 'Y_' + year_str + '_panel.pkl'
        file_X_txt = 'X_' + year_str + '.txt'
        file_Y_txt = 'Y_' + year_str + '.txt'
        file_active_features = 'features_' + year_str + '.npy'

        dataset = fc_dataset.load()
        fc_dataset.feature_dirty_df.to_pickle(file_X_panel)
        fc_dataset.label_dirty_df.to_pickle(file_Y_panel)

        # store which features and labels

        self.features = fc_dataset.feature_df.columns
        self.labels = fc_dataset.label_df.columns
        numpy.save(file_active_features, self.features.tolist())
        fc_dataset.feature_df.to_pickle(file_X)
        fc_dataset.label_df.to_pickle(file_Y)

        X = numpy.array(fc_dataset.feature_df, dtype=float)
        Y = numpy.array(fc_dataset.label_df, dtype=float)
        numpy.savetxt(file_X_txt, X)
        numpy.savetxt(file_Y_txt, Y)

    def train_db(self, fc_dataset):
        # read the dataset from the method point_in_time in access_db and
        # fit the properties X with the labels Y.
        logger.info(
            'Start training algorithm : %s' %
            self.name, extra=self.prefix)
        logger.debug(
            'Algorithm parameters : %s' %
            self.parameters, extra=self.prefix)

        if fc_dataset.Ys.std() == 0:
            warnings.warn(
                "In the selected time interval labels are all 0 ",
                UserWarning, stacklevel=2)
            exit(1)
        else:
            if self.name[0:2]=='R_':
                self.estimator.fit(fc_dataset.Xs, fc_dataset.Ys)
            else:
                self.estimator.fit(fc_dataset.Xs, (fc_dataset.Ys, fc_dataset.Y))

            logger.info('End training', extra=self.prefix)


    def predict_db(self, fc_dataset):

        # read the dataset from the method point_in_time in access_db and return the
        # prediction
        logger.info(
            'Start prediction algorithm : %s' %
            self.name, extra=self.prefix)
        logger.debug(
            'Algorithm parameters : %s' %
            self.parameters, extra=self.prefix)

        # prediction step
        if fc_dataset.Ys.std() == 0:
            warnings.warn(
                "In the selected time interval labels are all 0 ",
                UserWarning, stacklevel=2)
            #exit(2)

        try:
            tmp = fc_dataset.algorithm_info['parameters']['preprocessing']
            if tmp['standardization_feature']:
                if fc_dataset.Xs.shape[1] == self._mean_.shape[0]:
                    fc_dataset.Xs = testing_set_standardization(fc_dataset.X, self._mean_, self._std_)
                    if tmp['scaling_feature']:
                        warnings.warn(
                            "Both standardization and scaling on the features are active , standardization has been used",
                            UserWarning, stacklevel=2)
                else:
                    warnings.warn("different set of properties w.r.t. training set",
                                  UserWarning, stacklevel=2)
                    exit(100)
            elif tmp['scaling_feature']:
                if fc_dataset.Xs.shape[1] == self._max_.shape[0]:
                    fc_dataset.Xs = testing_set_scaling(fc_dataset.X, self._max_, self._min_)
                else:
                    warnings.warn("different set of properties w.r.t. training set",
                                  UserWarning, stacklevel=2)
                    exit(100)

            if tmp['standardization_label']:
                fc_dataset.Ys = testing_set_standardization(fc_dataset.Y, self._Ymean_, self._Ystd_)
                if tmp['scaling_label']:
                    warnings.warn(
                        "Both standardization and scaling on the labels are active, standardization has been used",
                        UserWarning, stacklevel=2)
            elif tmp['scaling_label']:
                fc_dataset.Ys = testing_set_scaling(fc_dataset.Y, self._Ymax_, self._Ymin_)


        except KeyError:
            pass

        if self.name[0:2] == 'R_':
            pred = self.estimator.predict(fc_dataset.Xs, fc_dataset.Ys)
        else:
            pred = self.estimator.predict(fc_dataset.Xs, (fc_dataset.Ys, fc_dataset.Y))



        prediction_df = fc_dataset.label_df.copy()

        if fc_dataset.algo_type == 'regression':
            inner_data = []
            for i in prediction_df.axes[1]:
                inner_data.append('data:result:' + i)
        else:
            inner_data = ['data:result'] # valid for classification and probability
        prediction_df.columns = inner_data

        # fill the dataframe columns
        # the operation *1 is a trick to convert bool to integer values !
        if len(inner_data) == 1:
            prediction_df[prediction_df.columns[0]] = pred * 1
        else:
            prediction_df[prediction_df.columns] = pred * 1

        prediction_df['source_data'] = fc_dataset.fc_id_df


        if hasattr(fc_dataset,'algo_type'):
            prediction_df['output_type'] = fc_dataset.algo_type
        else:
            prediction_df['output_type'] = '- no type declared -'


        prediction_df['window_latency'] = 0  # in hours
        prediction_df['window_duration'] = fc_dataset.flare_window  # in hours
        prediction_df['intensity_min'] = convert(fc_dataset.flare_class)
        prediction_df['intensity_max'] = convert(fc_dataset.flare_class_max)
        prediction_df['meta'] = [{}] * len(pred)
        prediction_df['flare_association'] = [{}] * len(pred)
        
        for it_aux_prediction_df in numpy.arange(prediction_df.shape[0]):
            aux_fc_id = prediction_df['source_data'].iloc[it_aux_prediction_df] 
            for it_flare_association in numpy.arange(len(fc_dataset.flare_association)):
                if aux_fc_id == fc_dataset.flare_association[it_flare_association]['fc_id']:
                    if type(fc_dataset.flare_association[it_flare_association]['data']['flare_association']) == dict:
                        prediction_df['flare_association'].iloc[it_aux_prediction_df] = [fc_dataset.flare_association[it_flare_association]['data']['flare_association']]
                    else:
                        prediction_df['flare_association'].iloc[it_aux_prediction_df] = fc_dataset.flare_association[it_flare_association]['data']['flare_association']#[flatdict.FlatDict(fc_dataset.flare_association[it_flare_association]['data'])]
        

        if hasattr(self.estimator, 'probability'):
            prediction_df_proba = prediction_df.copy()
            prediction_df_proba['data:result'] = self.estimator.probability
            prediction_df_proba['output_type'] = 'probability'

            self.prediction_proba = store_prediction(prediction_df_proba,fc_dataset)



        discarded = pandas.DataFrame(fc_dataset.discarded_fc_id_df['fc_id'].copy())

        if discarded.shape[0]>0:
            discarded['source_data'] = fc_dataset.discarded_fc_id_df['fc_id']
            discarded['window_latency'] = 0  # in hours
            discarded['window_duration'] = fc_dataset.flare_window  # in hours
            discarded['intensity_min'] = convert(fc_dataset.flare_class)
            discarded['intensity_max'] = convert(fc_dataset.flare_class_max)
            discarded['meta'] = [{}] * discarded.shape[0]
            del discarded['fc_id']


            if hasattr(fc_dataset,'algo_type'):
                discarded['output_type'] = fc_dataset.algo_type
            else:
                discarded['output_type'] = '- no type declared -'

            if hasattr(self.estimator, 'probability'):
                discarded_proba = discarded.copy()
                discarded_proba['output_type'] = 'probability'

                full_discarded = pandas.concat([discarded, discarded_proba], axis=0)

            else:
                full_discarded = discarded.copy()



            prediction_df = pandas.concat([prediction_df, full_discarded], axis=0)

        prediction_df = prediction_df.fillna(-1)
        self.prediction = store_prediction(prediction_df, fc_dataset)

        logger.debug(
            'Prediction phase : prediction %s' %
            pred, extra=self.prefix)

def store_discarded(prediction_df, dataset):
    prediction_df['source_data'] = dataset
    prediction_df['data:result'] = 'Null'

    discarded = {}

    return discarded

def store_prediction(prediction_df, dataset):
    out = []
    for i in prediction_df.to_dict(orient='record'):
        out.append(flatdict.FlatDict(i))
    for entry in out:
        entry['source_data'] = [entry['source_data']]
        

    prediction = {'source_data': dataset.data_training_fc_id,
                             'prediction_data':
                                 {'data': out,
                                  'testing_set': dataset.data_testing
                                  }

                             }
    return prediction

def convert(s):
    # Convert the flare_class in a number by using the standard conversion
    # formula
    flare_class = {'A': 0.01, 'B': 0.1, 'C': 1, 'M': 10, 'X': 100}
    try:
        try:
            s = flare_class[s[0]] * float(s[1::])
        except IndexError:
            s = 0
    except:
        pass
    return s


def visualization(db, algo, type, CI=True):
    # ROC ROC ROC
    y_prob = numpy.asarray(algo.prediction['prediction_data']['data'])
    y_obs = numpy.asarray(db.point_in_time()['y'])

    y_prob = y_prob.flatten()
    y_obs = y_obs.flatten()

    if type == 'ROC':
        r_plot_ROC(y_obs, y_prob)

    if type == 'RD':
        r_plot_RD(y_obs, y_prob, CI)

    if type == 'SSP':
        r_plot_SSP(y_obs, y_prob)


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


if __name__ == "__main__":
    # # read json
    # param = read_json()
    # db = d.property_db(param)

    # read properties
    db = property_db()
    # contains
    # db.read_json('params.json')
    # db.read_properties()

    # build two panels of point in time data
    p = db.point_in_time()

    # set algorithm from param.json
    alg = algorithm(db)

    # load the estimator
    method = alg.estimator

    # learning phase
    result = method.fit(p['X'], p['y'])

    # store the learning/predicting parameters
    params = method.get_params()

    # write params into the prediction config database
