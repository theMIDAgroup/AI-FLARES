# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 05:13:17 2017

@author: benvenuto, florios
"""

import numpy
from Py_flarecast_learning_algorithms import metrics_classification, optimize_threshold

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects import pandas2ri
rpy2.robjects.numpy2ri.activate()
#pandas2ri.activate()







def dict2string(parameters):
    """
    convert the parameter dictionary to a parameter string
    """

    parameter_string = """ """
    for k in parameters.keys():
        # concatenate keys to str(items) of the dictonary
        parameter_string += k + '=' + str(parameters[k]) + ', '
    # erase the last comma
    parameter_string = parameter_string[:-2]

    return parameter_string


class R_nn:
    """
    Python wrapper for the R neural network code
    """

    def __init__(self, **parameters):
        """
        define an "R estimator" by writing the fit and the predict functions
        according with the scikit-learn logic scheme, so that:
        1) an instance of the R_nn class is initialized with all the
        parameters needed to training the estimator.
        2) the .fit method acts on a pair X and y
        3) the .predict method acts on an X_new object
        The fit and predict functions are realized by means of the rpy2.robjects module which
        transforms R code (read as a string) into python object.
        In this way we can pass parameters to the original R functions
        """

        # convert the parameters dictionary to a parameter string
        # to concatenate the string to R_function_as_a_string
        # extract garson/olden flag
#       #garson_olden = parameters.pop('garson_olden')
        #global garson_olden
        # try:
        #    R_nn_parameters = parameters.pop('R_nn')
        #type = R_nn_parameters.pop('garson_olden')
        #type = R_nn_parameters.pop('feature_importance')
        #    garson_olden = parameters.pop('feature_importance')
        # except KeyError:
        #    R_nn_parameters = parameters
        #parameter_string = dict2string(R_nn_parameters)

        R_nn_parameters = parameters.pop('R_nn')
        #type = R_nn_parameters.pop('garson_olden')
        #type = R_nn_parameters.pop('feature_importance')
        self.garson_olden = parameters.pop('feature_importance')
        parameter_string = dict2string(R_nn_parameters)

        # define the R functions fit and predict
        # python reads these functions just as a string
        self._fit = robjects.r("""
        R_fit <- function(X,y) {
            require(nnet)
            p <- nnet(X,y,""" + parameter_string + """)
            return(p)
            #p.nnet < - nnet(X,y,""" + parameter_string + """)
            #p.nnet <- nnet(X,y)
            #p.nnet <- nnet(X,y,size=4,decay=0.1,entropy=TRUE,maxit=2000,MaxNWts=4000)
            #return(p.nnet)
            }
        """)

        self._predict = robjects.r("""
        R_predict <- function(mObject,X) {
            require(nnet)
            newData <- as.data.frame(X)
            p.predict <- predict(mObject,X)
            return(as.numeric(p.predict))
            #p.nnet.predict <- predict(mObject,X)
            #return(as.numeric(p.nnet.predict))
            }
        """)

        self._feature_importances = robjects.r("""
        ################################################################################################################
        ################### ADD HERE THE GARSON / OLDEN FEATURE SELECTION ####################
        ################################################################################################################
        R_feature_importances <- function(mObject,X,garson_olden) {
            type <- garson_olden
            cat("...:",type,"\n")
            if (type == "garson1") {

                #Garson importance
                require(NeuralNetTools)
                p.nnet <- mObject
                wts_in <- p.nnet$wts
                struct <- p.nnet$n
                #importancesG <- garson(wts_in,struct )
                #importancesGvec <- importancesG$data$rel_imp
                #importancesGnam <- as.character(importancesG$data$x_names)
                importances <- garson(wts_in,struct)
                priorities <- data.frame("xid"=character(struct[1]),"importanse"=numeric(struct[1]), "nid"=numeric(struct[1]))

                priorities$xid <- importances$data$x_names
                priorities$importanse <- importances$data$rel_imp
                priorities$nid <- as.numeric(row.names(importances$data))

                order.dat <- order(priorities$nid)

                prioritiesToReturn <- data.frame("importanse"=numeric(struct[1]), "nid"=numeric(struct[1]))
                #prioritiesToReturn$xid <- paste("x",1:15,sep="")
                prioritiesToReturn$importanse <- priorities$importanse[order.dat]
                prioritiesToReturn$nid <- 1:struct[1]
                #return(prioritiesToReturn)  #serialized
           }


        	if (type == "olden1") {
                 #Olden importance
                 require(NeuralNetTools)
                 p.nnet <- mObject
                 wts_in <- p.nnet$wts
                 struct <- p.nnet$n
                 #importancesO <- olden(wts_in,struct )
                 #importancesOvec <- importancesO$data$importance
                 #importancesOnam <-  as.character(importancesO$data$x_names)
                 importances <- olden(wts_in,struct)
                 priorities <- data.frame("xid"=character(struct[1]),"importanse"=numeric(struct[1]), "nid"=numeric(struct[1]))

            	 priorities$xid <- importances$data$x_names
                 priorities$importanse <- importances$data$importance
                 priorities$nid <- as.numeric(row.names(importances$data))

            	order.dat <- order(priorities$nid)

            	prioritiesToReturn <- data.frame("importanse"=numeric(struct[1]), "nid"=numeric(struct[1]))
                 #prioritiesToReturn$xid <- paste("x",1:15,sep="")
                 prioritiesToReturn$importanse <- priorities$importanse[order.dat]
                 prioritiesToReturn$nid <- 1:struct[1]
                 #return(prioritiesToReturn)  #serialized
        	}

        	return(prioritiesToReturn$importanse) #serialized
           #return(prioritiesToReturn) #serialized
        }
        """)
        
        self.metrics_training = {}
        self.metrics_testing = {}
        self.threshold = 0.25

    '''
    def fit(self, X_training, Y_training):

        # call the R function fit from python
        # and save the mObject in to a saparate variable
        self.mObject = self._fit(X_training, Y_training)
        #self.metrics_training = metrics_classification(Y_training,self.predict(X_training))
        self.metrics_training["feature importance"] = numpy.array(self._feature_importances()).tolist()


    def predict(self, X_testing, Y_testing = None):

        # call the R function predict from python
        Y_testing_prediction = numpy.array(self._predict(self.mObject, X_testing))
        if Y_testing is not None:
            self.metrics_testing = metrics_classification(Y_testing, Y_testing_prediction)
        return Y_testing_prediction
    '''

    def fit(self, X_training, Y_training):
        # call the R function fit from python
        # and save the mObject in to a saparate variable
        self.mObject = self._fit(X_training, Y_training)
        #probability_prediction = self.predict(X_training)
        self.fitted_values = numpy.array(
            self._predict(self.mObject, X_training))
        self.threshold, self.metrics_training = optimize_threshold(
            self.fitted_values, Y_training, 'hybrid')  # self.classification)
        # self.training_set_metrics = classification_metrics(Y_training,probability_prediction)
        #feature_importances = self._feature_importances(self.mObject, X_training, type)
        feature_importances = self._feature_importances(
            self.mObject, X_training, self.garson_olden)
        #self.metrics_training["feature importance"] = numpy.array(
        #    feature_importances).tolist()
        self.metrics_training["feature importance"] = numpy.array(
            feature_importances).tolist()

    def predict(self, X_testing, Y_testing=None):
        # call the R function predict from python
        self.probability = numpy.array(self._predict(self.mObject, X_testing))
        Y_testing_classification = self.probability > self.threshold
        if Y_testing is not None:
            self.metrics_testing = metrics_classification(
                Y_testing, Y_testing_classification)
        # return Y_testing_prediction
        # return self.probability
        return Y_testing


class R_svm:
    """
    Python wrapper for the R svm code
    """

    def __init__(self, **parameters):
        """
        define an "R estimator" by writing the fit and the predict functions
        according with the scikit-learn logic scheme, so that:
        1) an instance of the R_svm class is initialized with all the
        parameters needed to training the estimator.
        2) the .fit method acts on a pair X and y
        3) the .predict method acts on an X_new object
        The fit and predict functions are realized by means of the rpy2.robjects module which
        transforms R code (read as a string) into python object.
        In this way we can pass parameters to the original R functions
        """

        # convert the parameters dictionary to a parameter string
        # to concatenate the string to R_function_as_a_string
        parameters_R_svm = parameters.pop('R_svm')
        parameter_string = dict2string(parameters_R_svm)
        # TODO: gamma=0.5, cost=8, probability=TRUE

        # define the R functions fit and predict
        # python reads these functions just as a string
        self._fit = robjects.r("""
                                fc_svm <- function(X,y) {
                                  require(e1071)
                                  p.svm < - svm(X,y,""" + parameter_string + """)
                                  #p.svm <- svm(X,y)
                                  #p.svm <- svm(X,y,probability=TRUE)
                                  return(p.svm)
                                }
                                """)

        self._predict = robjects.r("""
                                    fc_svm_predict <- function(mObject,Xnew) {
                                      require(e1071)
                                      newData <- as.data.frame(Xnew)
                                      p.svm.predict <- predict(mObject,newData)
                                      return(as.numeric(p.svm.predict))
                                    }
                                    """)
        self.metrics_training = {}
        self.metrics_testing = {}
        self.threshold = 0.25

    def fit(self, X, y):

        # call the R function fit from python
        # and save the mObject in to a saparate variable
        self.mObject = self._fit(X, y)

    # def predict(self, X, Y = None):

        # call the R function predict from python
        #R_prediction = self._predict(self.mObject,X)
        # if Y is not None:
        #    self.metrics_testing = metrics_classification(Y, R_prediction)

        # return numpy.array(R_prediction)

    def predict(self, X_testing, Y_testing=None):
        self.probability = numpy.array(self._predict(self.mObject, X_testing))
        Y_testing_classification = self.probability > self.threshold
        if Y_testing is not None:
            self.metrics_testing = metrics_classification(
                Y_testing, Y_testing_classification)
        # return Y_testing_prediction
        # return self.probability
        return Y_testing


class R_svc:
    """
    Python wrapper for the R svm code
    """

    def __init__(self, **parameters):
        """
        define an "R estimator" by writing the fit and the predict functions
        according with the scikit-learn logic scheme, so that:
        1) an instance of the R_svc class is initialized with all the
        parameters needed to training the estimator.
        2) the .fit method acts on a pair X and y
        3) the .predict method acts on an X_new object
        The fit and predict functions are realized by means of the rpy2.robjects module which
        transforms R code (read as a string) into python object.
        In this way we can pass parameters to the original R functions
        """

        # convert the parameters dictionary to a parameter string
        # to concatenate the string to R_function_as_a_string
        parameters_R_svc = parameters.pop('R_svc')
        parameter_string = dict2string(parameters_R_svc)
        # TODO: gamma=0.5, cost=8, probability=TRUE

        # define the R functions fit and predict
        # python reads these functions just as a string
        self._fit = robjects.r("""
                                fc_svm <- function(X,y) {
                                  require(e1071)
                                  # sort in increasing order of y, libsvm issue
                                  te <- cbind(X,y)
                                  te <- te[order(te[,dim(te)[2]]),]
                                  X  <- te[,1:(dim(te)[2]-1)]
                                  y  <- te[,dim(te)[2]]
                                  #
                                  y <- as.factor(y)
                                  p.svm <- svm(X,as.factor(y),""" + parameter_string + """,type= "C-classification",kernel= "radial")
                                  #p.svm <- svm(X,y)
                                  #p.svm <- svm(X,y,probability=TRUE)
                                  y <- as.numeric(y) -1
                                  return(p.svm)
                                }
                                """)

        self._predict = robjects.r("""
                                    fc_svm_predict <- function(mObject,Xnew) {
                                      require(e1071)
                                      newData <- as.data.frame(Xnew)
                                      #p.svm.predict <- predict(mObject,newData)
                                      p.svm.pred <- predict(mObject,newData,probability=TRUE)
                                      p.svm.pred.num <- attr(p.svm.pred,"prob")[,2]
                                      #return(as.numeric(p.svm.predict))
                                      return(as.numeric(p.svm.pred.num))
                                    }
                                    """)
        self.metrics_training = {}
        self.metrics_testing = {}
        self.threshold = 0.25

    def fit(self, X, y):

        # call the R function fit from python
        # and save the mObject in to a saparate variable
        self.mObject = self._fit(X, y)

    # def predict(self, X, Y = None):

        # call the R function predict from python
        #R_prediction = self._predict(self.mObject,X)
        # if Y is not None:
        #    self.metrics_testing = metrics_classification(Y, R_prediction)

        # return numpy.array(R_prediction)

    def predict(self, X_testing, Y_testing=None):
        self.probability = numpy.array(self._predict(self.mObject, X_testing))
        Y_testing_classification = self.probability > self.threshold
        if Y_testing is not None:
            self.metrics_testing = metrics_classification(
                Y_testing, Y_testing_classification)
        # return Y_testing_prediction
        # return self.probability
        return Y_testing


class R_rf:
    """
    Python wrapper for the R rf code
    """

    def __init__(self, **parameters):
        """
        define an "R estimator" by writing the fit and the predict functions
        according with the scikit-learn logic scheme, so that:
        1) an instance of the R_rf class is initialized with all the
        parameters needed to training the estimator.
        2) the .fit method acts on a pair X and y
        3) the .predict method acts on an X_new object
        The fit and predict functions are realized by means of the rpy2.robjects module which
        transforms R code (read as a string) into python object.
        In this way we can pass parameters to the original R functions
        """

        # convert the parameters dictionary to a parameter string
        # to concatenate the string to R_function_as_a_string
        parameters_R_rf = parameters.pop('R_rf')
        parameter_string = dict2string(parameters_R_rf)
        #print(parameter_string)
        # TODO: gamma=0.5, cost=8, probability=TRUE

        # define the R functions fit and predict
        # python reads these functions just as a string
        self._fit = robjects.r("""
                                  fc_randomForest <- function(X,y) {
                                  require(randomForest)
                                  data <- as.data.frame(cbind(X,y))
                                  p.randomForest <- randomForest(X,y,""" + parameter_string + """)
                                  #p.randomForest <- randomForest(X,y)
                                  return(p.randomForest)
                                  }
                                """)

        self._predict = robjects.r("""
                                    fc_randomForest_predict <- function(mObject,Xnew) {
                                    require(randomForest)
                                    #newData <- as.data.frame(Xnew)
                                    #p.randomForest.predict <- predict(mObject,newData,type="response")
                                    p_randomForest_predict <- predict(mObject,Xnew,type="response")
                                    return(as.numeric(p_randomForest_predict))
                                    }
                                    """)
        self.metrics_training = {}
        self.metrics_testing = {}
        self.threshold = 0.25

    def fit(self, X, y):

        # call the R function fit from python
        # and save the mObject in to a saparate variable

        self.mObject = self._fit(X, y)

    # def predict(self, X, Y = None):

        # call the R function predict from python
        #R_prediction = self._predict(self.mObject,X)
        # if Y is not None:
        #    self.metrics_testing = metrics_classification(Y, R_prediction)
        # return numpy.array(R_prediction)

    def predict(self, X_testing, Y_testing=None):
        self.probability = numpy.array(self._predict(self.mObject, X_testing))
        Y_testing_classification = self.probability > self.threshold
        if Y_testing is not None:
            self.metrics_testing = metrics_classification(
                Y_testing, Y_testing_classification)
        # return Y_testing_prediction
        # return self.probability
        return Y_testing


class R_lm:
    """
    Python wrapper for the R rf code
    """

    def __init__(self, **parameters):
        """
        define an "R estimator" by writing the fit and the predict functions
        according with the scikit-learn logic scheme, so that:
        1) an instance of the R_nn class is initialized with all the
        parameters needed to training the estimator.

        2) the .fit method acts on a pair X and y
        3) the .predict method acts on an X_new object
        The fit and predict functions are realized by means of the rpy2.robjects module which
        transforms R code (read as a string) into python object.
        In this way we can pass parameters to the original R functions
        """

        # convert the parameters dictionary to a parameter string
        # to concatenate the string to R_function_as_a_string
        parameters_R_lm = parameters.pop('R_lm')
        parameter_string = dict2string(parameters_R_lm)
        # TODO: gamma=0.5, cost=8, probability=TRUE

        # define the R functions fit and predict
        # python reads these functions just as a string
        self._fit = robjects.r("""
                                fc_lm <- function(X,y) {
                                data <- as.data.frame(cbind(X,y))
                                names(data) <- paste("V",1:dim(data)[2],sep="")
                                names(data)[dim(data)[2]] <- "y"
                                m1 <- lm( y ~ .,data = data)
                                return(m1)
                                }
                                """)

        self._predict = robjects.r("""
                                    fc_lm_predict <- function(mObject,Xnew) {
                                    newData <- as.data.frame(Xnew)
                                    #names(newData) <- paste("V",1:dim(newData)[2],sep="")
                                    #names(newData)[dim(newData)[2]] <- "y"
                                    #newData <- newData[,-dim(newData)[2]]
                                    m1_predict <- predict(mObject,newdata=newData)
                                    return(as.numeric(m1_predict))
                                    }
                                """)
        self.metrics_training = {}
        self.metrics_testing = {}
        self.threshold = 0.25

    def fit(self, X, y):

        # call the R function fit from python
        # and save the mObject in to a saparate variable
        self.mObject = self._fit(X, y)

    # def predict(self, X, Y = None):

        # call the R function predict from python
        #R_prediction = self._predict(self.mObject,X)
        # if Y is not None:
        #    self.metrics_testing = metrics_classification(Y, R_prediction)
        # return numpy.array(R_prediction)

    def predict(self, X_testing, Y_testing=None):
        self.probability = numpy.array(self._predict(self.mObject, X_testing))
        Y_testing_classification = self.probability > self.threshold
        if Y_testing is not None:
            self.metrics_testing = metrics_classification(
                Y_testing, Y_testing_classification)
        # return Y_testing_prediction
        # return self.probability
        return Y_testing


class R_lda:
    """
    Python wrapper for the R rf code
    """

    def __init__(self, **parameters):
        """
        define an "R estimator" by writing the fit and the predict functions
        according with the scikit-learn logic scheme, so that:
        1) an instance of the R_nn class is initialized with all the
        parameters needed to training the estimator.

        2) the .fit method acts on a pair X and y
        3) the .predict method acts on an X_new object
        The fit and predict functions are realized by means of the rpy2.robjects module which
        transforms R code (read as a string) into python object.
        In this way we can pass parameters to the original R functions
        """

        # convert the parameters dictionary to a parameter string
        # to concatenate the string to R_function_as_a_string
        parameters_R_lda = parameters.pop('R_lda')
        parameter_string = dict2string(parameters_R_lda)
        # TODO: gamma=0.5, cost=8, probability=TRUE

        # define the R functions fit and predict
        # python reads these functions just as a string
        self._fit = robjects.r("""
                                fc_lda <- function(X,y) {
                                require(MASS)
                                y <- as.factor(y)
                                data <- as.data.frame(cbind(X,y))
                                names(data) <- paste("V",1:dim(data)[2],sep="")
                                names(data)[dim(data)[2]] <- "y"
                                m4 <- lda( y ~ .,data = data)
                                return(m4)
                                }
                                """)

        self._predict = robjects.r("""
                                    fc_lda_predict <- function(mObject,Xnew) {
                                    require(MASS)
                                    newData <- as.data.frame(Xnew)
                                    #names(newData) <- paste("V",1:dim(newData)[2],sep="")
                                    #names(newData)[dim(newData)[2]] <- "y"
                                    #newData <- newData[,-dim(newData)[2]]
                                    #m4_predict <- predict(mObject,newdata=newData)$class
                                    #m4_predict.num <- as.numeric(m4_predict) - 1
                                    m4_predict <- predict(mObject,newdata=newData)$posterior[,2]
                                    m4_predict.num <- as.numeric(m4_predict)
                                    return(as.numeric(m4_predict.num))
                                    }
                                """)
        self.metrics_training = {}
        self.metrics_testing = {}
        self.threshold = 0.25

    def fit(self, X, y):

        # call the R function fit from python
        # and save the mObject in to a saparate variable
        self.mObject = self._fit(X, y)

    # def predict(self, X, Y = None):

        # call the R function predict from python
        #R_prediction = self._predict(self.mObject,X)
        # if Y is not None:
        #    self.metrics_testing = metrics_classification(Y, R_prediction)
        # return numpy.array(R_prediction)

    def predict(self, X_testing, Y_testing=None):
        self.probability = numpy.array(self._predict(self.mObject, X_testing))
        Y_testing_classification = self.probability > self.threshold
        if Y_testing is not None:
            self.metrics_testing = metrics_classification(
                Y_testing, Y_testing_classification)
        # return Y_testing_prediction
        # return self.probability
        return Y_testing


class R_probit:
    """
    Python wrapper for the R rf code
    """

    def __init__(self, **parameters):
        """
        define an "R estimator" by writing the fit and the predict functions
        according with the scikit-learn logic scheme, so that:
        1) an instance of the R_nn class is initialized with all the
        parameters needed to training the estimator.
        2) the .fit method acts on a pair X and y
        3) the .predict method acts on an X_new object
        The fit and predict functions are realized by means of the rpy2.robjects module which
        transforms R code (read as a string) into python object.
        In this way we can pass parameters to the original R functions
        """

        # convert the parameters dictionary to a parameter string
        # to concatenate the string to R_function_as_a_string
        parameters_R_probit = parameters.pop('R_probit')
        parameter_string = dict2string(parameters_R_probit)
        # TODO: gamma=0.5, cost=8, probability=TRUE

        # define the R functions fit and predict
        # python reads these functions just as a string
        self._fit = robjects.r("""
                                fc_probit <- function(X,y) {
                                data <- as.data.frame(cbind(X,y))
                                names(data) <- paste("V",1:dim(data)[2],sep="")
                                names(data)[dim(data)[2]] <- "y"
                                m2 <- glm(y ~ ., data = data, family=binomial(probit))
                                return(m2)
                                }
                                """)

        self._predict = robjects.r("""
                                    fc_probit_predict <- function(mObject,Xnew) {
                                    newData <- as.data.frame(Xnew)
                                    #names(newData) <- paste("V",1:dim(newData)[2],sep="")
                                    #names(newData)[dim(newData)[2]] <- "y"
                                    #newData <- newData[,-dim(newData)[2]]
                                    m2_predict <- predict(mObject,newdata=newData,type="response")
                                    return(as.numeric(m2_predict))
                                    }
                                    """)
        self.metrics_training = {}
        self.metrics_testing = {}
        self.threshold = 0.25

    def fit(self, X, y):

        # call the R function fit from python
        # and save the mObject in to a saparate variable
        self.mObject = self._fit(X, y)

    # def predict(self, X, Y = None):

        # call the R function predict from python
        #R_prediction = self._predict(self.mObject,X)
        # if Y is not None:
        #    self.metrics_testing = metrics_classification(Y, R_prediction)
        # return numpy.array(R_prediction)

    def predict(self, X_testing, Y_testing=None):
        self.probability = numpy.array(self._predict(self.mObject, X_testing))
        Y_testing_classification = self.probability > self.threshold
        if Y_testing is not None:
            self.metrics_testing = metrics_classification(
                Y_testing, Y_testing_classification)
        # return Y_testing_prediction
        # return self.probability
        return Y_testing


class R_logit:
    """
    Python wrapper for the R rf code
    """

    def __init__(self, **parameters):
        """
        define an "R estimator" by writing the fit and the predict functions
        according with the scikit-learn logic scheme, so that:
        1) an instance of the R_nn class is initialized with all the
        parameters needed to training the estimator.
        2) the .fit method acts on a pair X and y
        3) the .predict method acts on an X_new object
        The fit and predict functions are realized by means of the rpy2.robjects module which
        transforms R code (read as a string) into python object.
        In this way we can pass parameters to the original R functions
        """

        # convert the parameters dictionary to a parameter string
        # to concatenate the string to R_function_as_a_string
        parameters_R_logit = parameters.pop('R_logit')
        parameter_string = dict2string(parameters_R_logit)
        # TODO: gamma=0.5, cost=8, probability=TRUE

        # define the R functions fit and predict
        # python reads these functions just as a string
        self._fit = robjects.r("""
                                fc_logit <- function(X,y) {
                                data <- as.data.frame(cbind(X,y))
                                names(data) <- paste("V",1:dim(data)[2],sep="")
                                names(data)[dim(data)[2]] <- "y"
                                m3 <- glm(y ~ ., data = data, family=binomial(logit))
                                return(m3)
                                }
                                """)

        self._predict = robjects.r("""
                                    fc_logit_predict <- function(mObject,Xnew) {
                                    newData <- as.data.frame(Xnew)
                                    #names(newData) <- paste("V",1:dim(newData)[2],sep="")
                                    #names(newData)[dim(newData)[2]] <- "y"
                                    #newData <- newData[,-dim(newData)[2]]
                                    m3_predict <- predict(mObject,newdata=newData,type="response")
                                    return(as.numeric(m3_predict))
                                    }
                                    """)
        self.metrics_training = {}
        self.metrics_testing = {}
        self.threshold = 0.25

    def fit(self, X, y):

        # call the R function fit from python
        # and save the mObject in to a saparate variable
        self.mObject = self._fit(X, y)

    # def predict(self, X, Y = None):

        # call the R function predict from python
        #R_prediction = self._predict(self.mObject,X)
        # if Y is not None:
        #    self.metrics_testing = metrics_classification(Y, R_prediction)
        # return numpy.array(R_prediction)

    def predict(self, X_testing, Y_testing=None):
        self.probability = numpy.array(self._predict(self.mObject, X_testing))
        Y_testing_classification = self.probability > self.threshold
        if Y_testing is not None:
            self.metrics_testing = metrics_classification(
                Y_testing, Y_testing_classification)
        # return Y_testing_prediction
        # return self.probability
        return Y_testing


##########################################################################
# VISUALIZATION ROUTINES


R_plot_ROC = robjects.r("""
        R_plot_ROC <- function(obs,prob) {
        #ROC curves
        #install.packages("ROCR",dependencies=TRUE)
        require(ROCR)

        #double censore prob to left censored at 0 and right censored at 1
        #new variable is probF
        probF <- numeric(length(prob))
        for (i in 1:length(prob)) {
             probF[i] <- max(0,min(prob[i],1))
        }

        #an underlying model e.g. a neural network
        pred <- ROCR::prediction(as.numeric(probF),as.numeric(obs))
        perf <- ROCR::performance(pred,"tpr","fpr")
        jpeg(paste("ROC_a_model_",".jpeg",sep=""))
        plot(perf,main="ROC curve")
        lines(seq(0,1,0.01),seq(0,1,0.01))
        dev.off()

        auc.perf = performance(pred, measure = "auc")
        auc.perf@y.values
        }
""")

r_plot_ROC = robjects.globalenv['R_plot_ROC']


R_plot_RD = robjects.r("""
        R_plot_RD <- function(obs,prob,CI) {
        #Reliability diagrams
        #install.packages("pracma",dependencies=TRUE)
        #install.packages("verification",dependencies=TRUE)
        require(verification)

        #double censore prob to left censored at 0 and right censored at 1
        #new variable is probF
        probF <- numeric(length(prob))
        for (i in 1:length(prob)) {
             probF[i] <- max(0,min(prob[i],1))
        }

        if (CI == TRUE) {
        #an underlying model e.g. a neural network
        mod <- verify(obs, prob, thresholds=seq(0,1,0.05))
        jpeg("ReliabilityDiagram_ErrorBars_a_model_.jpeg")
        plot(mod, CI=T, main="Reliability Diagram with Error Bars")
        lines(seq(0,1,0.01),seq(0,1,0.01))
        dev.off()

        }
        if (CI == FALSE) {
        #an underlying model e.g. a neural network
        mod <- verify(obs, prob, thresholds=seq(0,1,0.05))
        jpeg("ReliabilityDiagram_a_model_.jpeg")
        plot(mod, main="Reliability Diagram")
        lines(seq(0,1,0.01),seq(0,1,0.01))
        dev.off()

        }
        }
""")

r_plot_RD = robjects.globalenv['R_plot_RD']


R_plot_SSP = robjects.r("""
        R_plot_SSP <- function(obs,pred) {

        require(pracma)
        #because a model could give probabilities <0 and >1
        #first bound left to 0 and right to 1
        pred.new <- numeric(length(pred))
        for (i in 1:length(pred)) {
          pred.new[i] <- max(0,min(pred[i],1))
        }

        pred <- pred.new

        pred.01 <- numeric(length(pred))

        accuracy0_all <- numeric(101)   # a model is "0" e.g. neural networks
        tss0_all      <- numeric(101)
        hss0_all      <- numeric(101)

        ccp <- 0
        accuracy <- NA
        trueSS   <- NA
        HeidkeSS <- NA

        for (thresHOLD in seq(0.00,1.00,0.01))    {
        #thresHOLD <- 0.50
        #thresHOLD <- 0.25

        if ( mod(100*thresHOLD,10) == 0) {
        cat("threshold:...",100*thresHOLD," % completed...","\n")
        }
        ccp <- ccp + 1
        for (i in 1:length(pred)) {
          response=0
          if (!is.na(pred[i])) {
            if (pred[i] > thresHOLD) {
              response=1
            }
          }
          pred.01[i] <- response
        }

        cfmat <- table(pred.01, obs)

        #write.table(cfmat,paste("cfmat","_a_model","_thresHOLD_",thresHOLD,".txt",sep=""))

        accuracy <- NA
        trueSS   <- NA
        HeidkeSS <- NA

        if(dim(cfmat)[1]==2 && dim(cfmat)[2]==2) {

          accuracy <- sum(diag(cfmat)) / sum(cfmat)

          a=cfmat[2,2]
          d=cfmat[1,1]
          b=cfmat[2,1]
          c=cfmat[1,2]

          #trueSS <- TSS.Stat(cfmat)

          trueSS <- (a*d-b*c) / ((a+c)*(b+d))

          HeidkeSS <- 2 * (a*d-b*c) / ( (a+c)*(c+d) + (a+b)*(b+d) )

          accuracy0_all[ccp] <- accuracy
          tss0_all[ccp] <- trueSS
          hss0_all[ccp] <- HeidkeSS


          #The score has a range of -1 to +1, with 0 representing no skill.
          #Negative values would be associated with "perverse" forecasts,
          #and could be converted to positive skill simply by replacing
          #all the yes forecasts with no and vice-versa.
          #The KSS is also the difference between the hit rate and false alarm rate,
          #KSS=H-F.

          #hit rate and false alarm

          H = a / (a+c)    #http://www.eumetcal.org/resources/ukmeteocal/verification/www/english/msg/ver_categ_forec/uos2/uos2_ko2.htm
          F = (b) / (b+d)  #http://www.eumetcal.org/resources/ukmeteocal/verification/www/english/msg/ver_categ_forec/uos2/uos2_ko3.htm

          flag = abs(trueSS - (H-F)) < 10^(-4)


          }
          } ### thresHOLD loop

         #save all wrt thresHOLD arrays
         #a model e.g. neural network
         write.table(cbind(1:101,seq(0,1,0.01),accuracy0_all,tss0_all,hss0_all),file=paste("a_model_SKILLS_",".out",sep=""),row.names=FALSE,col.names=FALSE)

         #res
         res0 <- cbind(1:101,seq(0,1,0.01),accuracy0_all,tss0_all,hss0_all)

         jpeg(paste("a_model_","_SKILLS_",".jpeg",sep=""))
         plot(res0[,2],res0[,3],type="l",col=2,lwd=2.5,xlab="probability threshold",ylab="skills values",ylim=c(0,1),
                                                       main=paste("skills wrt threshold for a model\n","testing, original split",sep=" "))
         lines(res0[,2],res0[,4],type="l",lwd=2.5,col=3)
         lines(res0[,2],res0[,5],type="l",lwd=2.5,col=4)

         legend(0.7,0.6, # places a legend at the appropriate place
             c("acc","tss","hss"), # puts text in the legend
             lty=c(1,1,1), # gives the legend appropriate symbols (lines)
             lwd=c(2.5,2.5,2.5),col=c(2,3,4)) # gives the legend lines the correct color and width
         dev.off()
         }
""")


r_plot_SSP = robjects.globalenv['R_plot_SSP']
