from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from convergencer.models import base
import multiprocessing
import numpy as np
from convergencer.processors import catToInt
import math

class dtRegression(base):
    def _initParameter(self, X, y, parameters):
        #mst=np.sum(np.power(y-np.mean(y),2))/len(y)
        self._setParameter("max_depth",10,parameters)
        self._setParameter("min_samples_split",np.log2(X.shape[0])/X.shape[0],parameters)
        #self._setParameter("min_samples_leaf",np.log2(X.shape[0])/X.shape[0],parameters)
        self._setParameter("max_features",1.0,parameters)
        #self._setParameter("min_impurity_decrease",0.1,parameters)
        return super()._initParameter(X, y, parameters)
    
    def _getParameterRange(self,X,y,parameters={}):
        #mst=np.sum(np.power(y-np.mean(y),2))/len(y)
        self._setParameter("max_depth",(int,"uni",5,300),parameters)
        self._setParameter("min_samples_split",(float,"exp",0,10.0*np.log2(X.shape[0])/X.shape[0]),parameters)
        #self._setParameter("min_samples_leaf",(float,"uni",0.00001,10.0*np.log2(X.shape[0])/X.shape[0]),parameters)
        self._setParameter("max_features",(float,"exp",0.5,1.0),parameters)
        #self._setParameter("min_impurity_decrease",(float,"uni",0.0,1.0),parameters)
        return super()._getParameterRange(X,y,parameters=parameters)

    def _getModel(self, X, y, parameters, modelPath,metric):
        #mst=np.sum(np.power(y-np.mean(y),2))/len(y)
        if modelPath is None:
            return DecisionTreeRegressor(
            max_depth=parameters["max_depth"],
            min_samples_split=parameters["min_samples_split"],   #最少要有这么多个样本才分割
            #min_samples_leaf=parameters["min_samples_leaf"],   #每个叶节点最少要有这么多个样本
            max_features=parameters["max_features"],    #划分时考虑的特征数
            #min_impurity_decrease=parameters["min_impurity_decrease"]*mst/parameters["max_depth"]   #大于该最小增益才分割
            )
        return super()._getModel(X, y, parameters, modelPath,metric)

    def _getProcessors(self):
        return [catToInt().initialize({},verbose=0)]

    def __str__(self):
        return "dtRegression"

class rfRegression(dtRegression):
    def _initParameter(self, X, y, parameters):
        self._setParameter("num",300,parameters)
        return super()._initParameter(X, y, parameters)
    
    def _getParameterRange(self, X, y, parameters={}):
        self._setParameter("num",(int,"uni",100,1000),parameters)
        self._setParameter("max_depth",(int,"uni",1,200),parameters)
        return super()._getParameterRange(X, y, parameters=parameters)

    def _getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return RandomForestRegressor(
                n_estimators=parameters["num"],
                n_jobs=multiprocessing.cpu_count(),
                max_depth=parameters["max_depth"],
                min_samples_split=parameters["min_samples_split"],
                #min_samples_leaf=parameters["min_samples_leaf"],
                max_features=parameters["max_features"],
                #min_impurity_decrease=parameters["min_impurity_decrease"]
            )
        return super()._getModel(X, y, parameters, modelPath,metric)
        
    def __str__(self):
        return "rfRegression"

class gbRegression(rfRegression):
    def _initParameter(self, X, y, parameters):
        self._setParameter("num",100,parameters)
        self._setParameter("subsample",0.5,parameters)
        self._setParameter("learning_rate",0.3,parameters)
        return super()._initParameter(X, y, parameters)

    def _getParameterRange(self, X, y, parameters={}):
        self._setParameter("num",(int,"uni",10,1000),parameters)
        self._setParameter("subsample",(float,"uni",0.5,1.0),parameters)
        self._setParameter("learning_rate",(float,"exp",0.001,1.0),parameters)
        return super()._getParameterRange(X, y, parameters=parameters)

    def _getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return GradientBoostingRegressor(
            n_estimators=parameters["num"],
            subsample=parameters["subsample"],
            min_samples_split=parameters["min_samples_split"],
            max_depth=parameters["max_depth"],
            learning_rate=parameters["learning_rate"],
            #min_samples_leaf=parameters["min_samples_leaf"],
            max_features=parameters["max_features"],
            #min_impurity_decrease=parameters["min_impurity_decrease"]
            )
        return super()._getModel(X, y, parameters, modelPath,metric)
        
    def __str__(self):
        return "gbRegression"