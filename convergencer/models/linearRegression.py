from convergencer.models import base
from sklearn.linear_model import Ridge,Lasso,ElasticNet,LinearRegression
from convergencer.processors import catToOneHot,robustScaler,normalizeScaler
import numpy as np

class linear(base):
    def parasearchFit(self,X,y, metric="r2",maxEpoch=1000,modelSavePath=None,modelSaveFreq=50,historySavePath=None,historySaveFreq=50):
        return super().fit(X,y,metric,None,modelSavePath)

    def _getClass(self):
        return linear()

    def _getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return LinearRegression()
        return super()._getModel(X, y, parameters, modelPath,metric)

    def _getProcessors(self):
        return [catToOneHot().initialize({},verbose=0),robustScaler().initialize({},verbose=0)]
    
    def __str__(self):
        return "linear"


class ridge(base):
    def _getClass(self):
        return ridge()

    def _initParameter(self,X,y, parameters):
        self._setParameter("alpha",1.0,parameters)
        return super()._initParameter(X, y, parameters)

    def _getParameterRange(self, X, y, parameters={}):
        self._setParameter("alpha",(float,"uni",0.0001,20.0),parameters)
        return super()._getParameterRange(X, y, parameters=parameters)

    def _getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return Ridge(alpha=parameters["alpha"])
        return super()._getModel(X, y, parameters, modelPath,metric)
    
    def _getProcessors(self):
        return [catToOneHot().initialize({},verbose=0),robustScaler().initialize({},verbose=0)]

    def __str__(self):
        return "ridge"

class lasso(ridge):
    def _getClass(self):
        return lasso()

    def _initParameter(self,X,y, parameters):
        self._setParameter("alpha",0.0001,parameters)
        return super()._initParameter(X, y, parameters)

    def _getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return Lasso(alpha=parameters["alpha"],max_iter=1e7)
        return super()._getModel(X, y, parameters, modelPath,metric)
    
    def _getParameterRange(self, X, y, parameters={}):
        self._setParameter("alpha",(float,"uni",0.0001,0.001),parameters)
        return super()._getParameterRange(X, y, parameters=parameters)

    def __str__(self):
        return "lasso"

class elasticNet(lasso):
    def _getClass(self):
        return elasticNet()

    def _initParameter(self, X, y, parameters):
        self._setParameter("l1Rate",0.5,parameters)
        return super()._initParameter(X, y, parameters)
    
    def _getParameterRange(self,X,y,parameters={}):
        self._setParameter("l1Rate",(float,"uni",0.1,1.0),parameters)
        return super()._getParameterRange(X,y,parameters=parameters)

    def _getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return ElasticNet(alpha=parameters["alpha"],l1_ratio=parameters["l1Rate"],max_iter=1e7)
        return super()._getModel(X, y, parameters, modelPath,metric)   
    
    def __str__(self):
        return "elasticNet"