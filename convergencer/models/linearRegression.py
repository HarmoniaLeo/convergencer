from convergencer.models import base
from sklearn.linear_model import Ridge,Lasso,ElasticNet,LinearRegression
from convergencer.processors import catToMean
import numpy as np

class linear(base):
    def __init__(self,X,y,parameters={},metric="r2",maxEpoch=1000,modelLoadPath=None,modelSavePath=None,modelSaveFreq=50,historyLoadPath=None,historySavePath=None,historySaveFreq=50,verbose=1):
        super().__init__(X=X, y=y, parameters=parameters, metric=metric, maxEpoch=0,modelLoadPath=modelLoadPath,modelSavePath=modelSavePath,modelSaveFreq=modelSaveFreq,historyLoadPath=historyLoadPath,historySavePath=historySavePath,historySaveFreq=historySaveFreq,verbose=verbose)

    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return LinearRegression()
        return super().getModel(X, y, parameters, modelPath,metric)

    def getProcessors(self,X,y):
        return [catToMean(X,y,verbose=0)]
    
    def __str__(self):
        return "linear"


class ridge(base):
    def initParameter(self,X,y, parameters):
        self.setParameter("alpha",1.0,parameters)
        return super().initParameter(X, y, parameters)

    def getParameterRange(self, X, y, parameters={}):
        self.setParameter("alpha",(float,"uni",0.0001,10.0),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return Ridge(alpha=parameters["alpha"])
        return super().getModel(X, y, parameters, modelPath,metric)
    
    def getProcessors(self,X,y):
        return [catToMean(X,y,verbose=0)]

    def __str__(self):
        return "ridge"

class lasso(ridge):
    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return Lasso(alpha=parameters["alpha"])
        return super().getModel(X, y, parameters, modelPath,metric)
    
    def getParameterRange(self, X, y, parameters={}):
        self.setParameter("alpha",(float,"uni",0.01,10.0),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def __str__(self):
        return "lasso"

class elasticNet(ridge):
    def initParameter(self, X, y, parameters):
        self.setParameter("l1Rate",0.5,parameters)
        return super().initParameter(X, y, parameters)
    
    def getParameterRange(self,X,y,parameters={}):
        self.setParameter("alpha",(float,"uni",0.01,10.0),parameters)
        self.setParameter("l1Rate",(float,"uni",0.1,0.9),parameters)
        return super().getParameterRange(X,y,parameters=parameters)

    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return ElasticNet(alpha=parameters["alpha"],l1_ratio=parameters["l1Rate"])
        return super().getModel(X, y, parameters, modelPath,metric)   
    
    def __str__(self):
        return "elasticNet"