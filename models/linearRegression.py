from operator import mod
from base import base
from sklearn.linear_model import Ridge,Lasso,ElasticNet,MultiTaskLasso,MultiTaskElasticNet
import numpy as np

class ridge(base):
    def __init__(self,X=None,y=None,parameters={},metric="r2",maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        self.setParameter("alpha",1.0,parameters)
        super().__init__(X, y, parameters=parameters,metric=metric, maxEpoch=maxEpoch, checkPointPath=checkPointPath, checkPointFreq=checkPointFreq)

    def getParameterRange(self, X, y, parameters={}):
        self.setParameter("alpha",(float,"exp",0.0,1.0),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return Ridge(alpha=parameters["alpha"])
        return super().getModel(X, y, parameters, modelPath)
        
    def __str__(self):
        return "ridge"

class lasso(ridge):
    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return Lasso(alpha=np.exp(parameters["alpha"]))
        return super().getModel(X, y, parameters, modelPath)
    
    def __str__(self):
        return "lasso"

class elasticNet(ridge):
    def __init__(self,X=None,y=None,parameters={},metric="r2",maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        self.setParameter("l1Rate",0.5,parameters)
        super().__init__(X, y, parameters=parameters,metric=metric, maxEpoch=maxEpoch, checkPointPath=checkPointPath, checkPointFreq=checkPointFreq)

    def getParameterRange(self,X,y,parameters={}):
        self.setParameter("l1Rate",(float,"uni",0.0,1.0),parameters)
        return super().getParameterRange(X,y,parameters=parameters)

    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return ElasticNet(alpha=parameters["alpha"],l1_ratio=parameters["l1Rate"])
        return super().getModel(X, y, parameters, modelPath)        
    
    def __str__(self):
        return "elasticNet"

class multiTaskLasso(lasso):
    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return MultiTaskLasso(alpha=parameters["alpha"])
        return super().getModel(X, y, parameters, modelPath)
    
    def __str__(self):
        return "multiTaskLasso"

class multiTaskElasticNet(elasticNet):
    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return MultiTaskElasticNet(alpha=parameters["alpha"],l1_ratio=parameters["l1Rate"])
        return super().getModel(X, y, parameters, modelPath)

    def __str__(self):
        return "multiTaskElasticNet"