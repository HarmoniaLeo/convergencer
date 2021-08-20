from sklearn import metrics
from sklearn.svm import SVR
from base import base

class SVMRegressionBase(base):
    def __init__(self,X=None,y=None,parameters={},metric="r2",maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        self.setParameter("C",1.0,parameters)
        super().__init__(X, y, parameters=parameters,metric=metric, maxEpoch=maxEpoch, checkPointPath=checkPointPath, checkPointFreq=checkPointFreq)
    
    def getParameterRange(self,X,y,parameters={}):
        self.setParameter("C",(float,"exp",0.0,1.0),parameters)
        return super().getParameterRange(X,y,parameters=parameters)
        
    def __str__(self):
        return "SVRegression"

class SVMRegression(SVMRegressionBase):
    def __init__(self,X=None,y=None,parameters={},metric="r2",maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        self.setParameter("kernel","rbf",parameters)
        super().__init__(X, y, parameters=parameters,metric=metric, maxEpoch=maxEpoch, checkPointPath=checkPointPath, checkPointFreq=checkPointFreq)
    
    def getParameterRange(self,X,y,parameters={}):
        self.setParameter("kernel",(object,"linear","rbf","sigmoid","precomputed"),parameters)
        return super().getParameterRange(X,y,parameters=parameters)

    def getModel(self, X, y, parameters, modelPath):
        if modelPath is None:
            return SVR(kernel=parameters["kernel"],
            C=parameters["C"])
        return super().getModel(X, y, parameters, modelPath)

    def __str__(self):
        return "SVRegression"

class SVMRegression_poly(SVMRegressionBase):
    def __init__(self,X=None,y=None,parameters={},metric="r2",maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        self.setParameter("degree",3,parameters)
        super().__init__(X, y, parameters=parameters,metric=metric, maxEpoch=maxEpoch, checkPointPath=checkPointPath, checkPointFreq=checkPointFreq)
    
    def getParameterRange(self,X,y,parameters={}):
        self.setParameter("degree",(int,"uni",2,8),parameters)
        return super().getParameterRange(X,y,parameters=parameters)
    
    def getModel(self, X, y, parameters, modelPath):
        if modelPath is None:
            return SVR(kernel="poly",
        degree=parameters["degree"],
        C=parameters["C"])
        return super().getModel(X, y, parameters, modelPath)        

    def __str__(self):
        return "SVRegression_poly"