from sklearn import metrics
from sklearn.svm import SVR
from models.base import base
from processors import catToInt,normalizeScaler,catToMean

class SVMRegression(base):
    def initParameter(self, X, y, parameters):
        self.setParameter("C",1.0,parameters)
        return super().initParameter(X, y, parameters)
    
    def getParameterRange(self,X,y,parameters={}):
        self.setParameter("C",(float,"uni",0.1,10.0),parameters)
        return super().getParameterRange(X,y,parameters=parameters)
    
    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return SVR(kernel="rbf",
            C=parameters["C"])
        return super().getModel(X, y, parameters, modelPath,metric)
    
    def getProcessors(self,X,y):
        return [catToMean(X,y,verbose=0),normalizeScaler(X,verbose=0)]
        
    def __str__(self):
        return "SVMRegression"
