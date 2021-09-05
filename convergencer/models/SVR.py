from sklearn import metrics
from sklearn.svm import SVR,NuSVR
from convergencer.models import base
from convergencer.processors import catToInt,robustScaler,normalizeScaler,catToOneHot

class SVMRegression(base):
    def _initParameter(self, X, y, parameters):
        self._setParameter("C",1.0,parameters)
        return super()._initParameter(X, y, parameters)
    
    def _getParameterRange(self,X,y,parameters={}):
        self._setParameter("C",(float,"uni",0.1,20.0),parameters)
        return super()._getParameterRange(X,y,parameters=parameters)
    
    def _getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return SVR(kernel="rbf",
            C=parameters["C"])
        return super()._getModel(X, y, parameters, modelPath,metric)
    
    def _getProcessors(self):
        return [catToInt().initialize({},verbose=0),normalizeScaler().initialize({},verbose=0)]
        
    def __str__(self):
        return "SVMRegression"

class NuSVMRegression(SVMRegression):
    def _initParameter(self, X, y, parameters):
        self._setParameter("nu",1.0,parameters)
        return super()._initParameter(X, y, parameters)
    
    def _getParameterRange(self,X,y,parameters={}):
        self._setParameter("nu",(float,"uni",0.1,1.0),parameters)
        return super()._getParameterRange(X,y,parameters=parameters)
    
    def _getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return NuSVR(kernel="rbf",
            C=parameters["C"],nu=parameters["nu"])
        return super()._getModel(X, y, parameters, modelPath,metric)
    
    def _getProcessors(self):
        return [catToInt().initialize({},verbose=0),normalizeScaler().initialize({},verbose=0)]
        
    def __str__(self):
        return "NuSVMRegression"