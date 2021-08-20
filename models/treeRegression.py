from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from base import base
import multiprocessing
import numpy as np

class dtRegression(base):
    def __init__(self,X=None,y=None,parameters={},metric="r2",maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        mst=np.sum(np.power(y-np.mean(y),2))/len(y)
        self.setParameter("max_depth",10,parameters)
        self.setParameter("min_samples_split",0.0001,parameters)
        self.setParameter("min_samples_leaf",0.0001,parameters)
        self.setParameter("max_features",1.0,parameters)
        self.setParameter("min_impurity_decrease",0.001*mst,parameters)
        super().__init__(X, y, parameters=parameters,metric=metric, maxEpoch=maxEpoch, checkPointPath=checkPointPath, checkPointFreq=checkPointFreq)

    def getParameterRange(self,X,y,parameters={}):
        mst=np.sum(np.power(y-np.mean(y),2))/len(y)
        self.setParameter("max_depth",(int,"uni",5,200),parameters)
        self.setParameter("min_samples_split",(float,"exp",0.0,0.01),parameters)
        self.setParameter("min_samples_leaf",(float,"exp",0.0,0.01),parameters)
        self.setParameter("max_features",(float,"exp",0.5,1.0),parameters)
        self.setParameter("min_impurity_decrease",(float,"exp",0.0,0.01*mst)*mst,parameters)
        return super().getParameterRange(X,y,parameters=parameters)

    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return DecisionTreeRegressor(
            max_depth=parameters["max_depth"],
            min_samples_split=parameters["min_samples_split"],   #最少要有这么多个样本才分割
            min_samples_leaf=parameters["min_samples_split"],   #每个叶节点最少要有这么多个样本
            max_features=parameters["max_features"],    #划分时考虑的特征数
            min_impurity_decrease=parameters["min_impurity_decrease"]   #大于该最小增益才分割
            )
        return super().getModel(X, y, parameters, modelPath)
        
    def __str__(self):
        return "dtRegression"

class rfRegression(dtRegression):
    def __init__(self,X=None,y=None,parameters={},metric="r2",maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        self.setParameter("num",300,parameters)
        super().__init__(X, y, parameters=parameters,metric=metric, maxEpoch=maxEpoch, checkPointPath=checkPointPath, checkPointFreq=checkPointFreq)

    def getParameterRange(self, X, y, parameters={}):
        self.setParameter("num",(object,100,300,500,650,800),parameters)
        self.setParameter("max_depth",(int,"uni",1,10),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return RandomForestRegressor(
                n_estimators=parameters["num"],
                n_jobs=multiprocessing.cpu_count(),
                max_depth=parameters["max_depth"],
                min_samples_split=parameters["min_samples_split"],
                min_samples_leaf=parameters["min_samples_split"],
                max_features=parameters["max_features"],
                min_impurity_decrease=parameters["min_impurity_decrease"]
            )
        return super().getModel(X, y, parameters, modelPath)
        
    def __str__(self):
        return "rfRegression"

class gbRegression(rfRegression):
    def __init__(self,X=None,y=None,parameters={},metric="r2",maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        self.setParameter("num",100,parameters)
        self.setParameter("subsample",0.5,parameters)
        self.setParameter("learning_rate",0.3,parameters)
        super().__init__(X, y, parameters=parameters,metric=metric, maxEpoch=maxEpoch, checkPointPath=checkPointPath, checkPointFreq=checkPointFreq)

    def getParameterRange(self, X, y, parameters={}):
        self.setParameter("num",(object,10,50,100,200,400),parameters)
        self.setParameter("subsample",(float,"uni",0.5,1.0),parameters)
        self.setParameter("learning_rate",(float,"exp",0.0,1.0),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return GradientBoostingRegressor(
            n_estimators=parameters["num"],
            subsample=parameters["subsample"],
            min_samples_split=parameters["min_samples_split"],
            max_depth=parameters["max_depth"],
            learning_rate=parameters["learning_rate"],
            min_samples_leaf=parameters["min_samples_leaf"],
            max_features=parameters["max_features"],
            min_impurity_decrease=parameters["min_impurity_decrease"]
            )
        return super().getModel(X, y, parameters, modelPath)
        
    def __str__(self):
        return "gbRegressor"