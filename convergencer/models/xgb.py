import xgboost as xgb
from convergencer.models import base
import multiprocessing
import numpy as np
import torch
from convergencer.processors import catToIntPdCat
import pandas as pd
from sklearn.model_selection import train_test_split

class xgbRegression(base):
    def _initParameter(self, X, y, parameters):
        #mst=np.sum(np.power(y-np.mean(y),2))/len(y)
        self._setParameter("num_boost_round",300,parameters)
        self._setParameter("eta",0.3,parameters)
        #self._setParameter("gamma",0.001*mst,parameters)
        self._setParameter("min_child_weight",np.log2(X.shape[0]),parameters)
        self._setParameter("max_depth",8,parameters)
        self._setParameter("subsample",0.5,parameters)
        self._setParameter("colsample_bytree",1.0,parameters)
        self._setParameter("colsample_bylevel",1.0,parameters)
        #self._setParameter("alpha",0.0001,parameters)
        #self._setParameter("lambda",1.0,parameters)
        self._setParameter("early_stopping_round",0.05,parameters)
        return super()._initParameter(X, y, parameters)
        
    def _getParameterRange(self, X, y, parameters={}):
        #mst=np.sum(np.power(y-np.mean(y),2))/len(y)
        self._setParameter("num_boost_round",(int,"uni",100,1000),parameters)
        self._setParameter("eta",(float,"exp",0.001,1.0),parameters)
        #self._setParameter("gamma",(float,"exp",0.0,0.01*mst),parameters)
        self._setParameter("min_child_weight",(float,"uni",0.0,10.0*np.log2(X.shape[0])),parameters)
        self._setParameter("max_depth",(int,"uni",1,100),parameters)
        self._setParameter("subsample",(float,"uni",0.5,1.0),parameters)
        self._setParameter("colsample_bytree",(float,"uni",0.5,1.0),parameters)
        self._setParameter("colsample_bylevel",(float,"uni",0.5,1.0),parameters)
        #self._setParameter("alpha",(float,"exp",0.0001,10.0),parameters)
        #self._setParameter("lambda",(float,"exp",0.0001,10.0),parameters)
        self._setParameter("early_stopping_round",(float,"uni",0.02,0.1),parameters)
        return super()._getParameterRange(X, y, parameters=parameters)
    
    def _getProcessors(self):
        return [catToIntPdCat().initialize({},verbose=0)]

    def _getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return None
        model = xgb.Booster({"tree_method":"auto","nthread":multiprocessing.cpu_count()})
        model.load_model(modelPath)
        return model

    def _fitModel(self, X,y, model, parameters,metric):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
        xgtrain = xgb.DMatrix(X_train, label=y_train,enable_categorical=True)
        xgtest = xgb.DMatrix(X_test, label=y_test,enable_categorical=True)
        watchlist = [(xgtrain, 'train'),(xgtest, 'val')]
        parameters=parameters.copy()
        num_rounds=parameters.pop("num_boost_round")
        early_stop=int(parameters.pop("early_stopping_round")*num_rounds)
        self._setParameter("tree_method","auto",parameters)
        self._setParameter("nthread",multiprocessing.cpu_count(),parameters)
        return xgb.train(parameters, xgtrain, num_rounds, watchlist,early_stopping_rounds=early_stop,feval=metric.xgb(),maximize=metric.maximum(),verbose_eval=False)
    
    def _modelPredict(self, model, X):
        data = xgb.DMatrix(X,enable_categorical=True)
        return pd.Series(model.predict(data,iteration_range=(0, model.best_iteration)),index=X.index)

    def _saveModel(self, path):
        self.model.save_model(path)
    
    def __str__(self):
        return "xgbRegression"

class xgbRegression_dart(xgbRegression):
    def _initParameter(self, X, y, parameters):
        self._setParameter("rate_drop",0.0,parameters)
        self._setParameter("skip_drop",0.0,parameters)
        return super()._initParameter(X, y, parameters)
   
    def _getParameterRange(self, X, y, parameters={}):
        self._setParameter("rate_drop",(float,"uni",0.0,0.5),parameters)
        self._setParameter("skip_drop",(float,"uni",0.0,0.5),parameters)
        return super()._getParameterRange(X, y, parameters=parameters)

    def _fitModel(self, X_train, y_train, X_test, y_test, model, parameters,metric):
        parameters=parameters.copy()
        self._setParameter("booster","dart",parameters)
        return super()._fitModel(X_train, y_train, X_test, y_test, model, parameters,metric)
        
    def __str__(self):
        return "xgbRegression_dart"