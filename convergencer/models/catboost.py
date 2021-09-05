from convergencer.processors import catToPdCat
from catboost import CatBoostRegressor,Pool
from convergencer.models import base
import multiprocessing
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from convergencer.utils.metrics import mspe
from sklearn.model_selection import KFold
import pandas as pd

class catBoostRegression(base):
    def _initParameter(self, X, y, parameters):
        self._setParameter("iterations",1000,parameters)
        self._setParameter("learning_rate",0.3,parameters)
        self._setParameter("depth",8,parameters)
        self._setParameter("max_bin",128,parameters)
        self._setParameter("bagging_temperature",1.0,parameters)
        self._setParameter("random_strength",1.0,parameters)
        self._setParameter("rsm",1.0,parameters)
        self._setParameter("l2_leaf_reg",3.0,parameters)
        self._setParameter("od_pval",1e-2,parameters)
        return super()._initParameter(X, y, parameters)
    
    def _getParameterRange(self, X, y, parameters={}):
        self._setParameter("iterations",(int,"exp",100,20000),parameters)
        self._setParameter("learning_rate",(float,"exp",0.001,1.0),parameters)
        self._setParameter("depth",(int,"uni",1,10),parameters)
        self._setParameter("max_bin",(object,32,64,128,255,512),parameters)
        self._setParameter("bagging_temperature",(float,"uni",0.0,10.0),parameters)
        self._setParameter("random_strength",(float,"exp",0.0,1.0),parameters)
        self._setParameter("rsm",(float,"uni",0.5,1.0),parameters)
        self._setParameter("l2_leaf_reg",(float,"exp",0.0001,20.0),parameters)
        self._setParameter("od_pval",(float,"exp",1e-10,1e-2),parameters)
        return super()._getParameterRange(X, y, parameters=parameters)

    def _getProcessors(self):
        return [catToPdCat().initialize({},verbose=0)]

    def _getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            parameters=parameters.copy()
            self._setParameter("thread_count",multiprocessing.cpu_count(),parameters)
            parameters["eval_metric"]=metric.catboost()
            parameters["allow_writing_files"]=False
            return CatBoostRegressor(**parameters)
        else:
            model = CatBoostRegressor(thread_count=multiprocessing.cpu_count(),allow_writing_files=False)
            model.load_model(modelPath)
            return model
    
    def _fitModel(self, X,y, model, parameters,metric):
        ttn = X.select_dtypes(exclude=[np.number])
        cols=ttn.columns
        categorical_feature=cols.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        train_pool = Pool(X_train,y_train,cat_features=categorical_feature)
        test_pool = Pool(X_test,y_test,cat_features=categorical_feature)
        model.fit(train_pool,eval_set=test_pool,logging_level="Silent",use_best_model=True)
        return model

    def _modelPredict(self, model, X):
        ttn = X.select_dtypes(exclude=[np.number])
        cols=ttn.columns
        categorical_feature=cols.tolist()
        data=Pool(X,cat_features=categorical_feature)
        return pd.Series(model.predict(data),X.index)
    
    def _saveModel(self, path):
        self.model.save_model(path)
        
    def __str__(self):
        return "catBoostRegression"