from convergencer.processors import catToInt, catToIntPdCat,catToOneHot
import lightgbm as lgb
from convergencer.models import base
import numpy as np
import pandas as pd
import math
import torch
import warnings
from sklearn.model_selection import train_test_split

class lgbmRegressionBase(base):
    def _getClass(self):
        return lgbmRegression()

    def _initParameter(self, X, y, parameters):
        #mst=np.sum(np.power(y-np.mean(y),2))/len(y)
        self._setParameter("num",1000,parameters)
        self._setParameter("learning_rate",0.3,parameters)
        #self._setParameter("min_gain_to_split",0.001*mst,parameters)
        #self._setParameter("min_data_in_leaf",int(np.log2(X.shape[0]))+1,parameters)
        self._setParameter("num_leaves",10,parameters)
        self._setParameter("max_depth",8,parameters)
        self._setParameter("max_bin",256,parameters)
        self._setParameter("feature_fraction",1.0,parameters)
        #self._setParameter("lambda_l1",0.0001,parameters)
        #self._setParameter("lambda_l2",1.0,parameters)
        #self._setParameter("early_stopping_round",0.05,parameters)
        return super()._initParameter(X, y, parameters)
    
    def _getParameterRange(self, X, y, parameters={}):
        #mst=np.sum(np.power(y-np.mean(y),2))/len(y)
        self._setParameter("num",(int,"exp",100,10000),parameters)
        self._setParameter("learning_rate",(float,"exp",0.001,1.0),parameters)
        #self._setParameter("min_gain_to_split",(float,"exp",0.0,0.01*mst),parameters)
        #self._setParameter("min_data_in_leaf",(int,"uni",1,int(10.0*np.log2(X.shape[0]))),parameters)
        self._setParameter("num_leaves",(int,"uni",10,100),parameters)
        self._setParameter("max_depth",(int,"uni",1,10),parameters)
        self._setParameter("max_bin",(object,32,64,128,255,512),parameters)
        self._setParameter("feature_fraction",(float,"uni",0.5,1.0),parameters)
        #self._setParameter("lambda_l1",(float,"exp",0.0001,10.0),parameters)
        #self._setParameter("lambda_l2",(float,"uni",0.0001,10.0),parameters)
        #self._setParameter("early_stopping_round",(float,"uni",0.02,0.1),parameters)
        return super()._getParameterRange(X, y, parameters=parameters)

    def _getProcessors(self):
        return [catToIntPdCat().initialize({},verbose=0)]

    def _getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return None
        return lgb.Booster(model_file=modelPath,silent=True,params={"verbose":-1})
    
    def _fitModel(self, X,y, model, parameters,metric):
        ttn = X.select_dtypes(exclude=[np.number])
        cols=ttn.columns
        categorical_feature=cols.tolist()
        feature_name=X.columns.tolist()
        parameters=parameters.copy()
        rounds=parameters.pop("num")
        es=int(parameters.pop("early_stopping_round")*rounds)
        self._setParameter("boosting","gbdt",parameters)
        self._setParameter("verbose",-1,parameters)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
        train_data=lgb.Dataset(X_train,y_train,params=parameters,free_raw_data=False,feature_name=feature_name,categorical_feature=categorical_feature)
        test_data=lgb.Dataset(X_test,y_test,params=parameters,free_raw_data=False,feature_name=feature_name,categorical_feature=categorical_feature)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return lgb.train(parameters, train_data,rounds, valid_sets=[test_data],feval=metric.lgbm(),early_stopping_rounds=es,verbose_eval=False)
    
    def _saveModel(self, path):
        self.model.save_model(path,num_iteration=self.model.best_iteration)
    
    def _modelPredict(self, model, X):
        return pd.Series(model.predict(X,num_iteration=model.best_iteration),X.index)
    
class lgbmRegression(lgbmRegressionBase):
    def _getClass(self):
        return lgbmRegression()

    def _initParameter(self, X, y, parameters):
        self._setParameter("early_stopping_round",0.05,parameters)
        self._setParameter("bagging_freq",10,parameters)
        self._setParameter("bagging_fraction",1.0,parameters)
        return super()._initParameter(X, y, parameters)
    
    def _getParameterRange(self, X, y, parameters={}):
        self._setParameter("early_stopping_round",(float,"uni",0.02,0.1),parameters)
        self._setParameter("bagging_freq",(object,10,20,50),parameters)
        self._setParameter("bagging_fraction",(float,"uni",0.5,1.0),parameters)
        return super()._getParameterRange(X, y, parameters=parameters)
    
    def __str__(self):
        return "lgbmRegression"

class lgbmRegression_dart(lgbmRegressionBase):
    def _getClass(self):
        return lgbmRegression_dart()

    def _initParameter(self, X, y, parameters):
        self._setParameter("drop_rate",0.1,parameters)
        self._setParameter("bagging_freq",10,parameters)
        self._setParameter("bagging_fraction",1.0,parameters)
        return super()._initParameter(X, y, parameters)

    def _getParameterRange(self, X, y, parameters={}):
        self._setParameter("drop_rate",(float,"uni",0.0,0.5),parameters)
        self._setParameter("bagging_freq",(object,10,20,50),parameters)
        self._setParameter("bagging_fraction",(float,"uni",0.5,1.0),parameters)
        return super()._getParameterRange(X, y, parameters=parameters)

    def _fitModel(self, X, y, model, parameters,metric):
        parameters=parameters.copy()
        rounds=parameters.pop("num")
        if(torch.cuda.is_available()):self._setParameter("device","gpu",parameters)
        self._setParameter("boosting","dart",parameters)
        self._setParameter("verbose",-1,parameters)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        train_data=lgb.Dataset(X_train,y_train,params=parameters,free_raw_data=False)
        test_data=lgb.Dataset(X_test,y_test,params=parameters,free_raw_data=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return lgb.train(parameters, train_data,rounds, valid_sets=[test_data],feval=metric.lgbm,verbose_eval=False)
    
    def __str__(self):
        return "lgbmRegression_dart"

class lgbmRegression_goss(lgbmRegressionBase):
    def _getClass(self):
        return lgbmRegression_goss()

    def _initParameter(self, X, y, parameters):
        self._setParameter("early_stopping_round",0.05,parameters)
        return super()._initParameter(X, y, parameters)
    
    def _getParameterRange(self, X, y, parameters={}):
        self._setParameter("early_stopping_round",(float,"uni",0.02,0.1),parameters)
        return super()._getParameterRange(X, y, parameters=parameters)

    def _fitModel(self, X,y, model, parameters,metric):
        parameters=parameters.copy()
        self._setParameter("boosting","goss",parameters)
        return super()._fitModel(X, y, model, parameters,metric)
        
    def __str__(self):
        return "lgbmRegression_goss"