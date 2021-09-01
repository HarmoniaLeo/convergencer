from convergencer.processors import catToInt, catToIntPdCat
import lightgbm as lgb
from convergencer.models import base
import numpy as np
import pandas as pd
import math
import torch
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_squared_log_error
from convergencer.utils.metrics import mape,mspe
import warnings

def fr2(preds, train_data):
    label = train_data.get_label()
    score = r2_score(label,preds)
    return 'r2',score,True

def fmse(preds, train_data):
    label = train_data.get_label()
    score = mean_squared_error(label,preds)
    return 'mse',score,False

def fmae(preds, train_data):
    label = train_data.get_label()
    score = mean_absolute_error(label,preds)
    return 'mse',score,False

def fmsle(preds, train_data):
    label = train_data.get_label()
    score = mean_squared_log_error(label,preds)
    return 'msle',score,False

def fmape(preds, train_data):
    label = train_data.get_label()
    score = mape(label,preds)
    return 'mape',score,False

def fmspe(preds, train_data):
    label = train_data.get_label()
    score = mspe(label,preds)
    return 'mspe',score,False

class lgbmRegressionBase(base):
    def initParameter(self, X, y, parameters):
        #mst=np.sum(np.power(y-np.mean(y),2))/len(y)
        self.setParameter("num",1000,parameters)
        self.setParameter("learning_rate",0.3,parameters)
        #self.setParameter("min_gain_to_split",0.001*mst,parameters)
        #self.setParameter("min_data_in_leaf",int(np.log2(X.shape[0]))+1,parameters)
        self.setParameter("num_leaves",10,parameters)
        self.setParameter("max_depth",8,parameters)
        self.setParameter("max_bin",256,parameters)
        self.setParameter("feature_fraction",1.0,parameters)
        #self.setParameter("lambda_l1",0.0001,parameters)
        #self.setParameter("lambda_l2",1.0,parameters)
        #self.setParameter("early_stopping_round",0.05,parameters)
        return super().initParameter(X, y, parameters)
    
    def getParameterRange(self, X, y, parameters={}):
        #mst=np.sum(np.power(y-np.mean(y),2))/len(y)
        self.setParameter("num",(int,"exp",100,10000),parameters)
        self.setParameter("learning_rate",(float,"exp",0.001,1.0),parameters)
        #self.setParameter("min_gain_to_split",(float,"exp",0.0,0.01*mst),parameters)
        #self.setParameter("min_data_in_leaf",(int,"uni",1,int(10.0*np.log2(X.shape[0]))),parameters)
        self.setParameter("num_leaves",(int,"uni",10,100),parameters)
        self.setParameter("max_depth",(int,"uni",1,10),parameters)
        self.setParameter("max_bin",(object,32,64,128,256,512),parameters)
        self.setParameter("feature_fraction",(float,"uni",0.5,1.0),parameters)
        #self.setParameter("lambda_l1",(float,"exp",0.0001,10.0),parameters)
        #self.setParameter("lambda_l2",(float,"uni",0.0001,10.0),parameters)
        #self.setParameter("early_stopping_round",(float,"uni",0.02,0.1),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def getProcessors(self,X,y):
        return [catToIntPdCat(X,verbose=0)]
    
    def preprocess(self, X):
        ttn = X.select_dtypes(exclude=[np.number])
        cols=ttn.columns
        self.categorical_feature=cols.tolist()
        self.feature_name=X.columns.tolist()
        return super().preprocess(X)

    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return None
        return lgb.Booster(model_file=modelPath,silent=True,params={"verbose":-1,"device":"gpu" if torch.cuda.is_available() else "cpu"})
    
    def fitModel(self, X_train, y_train, X_test, y_test, model, parameters,metric):
        parameters=parameters.copy()
        rounds=parameters.pop("num")
        es=int(parameters.pop("early_stopping_round")*rounds)
        if(torch.cuda.is_available()):self.setParameter("device","gpu",parameters)
        self.setParameter("boosting","gbdt",parameters)
        self.setParameter("verbose",-1,parameters)
        train_data=lgb.Dataset(X_train,y_train,params=parameters,free_raw_data=False,feature_name=self.feature_name,categorical_feature=self.categorical_feature)
        test_data=lgb.Dataset(X_test,y_test,params=parameters,free_raw_data=False,feature_name=self.feature_name,categorical_feature=self.categorical_feature)
        if metric=="r2":
            score=fr2
        elif metric=="mse":
            score=fmse
        elif metric=="mae":
            score=fmae
        elif metric=="msle":
            score=fmsle
        elif metric=="mape":
            score=fmape
        elif metric=="mspe":
            score=fmspe
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return lgb.train(parameters, train_data,rounds, valid_sets=[test_data],feval=score,early_stopping_rounds=es,verbose_eval=False)
    
    def saveModel(self, path):
        print("Save model as: ",path)
        self.model.save_model(path,num_iteration=self.model.best_iteration)
    
    def modelPredict(self, model, X,index):
        return pd.Series(model.predict(X,num_iteration=model.best_iteration),index=index)
    
class lgbmRegression(lgbmRegressionBase):
    def initParameter(self, X, y, parameters):
        self.setParameter("early_stopping_round",0.05,parameters)
        self.setParameter("bagging_freq",10,parameters)
        self.setParameter("bagging_fraction",1.0,parameters)
        return super().initParameter(X, y, parameters)
    
    def getParameterRange(self, X, y, parameters={}):
        self.setParameter("early_stopping_round",(float,"uni",0.02,0.1),parameters)
        self.setParameter("bagging_freq",(object,10,20,50),parameters)
        self.setParameter("bagging_fraction",(float,"uni",0.5,1.0),parameters)
        return super().getParameterRange(X, y, parameters=parameters)
    
    def __str__(self):
        return "lgbmRegression"

class lgbmRegression_dart(lgbmRegressionBase):
    def initParameter(self, X, y, parameters):
        self.setParameter("drop_rate",0.1,parameters)
        self.setParameter("bagging_freq",10,parameters)
        self.setParameter("bagging_fraction",1.0,parameters)
        return super().initParameter(X, y, parameters)

    def getParameterRange(self, X, y, parameters={}):
        self.setParameter("drop_rate",(float,"uni",0.0,0.5),parameters)
        self.setParameter("bagging_freq",(object,10,20,50),parameters)
        self.setParameter("bagging_fraction",(float,"uni",0.5,1.0),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def fitModel(self, X_train, y_train, X_test, y_test, model, parameters,metric):
        parameters=parameters.copy()
        rounds=parameters.pop("num")
        if(torch.cuda.is_available()):self.setParameter("device","gpu",parameters)
        self.setParameter("boosting","dart",parameters)
        self.setParameter("verbose",-1,parameters)
        train_data=lgb.Dataset(X_train,y_train,params=parameters,free_raw_data=False)
        test_data=lgb.Dataset(X_test,y_test,params=parameters,free_raw_data=False)
        if metric=="r2":
            score=fr2
        elif metric=="mse":
            score=fmse
        elif metric=="mae":
            score=fmae
        elif metric=="msle":
            score=fmsle
        elif metric=="mape":
            score=fmape
        elif metric=="mspe":
            score=fmspe
        return lgb.train(parameters, train_data,rounds, valid_sets=[test_data],feval=score,verbose_eval=False)
    
        
    def __str__(self):
        return "lgbmRegression_dart"

class lgbmRegression_goss(lgbmRegressionBase):
    def initParameter(self, X, y, parameters):
        self.setParameter("early_stopping_round",0.05,parameters)
        return super().initParameter(X, y, parameters)
    
    def getParameterRange(self, X, y, parameters={}):
        self.setParameter("early_stopping_round",(float,"uni",0.02,0.1),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def fitModel(self, X_train, y_train, X_test, y_test, model, parameters,metric):
        parameters=parameters.copy()
        self.setParameter("boosting","goss",parameters)
        return super().fitModel(X_train, y_train, X_test, y_test, model, parameters,metric)
        
    def __str__(self):
        return "lgbmRegression_goss"