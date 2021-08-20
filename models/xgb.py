import xgboost as xgb
from base import base
import multiprocessing
import numpy as np
import torch
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_squared_log_error
from utils.metrics import mape,mspe

def fr2(preds, xgtrain):
    label = xgtrain.get_label()
    score = r2_score(label,preds)
    return 'r2',score

def fmse(preds, xgtrain):
    label = xgtrain.get_label()
    score = mean_squared_error(label,preds)
    return 'mse',score

def fmae(preds, xgtrain):
    label = xgtrain.get_label()
    score = mean_absolute_error(label,preds)
    return 'mse',score

def fmsle(preds, xgtrain):
    label = xgtrain.get_label()
    score = mean_squared_log_error(label,preds)
    return 'msle',score

def fmape(preds, xgtrain):
    label = xgtrain.get_label()
    score = mape(label,preds)
    return 'mape',score

def fmspe(preds, xgtrain):
    label = xgtrain.get_label()
    score = mspe(label,preds)
    return 'mspe',score

class xgbRegression(base):
    def __init__(self,X=None,y=None,parameters={},metric="r2",maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        mst=np.sum(np.power(y-np.mean(y),2))/len(y)
        self.setParameter("num_boost_round",300,parameters)
        self.setParameter("eta",0.3,parameters)
        self.setParameter("gamma",0.001*mst,parameters)
        self.setParameter("min_child_weight",X.shape[0]*0.0001,parameters)
        self.setParameter("max_depth",8,parameters)
        self.setParameter("subsample",0.5,parameters)
        self.setParameter("colsample_bytree",1.0,parameters)
        self.setParameter("colsample_bylevel",1.0,parameters)
        self.setParameter("alpha",1.0,parameters)
        self.setParameter("lambda",1.0,parameters)
        self.setParameter("early_stopping_round",10,parameters)
        super().__init__(X, y, parameters=parameters,metric=metric, maxEpoch=maxEpoch, checkPointPath=checkPointPath, checkPointFreq=checkPointFreq)

    def getParameterRange(self, X, y, parameters={}):
        mst=np.sum(np.power(y-np.mean(y),2))/len(y)
        self.setParameter("num_boost_round",(object,100,300,500,650,800),parameters)
        self.setParameter("eta",(float,"exp",0.0,1.0),parameters)
        self.setParameter("gamma",(float,"exp",0.0,0.01*mst),parameters)
        self.setParameter("min_child_weight",(float,"exp",0.0,X.shape[0]*0.0001),parameters)
        self.setParameter("max_depth",(int,"uni",1,10),parameters)
        self.setParameter("subsample",(float,"uni",0.5,1.0),parameters)
        self.setParameter("colsample_bytree",(float,"uni",0.5,1.0),parameters)
        self.setParameter("colsample_bylevel",(float,"uni",0.5,1.0),parameters)
        self.setParameter("alpha",(float,"exp",0.0,1.0),parameters)
        self.setParameter("lambda",(float,"exp",0.0,1.0),parameters)
        self.setParameter("early_stopping_round",(object,5,10,15),parameters)
        return super().getParameterRange(X, y, parameters=parameters)
    
    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return None
        model = xgb.Booster()
        model.load_model(modelPath)
        return model

    def fitModel(self, X_train, y_train, X_test, y_test, model, parameters,metric):
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        xgtest = xgb.DMatrix(X_test, label=y_test)
        watchlist = [(xgtrain, 'train'),(xgtest, 'val')]
        parameters=parameters.copy()
        num_rounds=parameters.pop("num_boost_round")
        early_stop=parameters.pop("early_stopping_round")
        if(torch.cuda.is_available()):self.setParameter("tree_method","gpu",parameters)
        self.setParameter("nthread",multiprocessing.cpu_count(),parameters)
        if metric=="r2":
            score=fr2
            maximize=True
        elif metric=="mse":
            score=fmse
            maximize=False
        elif metric=="mae":
            score=fmae
            maximize=False
        elif metric=="msle":
            score=fmsle
            maximize=False
        elif metric=="mape":
            score=fmape
            maximize=False
        elif metric=="mspe":
            score=fmspe
            maximize=False
        return xgb.train(parameters, xgtrain, num_rounds, watchlist,early_stopping_rounds=early_stop,feval=score,maximize=maximize)
        
    def saveModel(self, path):
        print("Save model as: ",path)
        self.model.save_model(path)
    
    def __str__(self):
        return "xgbRegression"

class xgbRegression_dart(xgbRegression):
    def __init__(self,X=None,y=None,parameters={},metric="r2",maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        self.setParameter("rate_drop",0.0,parameters)
        self.setParameter("skip_drop",0.0,parameters)
        super().__init__(X, y, parameters=parameters,metric=metric, maxEpoch=maxEpoch, checkPointPath=checkPointPath, checkPointFreq=checkPointFreq)

    def getParameterRange(self, X, y, parameters={}):
        self.setParameter("rate_drop",(float,"uni",0.0,0.5),parameters)
        self.setParameter("skip_drop",(float,"uni",0.0,0.5),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def fitModel(self, X_train, y_train, X_test, y_test, model, parameters,metric):
        parameters=parameters.copy()
        self.setParameter("booster","dart",parameters)
        return super().fitModel(X_train, y_train, X_test, y_test, model, parameters,metric)
        
    def __str__(self):
        return "xgbRegression_dart"