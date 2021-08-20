import lightgbm as lgb
from base import base
import numpy as np
import math
import torch

class lgbmRegression(base):
    def __init__(self,X,y,parameters={},maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        mst=np.sum(np.power(y-np.mean(y),2))/len(y)
        self.setParameter("num",1000,parameters)
        self.setParameter("learning_rate",0.3,parameters)
        self.setParameter("boosting","gbdt",parameters)
        self.setParameter("min_gain_to_split",0.001*mst,parameters)
        self.setParameter("min_data_in_leaf",math.ceil(X.shape[0]*0.0001),parameters)
        self.setParameter("num_leaves",math.ceil(X.shape[0]*0.0001),parameters)
        self.setParameter("max_depth",8,parameters)
        self.setParameter("max_bin",256,parameters)
        self.setParameter("bagging_freq",10,parameters)
        self.setParameter("bagging_fraction",1.0,parameters)
        self.setParameter("feature_fraction",1.0,parameters)
        self.setParameter("lambda_l1",1.0,parameters)
        self.setParameter("lambda_l2",1.0,parameters)
        self.setParameter("early_stopping_round",10,parameters)
        super().__init__(X, y, parameters=parameters, maxEpoch=maxEpoch, checkPointPath=checkPointPath, checkPointFreq=checkPointFreq)

    def getParameterRange(self, X, y, parameters={}):
        mst=np.sum(np.power(y-np.mean(y),2))/len(y)
        self.setParameter("num",(object,100,200,500,800,1000,2000,5000,10000),parameters)
        self.setParameter("learning_rate",(float,"exp",0.0,1.0),parameters)
        self.setParameter("boosting",(object,"gbdt","goss"),parameters)
        self.setParameter("min_gain_to_split",(float,"exp",0.0,0.01*mst),parameters)
        self.setParameter("min_data_in_leaf",(int,"exp",1,math.ceil(X.shape[0]*0.0001)),parameters)
        self.setParameter("num_leaves",(int,"uni",10,100),parameters)
        self.setParameter("max_depth",(int,"uni",1,10),parameters)
        self.setParameter("max_bin",(object,32,64,128,256,512),parameters)
        self.setParameter("bagging_freq",(object,10,20,50),parameters)
        self.setParameter("bagging_fraction",(float,"uni",0.5,1.0),parameters)
        self.setParameter("feature_fraction",(float,"uni",0.5,1.0),parameters)
        self.setParameter("lambda_l1",(float,"exp",0.0,1.0),parameters)
        self.setParameter("lambda_l2",(float,"exp",0.0,1.0),parameters)
        self.setParameter("early_stopping_round",(object,5,10,15),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def getModel(self, X, y, parameters, modelPath):
        if modelPath is None:
            return None
        return lgb.Booster(model_file=modelPath)

    def fitModel(self, X_train, y_train, X_test, y_test, model, parameters):
        parameters=parameters.copy()
        rounds=parameters.pop("num")
        if(torch.cuda.is_available()):self.setParameter("device","gpu",parameters)
        self.setParameter("boosting","gbdt",parameters)
        train_data=lgb.Dataset(X_train,y_train)
        test_data=lgb.Dataset(X_test,y_test)
        return lgb.train(parameters, train_data,rounds, valid_sets=[test_data])
    
    def saveModel(self, path):
        print("Save model as: ",path)
        self.model.save_model(path)
        
    def __str__(self):
        return "lgbmRegression"

class lgbmRegression_dart(lgbmRegression):
    def __init__(self,X,y,parameters={},maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        self.setParameter("drop_rate",0.1,parameters)
        super().__init__(X, y, parameters=parameters, maxEpoch=maxEpoch, checkPointPath=checkPointPath, checkPointFreq=checkPointFreq)

    def getParameterRange(self, X, y, parameters={}):
        self.setParameter("drop_rate",(float,"uni",0.0,0.5),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def fitModel(self, X_train, y_train, X_test, y_test, model, parameters):
        parameters=parameters.copy()
        self.parameters("boosting","dart",parameters)
        return super().fitModel(X_train, y_train, X_test, y_test, model, parameters)
        
    def __str__(self):
        return "lgbmRegression_dart"

class lgbmRegression_goss(lgbmRegression):
    def fitModel(self, X_train, y_train, X_test, y_test, model, parameters):
        parameters=parameters.copy()
        self.parameters("boosting","goss",parameters)
        return super().fitModel(X_train, y_train, X_test, y_test, model, parameters)
        
    def __str__(self):
        return "lgbmRegression_goss"