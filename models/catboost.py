from catboost import CatBoostRegressor
from base import base
import multiprocessing
import torch

class catBoostRegression(base):
    def __init__(self,X,y,parameters={},maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        self.setParameter("iterations",5000,parameters)
        self.setParameter("learning_rate",0.3,parameters)
        self.setParameter("depth",8,parameters)
        self.setParameter("max_bin",256,parameters)
        self.setParameter("bagging_temperature",1.0,parameters)
        self.setParameter("random_strength",1.0,parameters)
        self.setParameter("rsm",1.0,parameters)
        self.setParameter("l2_leaf_reg",3.0,parameters)
        self.setParameter("early_stopping_round",200,parameters)
        self.setParameter("od_type","IncToDec",parameters)
        self.setParameter("od_pval",1e-10,parameters)
        self.setParameter("od_wait",20,parameters)
        super().__init__(X, y, parameters=parameters, maxEpoch=maxEpoch, checkPointPath=checkPointPath, checkPointFreq=checkPointFreq)

    def getParameterRange(self, X, y, parameters={}):
        self.setParameter("iterations",(object,500,1000,2000,5000,10000,20000),parameters)
        self.setParameter("learning_rate",(float,"exp",0.0,1.0),parameters)
        self.setParameter("depth",(int,"uni",1,10),parameters)
        self.setParameter("max_bin",(object,32,64,128,256,512),parameters)
        self.setParameter("bagging_temperature",(float,"uni",0.0,10.0),parameters)
        self.setParameter("random_strength",(float,"exp",0.0,1.0),parameters)
        self.setParameter("rsm",(float,"uni",0.5,1.0),parameters)
        self.setParameter("l2_leaf_reg",(float,"exp",0.0,20.0),parameters)
        self.setParameter("early_stopping_round",(object,100,200,400),parameters)
        self.setParameter("od_type",(object,"IncToDec","Iter"),parameters)
        self.setParameter("od_pval",(float,"exp",1e-10,1e-2),parameters)
        self.setParameter("od_wait",(object,20,100,500),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def getModel(self, X, y, parameters, modelPath):
        if modelPath is None:
            parameters=parameters.copy()
            if(torch.cuda.is_available()):parameters["task_type"]="GPU"
            parameters["thread_count"]=multiprocessing.cpu_count()
            return CatBoostRegressor(**parameters)
        else:
            model = CatBoostRegressor()
            model.load_model(modelPath)
            return model
    
    def fitModel(self, X_train, y_train, X_test, y_test, model, parameters):
        eval_set = [(X_test, y_test)]
        model.fit(X_train,y_train,eval_set=eval_set)
        return model
    
    def saveModel(self, path):
        self.model.save_model(path)
        
    def __str__(self):
        return "catBoostRegression"