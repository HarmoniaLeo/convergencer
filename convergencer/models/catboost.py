from convergencer.processors import catToPdCat
from catboost import CatBoostRegressor,Pool
from convergencer.models import base
import multiprocessing
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from convergencer.utils.metrics import mspe
from sklearn.model_selection import KFold
import pandas as pd

class fmse(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        if weight is None:
            weight=np.ones_like(target)
        
        return mean_squared_error(target,approx),np.sum(weight)

class fmspe(object):
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        if weight is None:
            weight=np.ones_like(target)
        
        return mspe(target,approx),np.sum(weight)

class catBoostRegression(base):
    def initParameter(self, X, y, parameters):
        self.setParameter("iterations",1000,parameters)
        self.setParameter("learning_rate",0.3,parameters)
        self.setParameter("depth",8,parameters)
        self.setParameter("max_bin",256,parameters)
        self.setParameter("bagging_temperature",1.0,parameters)
        self.setParameter("random_strength",1.0,parameters)
        self.setParameter("rsm",1.0,parameters)
        self.setParameter("l2_leaf_reg",3.0,parameters)
        self.setParameter("od_pval",1e-2,parameters)
        return super().initParameter(X, y, parameters)
    
    def getParameterRange(self, X, y, parameters={}):
        self.setParameter("iterations",(int,"exp",100,20000),parameters)
        self.setParameter("learning_rate",(float,"exp",0.001,1.0),parameters)
        self.setParameter("depth",(int,"uni",1,10),parameters)
        self.setParameter("max_bin",(object,32,64,128,256,512),parameters)
        self.setParameter("bagging_temperature",(float,"uni",0.0,10.0),parameters)
        self.setParameter("random_strength",(float,"exp",0.0,1.0),parameters)
        self.setParameter("rsm",(float,"uni",0.5,1.0),parameters)
        self.setParameter("l2_leaf_reg",(float,"exp",0.0001,20.0),parameters)
        self.setParameter("od_pval",(float,"exp",1e-10,1e-2),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def getProcessors(self,X,y):
        return [catToPdCat(X,verbose=0)]
    
    def preprocess(self, X):
        ttn = X.select_dtypes(exclude=[np.number])
        cols=ttn.columns
        self.categorical_feature=cols.tolist()
        return super().preprocess(X)

    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            if metric=="r2":
                score="R2"
            elif metric=="mse":
                score=fmse
            elif metric=="mae":
                score="MAE"
            elif metric=="msle":
                score="MSLE"
            elif metric=="mape":
                score="MAPE"
            elif metric=="mspe":
                score=fmspe
            parameters=parameters.copy()
            if(torch.cuda.is_available()):self.setParameter("task_type","GPU",parameters)
            self.setParameter("thread_count",multiprocessing.cpu_count(),parameters)
            parameters["eval_metric"]=score
            parameters["allow_writing_files"]=False
            return CatBoostRegressor(**parameters)
        else:
            model = CatBoostRegressor(task_type="GPU" if torch.cuda.is_available() else "CPU",thread_count=multiprocessing.cpu_count(),allow_writing_files=False)
            model.load_model(modelPath)
            return model
    
    def fitModel(self, X_train, y_train, X_test, y_test, model, parameters,metric):
        train_pool = Pool(X_train,y_train,cat_features=self.categorical_feature)
        test_pool = Pool(X_test,y_test,cat_features=self.categorical_feature)
        model.fit(train_pool,eval_set=test_pool,logging_level="Silent",use_best_model=True)
        return model

    def modelPredict(self, model, X):
        data=Pool(X,cat_features=self.categorical_feature)
        return pd.Series(model.predict(data),X.index)
    
    def saveModel(self, path):
        self.model.save_model(path)
        
    def __str__(self):
        return "catBoostRegression"