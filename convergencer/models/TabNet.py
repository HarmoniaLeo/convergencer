from pytorch_tabnet.tab_model import TabNetRegressor
from convergencer.models import base
import multiprocessing
import numpy as np
import pandas as pd
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_squared_log_error
from convergencer.utils.metrics import mape,mspe
from sklearn.model_selection import train_test_split
from convergencer.processors import normalizeScaler,catToMean

class fr2(Metric):
    def __init__(self):
        self._name = "r2"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return r2_score(y_true[:, 0], y_score[:, 0])

class fmsle(Metric):
    def __init__(self):
        self._name = "msle"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return mean_squared_log_error(y_true[:, 0], y_score[:, 0])

class fmape(Metric):
    def __init__(self):
        self._name = "mape"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return mape(y_true[:, 0], y_score[:, 0])

class fmspe(Metric):
    def __init__(self):
        self._name = "mspe"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return mspe(y_true[:, 0], y_score[:, 0])

class TabNetRegression(base):
    def initParameter(self, X, y, parameters):
        self.setParameter("n_d",8,parameters)
        self.setParameter("n_steps",3,parameters)
        self.setParameter("gamma",1.3,parameters)
        self.setParameter("n_independent",2,parameters)
        self.setParameter("n_shared",2,parameters)
        self.setParameter("learning_rate",0.1,parameters)
        self.setParameter("momentum",0.02,parameters)
        self.setParameter("lambda_sparse",1e-3,parameters)
        self.setParameter("batch_size",32,parameters)
        self.setParameter("iterations",500,parameters)
        self.setParameter("earlystop",0.02,parameters)
        return super().initParameter(X, y, parameters)
 
    def getParameterRange(self, X, y, parameters={}):
        self.setParameter("n_d",(int,"uni",8,64),parameters)
        self.setParameter("n_steps",(int,"uni",3,10),parameters)
        self.setParameter("gamma",(float,"uni",1.0,2.0),parameters)
        self.setParameter("n_independent",(int,"uni",1,5),parameters)
        self.setParameter("n_shared",(int,"uni",1,5),parameters)
        self.setParameter("learning_rate",(float,"exp",0.001,0.1),parameters)
        self.setParameter("momentum",(float,"exp",0.01,0.4),parameters)
        self.setParameter("lambda_sparse",(float,"exp",0.0,1.0),parameters)
        self.setParameter("batch_size",(object,128,256,512,1024,2048),parameters)
        self.setParameter("iterations",(int,"uni",100,5000),parameters)
        self.setParameter("earlystop",(float,"uni",0.02,0.04),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            return TabNetRegressor(
                n_d=parameters["n_d"],
                n_a=parameters["n_d"],
                n_steps=parameters["n_steps"],
                gamma=parameters["gamma"],
                n_independent=parameters["n_independent"],
                n_shared=parameters["n_shared"],
                momentum=parameters["momentum"],
                lambda_sparse=parameters["lambda_sparse"],
                optimizer_params=dict(lr=parameters["learning_rate"]),
                scheduler_params=dict(mode='max' if metric=="r2" else "min",min_lr=1e-5)
            )
        else:
            model=TabNetRegressor()
            return model.load_model(modelPath)

    def fitModel(self, X_train, y_train, X_test, y_test, model, parameters,metric):
        if metric=="r2":
            score=fr2
        elif metric=="mse":
            score="mse"
        elif metric=="mae":
            score="mae"
        elif metric=="msle":
            score=fmsle
        elif metric=="mape":
            score=fmape
        elif metric=="mspe":
            score=fmspe
        model.fit(
            X_train.values,np.array([y_train.values]).T,
            eval_set=[(X_test.values, np.array([y_test.values]).T)],
            patience=int(parameters["earlystop"]*parameters["iterations"]),
            batch_size=parameters["batch_size"],
            max_epochs=parameters["iterations"],
            num_workers=multiprocessing.cpu_count(),
            eval_metric=[score]
            )
        return model
    
    def modelPredict(self, model, X):
        return pd.Series(model.predict(X.values).T[0],index=X.index)

    def trainModel(self,X,y,parameters,metric):
        if metric=="r2":
            score=r2_score
        elif metric=="mse":
            score=mean_squared_error
        elif metric=="mae":
            score=mean_absolute_error
        elif metric=="msle":
            score=mean_squared_log_error
        elif metric=="mape":
            score=mape
        elif metric=="mspe":
            score=mspe
        else:
            raise Exception("Unsupported metric. ")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model=self.getModel(X,y,parameters,None,metric)
        model=self.fitModel(X_train,y_train,X_test,y_test,model,parameters,metric)
        y_train_pred=self.modelPredict(model,X_train)
        y_test_pred=self.modelPredict(model,X_test)
        return model,score(y_train,y_train_pred),score(y_test,y_test_pred)

    def getProcessors(self,X,y):
        return [catToMean(X,y,verbose=0),normalizeScaler(X,verbose=0)]

    def saveModel(self, path):
        self.model.save_model(path)
        
    def __str__(self):
        return "ANNRegression"