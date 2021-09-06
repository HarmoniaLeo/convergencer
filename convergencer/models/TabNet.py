from pytorch_tabnet.tab_model import TabNetRegressor
from convergencer.models import base
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from convergencer.processors import normalizeScaler,catToMean


class TabNetRegression(base):
    def _getClass(self):
        return TabNetRegression()

    def _initParameter(self, X, y, parameters):
        self._setParameter("n_d",8,parameters)
        self._setParameter("n_steps",3,parameters)
        self._setParameter("gamma",1.3,parameters)
        self._setParameter("n_independent",2,parameters)
        self._setParameter("n_shared",2,parameters)
        self._setParameter("learning_rate",0.1,parameters)
        self._setParameter("momentum",0.02,parameters)
        self._setParameter("lambda_sparse",1e-3,parameters)
        self._setParameter("batch_size",32,parameters)
        self._setParameter("iterations",500,parameters)
        self._setParameter("earlystop",0.02,parameters)
        return super()._initParameter(X, y, parameters)
 
    def _getParameterRange(self, X, y, parameters={}):
        self._setParameter("n_d",(int,"uni",8,64),parameters)
        self._setParameter("n_steps",(int,"uni",3,10),parameters)
        self._setParameter("gamma",(float,"uni",1.0,2.0),parameters)
        self._setParameter("n_independent",(int,"uni",1,5),parameters)
        self._setParameter("n_shared",(int,"uni",1,5),parameters)
        self._setParameter("learning_rate",(float,"exp",0.001,0.1),parameters)
        self._setParameter("momentum",(float,"exp",0.01,0.4),parameters)
        self._setParameter("lambda_sparse",(float,"exp",0.0,1.0),parameters)
        self._setParameter("batch_size",(object,128,256,512,1024,2048),parameters)
        self._setParameter("iterations",(int,"uni",100,5000),parameters)
        self._setParameter("earlystop",(float,"uni",0.02,0.04),parameters)
        return super()._getParameterRange(X, y, parameters=parameters)

    def _getModel(self, X, y, parameters, modelPath,metric):
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

    def _fitModel(self, X, y, model, parameters,metric):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
        model.fit(
            X_train.values,np.array([y_train.values]).T,
            eval_set=[(X_test.values, np.array([y_test.values]).T)],
            patience=int(parameters["earlystop"]*parameters["iterations"]),
            batch_size=parameters["batch_size"],
            max_epochs=parameters["iterations"],
            num_workers=multiprocessing.cpu_count(),
            eval_metric=[metric.tabnet()]
            )
        return model
    
    def _modelPredict(self, model, X):
        return pd.Series(model.predict(X.values).T[0],index=X.index)

    def _trainModel(self,X,y,parameters,metric):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model=self._getModel(X,y,parameters,None,metric)
        model=self._fitModel(X_train,y_train,model,parameters,metric)
        y_train_pred=self._modelPredict(model,X_train)
        y_test_pred=self._modelPredict(model,X_test)
        return metric.evaluate(y_train,y_train_pred),metric.evaluate(y_test,y_test_pred),None

    def _getProcessors(self):
        return [catToMean().initialize({},verbose=0),normalizeScaler().initialize({},verbose=0)]

    def _saveModel(self, path):
        self.model.save_model(path)
        
    def __str__(self):
        return "TabNetRegression"