from convergencer.utils.optimizing import bayesianOpt
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from joblib import dump, load
from convergencer.utils.io import saveDict,readDict
from convergencer.utils.metrics import getMetric
import os
from tqdm import tqdm

class base:
    def initialize(self,X,y,parameters={},historyLoadPath=None,verbose=1):
        self.verbose=verbose
        self.paramHistory=[]
        self.processors=self._getProcessors()
        if historyLoadPath==None:
            self.parameters=self._initParameter(X,y,parameters)
        else:
            if self.verbose>0:
                print("Read parameters from:",historyLoadPath)
            self.paramHistory = readDict(historyLoadPath)
            self.parameters=self.paramHistory[-1]["best_params"]
        if self.verbose>0:
            print("\n-------------------------Model: "+str(self)+" initialized-------------------------")
        if self.verbose==2:
            print("Initial parameters: ",self.parameters)
        return self

    def fit(self,X,y,metric="r2",modelLoadPath=None,modelSavePath=None):
        metric=getMetric(metric)
        if self.verbose>0:
            print("\n-------------------------Model: "+str(self)+" fitting-------------------------")
        if modelLoadPath is None:
            for p in self.processors:
                X=p.fit(X,y).transform(X)
            self.trainAcc,self.testAcc=self._trainModel(X,y,self.parameters,metric)
            if self.verbose>0:
                print("Initial score on train set: {0}".format(self.trainAcc)+" Initial score on test set: {0}".format(self.testAcc))
            self.model=self._fitModel(X,y,self._getModel(X,y,self.parameters,None,metric),self.parameters,metric)
            if not (modelSavePath is None):
                if not(os.path.exists(modelSavePath)):
                    os.makedirs(modelSavePath)
                if self.verbose==2:
                    print("Save model as: ",modelSavePath)
                self._saveModel(os.path.join(modelSavePath,str(self)+"-model-init"))
        else:
            self.model=self._getModel(X,y,self.parameters,modelLoadPath,metric)
            if self.verbose>0:
                print("Loaded from: "+modelLoadPath)
        return self
    
    def parasearchFit(self,X,y,metric="r2",maxEpoch=1000,modelSavePath=None,modelSaveFreq=50,historySavePath=None,historySaveFreq=50):
        self.fit(X,y,metric,None,modelSavePath)
        for p in self.processors:
            X=p.transform(X)
        metric=getMetric(metric)
        startEpoch=0
        optimizer = bayesianOpt(pbounds=self._getParameterRange(X,y,parameters={}),initParams=self.parameters)
        model=self.model
        testAcc=self.testAcc
        next_point_to_probe = optimizer.next(target=testAcc,ifMax=metric.maximum())
        for i in range(0,len(self.paramHistory)):
            next_point_to_probe = optimizer.next(target=self.paramHistory[i]["score"],ifMax=metric.maximum(),params=self.paramHistory[i]["params"])
            startEpoch+=1

        for i in (tqdm(range(startEpoch,maxEpoch)) if self.verbose==1 else range(startEpoch,maxEpoch)):
            if self.verbose==2:
                print("Model: "+str(self)+" Epoch {0} ".format(i))
                print("Parameters: ",next_point_to_probe)
            trainAcc,testAcc=self._trainModel(X,y,next_point_to_probe,metric)
            if ((testAcc>self.testAcc) if metric.maximum() else (testAcc<self.testAcc)):
                self.parameters=next_point_to_probe
                self.trainAcc=trainAcc
                self.testAcc=testAcc
            if self.verbose==2:
                print("Score on train set: {0}".format(trainAcc)+" Score on test set: {0}".format(testAcc))
            if not (historySavePath is None):
                self.paramHistory.append({"params":next_point_to_probe,"score":testAcc,"best_params":self.parameters,"metric":metric})
                if (i+1)%historySaveFreq==0:
                    if not(os.path.exists(historySavePath)):
                        os.makedirs(historySavePath)
                    if self.verbose==2:
                        print("Save parameters to: ",historySavePath)
                    saveDict(self.paramHistory,os.path.join(historySavePath,str(self)+"-"+metric+"-history-epoch{0}".format(i)))
            if (not (modelSavePath is None)) and ((i+1)%modelSaveFreq==0):
                if not(os.path.exists(modelSavePath)):
                    os.makedirs(modelSavePath)
                if self.verbose==2:
                    print("Save model as: ",modelSavePath)
                self.model=self._fitModel(X,y,self._getModel(X,y,self.parameters,None,metric),self.parameters,metric)
                self.saveModel(os.path.join(modelSavePath,str(self)+"-model-epoch{0}".format(i)))
            next_point_to_probe = optimizer.next(target=testAcc,ifMax=metric.maximum())

        print("\nModel: "+str(self))
        print("Score on train set: {0}".format(self.trainAcc)+" Score on test set: {0}".format(self.testAcc))
        print("Parameters: ",self.parameters)
        self.model=self._fitModel(X,y,self._getModel(X,y,self.parameters,None,metric),self.parameters,metric)
        if not (historySavePath is None):
            if not(os.path.exists(historySavePath)):
                os.makedirs(historySavePath)
            saveDict(self.paramHistory,os.path.join(historySavePath,str(self)+"-"+metric+"-history-final"))
        if not (modelSavePath is None):
            if not(os.path.exists(modelSavePath)):
                os.makedirs(modelSavePath)
            self._saveModel(os.path.join(modelSavePath,str(self)+"-model-final"))
        return self
        
    def predict(self,X):
        for p in self.processors:
            X=p.transform(X)
        data=X
        return self._modelPredict(self.model,data)

    def _setParameter(self,key,value,parameters):
        if key not in parameters:
            parameters[key]=value

    def _initParameter(self,X,y,parameters):
        return parameters

    def _getParameterRange(self,X,y,parameters={}):
        return parameters
    
    def _getProcessors(self):
        return []

    def _getModel(self,X,y,parameters,modelPath,metric):
        if modelPath is None:
            return None
        else:
            return load(modelPath)
    
    def _fitModel(self,X,y,model,parameters,metric):
        model.fit(X,y)
        return model

    def _modelPredict(self,model,X):
        return pd.Series(model.predict(X),X.index)

    def _trainModel(self,X,y,parameters,metric):
        kf = KFold(n_splits=5,shuffle=True)
        trainAccs=[]
        testAccs=[]
        for train_index, test_index in kf.split(X,y):
            X_train=X.iloc[train_index]
            X_test=X.iloc[test_index]
            y_train=y.iloc[train_index]
            y_test=y.iloc[test_index]
            model=self._getModel(X,y,parameters,None,metric)
            model=self._fitModel(X_train,y_train,model,parameters,metric)
            y_train_pred=self._modelPredict(model,X_train)
            y_test_pred=self._modelPredict(model,X_test)
            trainAccs.append(metric.evaluate(y_train,y_train_pred))
            testAccs.append(metric.evaluate(y_test,y_test_pred))
        return np.mean(trainAccs),np.mean(testAccs)

    def _saveModel(self,path):
        dump(self.model, path)

    def __str__(self):
        return "base"