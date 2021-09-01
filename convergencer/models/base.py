from convergencer.utils.optimizing import bayesianOpt
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from joblib import dump, load
from convergencer.utils.io import saveDict,readDict
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_squared_log_error
from convergencer.utils.metrics import mape,mspe
import os
from tqdm import tqdm

class base:
    def __init__(self,X,y,parameters={},metric="r2",maxEpoch=1000,modelLoadPath=None,modelSavePath=None,modelSaveFreq=50,historyLoadPath=None,historySavePath=None,historySaveFreq=50,verbose=1):
        self.processors=self.getProcessors(X,y)
        if modelLoadPath is None:
            X=self.preprocess(X)
            if metric=="r2":
                maximum=True
            else:
                maximum=False
            history=[]
            startEpoch=0
            if not (historyLoadPath is None):
                paramHistory = readDict(historyLoadPath)
                optimizer = bayesianOpt(pbounds=self.getParameterRange(X,y),initParams=paramHistory[0]["params"])
                self.testAcc = -np.inf if maximum else np.inf
                for i in range(0,len(paramHistory)):
                    next_point_to_probe = optimizer.next(target=paramHistory[i]["score"],ifMax=maximum,params=paramHistory[i]["params"])
                    if ((paramHistory[i]["score"]>self.testAcc) if maximum else (paramHistory[i]["score"]<self.testAcc)):
                        self.parameters=paramHistory[i]["params"]
                    startEpoch+=1
            else:
                self.parameters=self.initParameter(X,y,parameters)
                optimizer = bayesianOpt(pbounds=self.getParameterRange(X,y),initParams=self.parameters)
            print("\n-------------------------Model: "+str(self)+" initialized-------------------------")
            if verbose==2:
                print("Initial parameters: ",self.parameters)
            self.model,self.trainAcc,self.testAcc=self.trainModel(X,y,self.parameters,metric)
            model=self.model
            testAcc=self.testAcc
            if verbose==2:
                print("Initial score on train set: {0}".format(self.trainAcc)+" Initial score on test set: {0}".format(self.testAcc))
            for i in (tqdm(range(startEpoch,maxEpoch)) if verbose==1 else range(startEpoch,maxEpoch)):
                next_point_to_probe = optimizer.next(target=testAcc,ifMax=maximum)
                if verbose==2:
                    print("Model: "+str(self)+" Epoch {0} ".format(i))
                    print("Parameters: ",next_point_to_probe)
                model,trainAcc,testAcc=self.trainModel(X,y,next_point_to_probe,metric)
                if ((testAcc>self.testAcc) if maximum else (testAcc<self.testAcc)):
                    self.model=model
                    self.parameters=next_point_to_probe
                    self.trainAcc=trainAcc
                    self.testAcc=testAcc
                if verbose==2:
                    print("Score on train set: {0}".format(trainAcc)+" Score on test set: {0}".format(testAcc))
                if not (historySavePath is None):
                    history.append({"params":next_point_to_probe,"score":testAcc})
                    if (i+1)%historySaveFreq==0:
                        if not(os.path.exists(historySavePath)):
                            os.makedirs(historySavePath)
                        saveDict(history,os.path.join(historySavePath,str(self)+"-"+metric+"-history-epoch{0}".format(i)))
                if (not (modelSavePath is None)) and ((i+1)%modelSaveFreq==0):
                    if not(os.path.exists(modelSavePath)):
                        os.makedirs(modelSavePath)
                    self.saveModel(os.path.join(modelSavePath,str(self)+"-model-epoch{0}".format(i)))
            print("\nModel: "+str(self))
            print("Score on train set: {0}".format(self.trainAcc)+" Score on test set: {0}".format(self.testAcc))
            print("Parameters: ",self.parameters)
            if not (historySavePath is None):
                if not(os.path.exists(historySavePath)):
                    os.makedirs(historySavePath)
                saveDict(history,os.path.join(historySavePath,str(self)+"-"+metric+"-history-final"))
            if not (modelSavePath is None):
                if not(os.path.exists(modelSavePath)):
                    os.makedirs(modelSavePath)
                self.saveModel(os.path.join(modelSavePath,str(self)+"-model-final"))
        else:
            self.model=self.getModel(X,y,parameters,modelLoadPath,metric)
            print("Model: "+str(self))
            print("Loaded from: "+modelLoadPath)

    def setParameter(self,key,value,parameters):
        if key not in parameters:
            parameters[key]=value

    def initParameter(self,X,y,parameters):
        return parameters

    def getParameterRange(self,X,y,parameters={}):
        return parameters
    
    def getProcessors(self,X,y):
        return []
    
    def preprocess(self,X):
        for p in self.processors:
            X=p.transform(X)
        return X

    def getModel(self,X,y,parameters,modelPath,metric):
        if modelPath is None:
            return None
        else:
            return load(modelPath)
    
    def fitModel(self,X_train,y_train,X_test,y_test,model,parameters,metric):
        model.fit(X_train,y_train)
        return model

    def modelPredict(self,model,X):
        return pd.Series(model.predict(X),X.index)

    def trainModel(self,X,y,parameters,metric):
        kf = KFold(n_splits=10,shuffle=True)
        trainAccs=[]
        testAccs=[]
        maximum=False
        bestScore=np.inf
        if metric=="r2":
            score=r2_score
            maximum=True
            bestScore=-np.inf
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
        bestModel=None
        for train_index, test_index in kf.split(X,y):
            X_train=X.iloc[train_index]
            X_test=X.iloc[test_index]
            y_train=y.iloc[train_index]
            y_test=y.iloc[test_index]
            model=self.getModel(X,y,parameters,None,metric)
            model=self.fitModel(X_train,y_train,X_test,y_test,model,parameters,metric)
            y_train_pred=self.modelPredict(model,X_train)
            y_test_pred=self.modelPredict(model,X_test)
            
            trainAccs.append(score(y_train,y_train_pred))
            testAccs.append(score(y_test,y_test_pred))
            if maximum:
                if score(y_test,y_test_pred)>bestScore:
                    bestScore=score(y_test,y_test_pred)
                    bestModel=model
            else:
                if score(y_test,y_test_pred)<bestScore:
                    bestScore=score(y_test,y_test_pred)
                    bestModel=model
        return bestModel,np.mean(trainAccs),np.mean(testAccs)

    def inference(self,X):
        data=self.preprocess(X)
        return self.modelPredict(self.model,data)

    def saveModel(self,path):
        print("Save model as: ",path)
        dump(self.model, path)

    def __str__(self):
        return "base"