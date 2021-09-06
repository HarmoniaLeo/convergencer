from convergencer.utils.optimizing import bayesianOpt
import numpy as np
import pandas as pd
from joblib import dump, load
from convergencer.utils.io import saveDict,readDict
from convergencer.utils.metrics import getMetric
import os
from tqdm import tqdm

class base:
    def __init__(self):
        self.verbose=1
        self.paramHistory=[]
        self.processors=self._getProcessors()
        self.model=None
        self.parameters=None
        self.trainAcc=None
        self.testAcc=None
        self.trainPred=None

    def initialize(self,X,y,parameters={},historyLoadPath=None,verbose=1):
        newModel=self._getClass()
        newModel.verbose=verbose
        if historyLoadPath==None:
            newModel.parameters=newModel._initParameter(X,y,parameters)
        else:
            if newModel.verbose>0:
                print("Read parameters from:",historyLoadPath)
            newModel.paramHistory = readDict(historyLoadPath)
            newModel.parameters=newModel.paramHistory[-1]["best_params"]
        if newModel.verbose>0:
            print("\n-------------------------Model: "+str(newModel)+" initialized-------------------------")
        if newModel.verbose==2:
            print("Initial parameters: ",newModel.parameters)
        return newModel

    def fit(self,X,y,metric="r2",modelLoadPath=None,modelSavePath=None):
        newModel=self._getClass()
        newModel.parameters=self.parameters
        newModel.verbose=self.verbose
        newModel.paramHistory=self.paramHistory
        metric=getMetric(metric)
        if newModel.verbose>0:
            print("\n-------------------------Model: "+str(newModel)+" fitting-------------------------")
        if modelLoadPath is None:
            for p in newModel.processors:
                X=p.fit(X,y).transform(X)
            newModel.trainAcc,newModel.testAcc,newModel.trainPred=newModel._trainModel(X,y,newModel.parameters,metric)
            if newModel.verbose>0:
                print("Initial score on train set: {0}".format(newModel.trainAcc)+" Initial score on test set: {0}".format(newModel.testAcc))
            newModel.model=newModel._fitModel(X,y,newModel._getModel(X,y,newModel.parameters,None,metric),newModel.parameters,metric)
            if not (modelSavePath is None):
                if not(os.path.exists(modelSavePath)):
                    os.makedirs(modelSavePath)
                if newModel.verbose==2:
                    print("Save model as: ",modelSavePath)
                newModel._saveModel(os.path.join(modelSavePath,str(newModel)+"-model-init"))
        else:
            newModel.model=newModel._getModel(X,y,newModel.parameters,modelLoadPath,metric)
            if newModel.verbose>0:
                print("Loaded from: "+modelLoadPath)
        return newModel
    
    def parasearchFit(self,X,y,metric="r2",maxEpoch=1000,modelSavePath=None,modelSaveFreq=50,historySavePath=None,historySaveFreq=50):
        newModel=self.fit(X,y,metric,None,modelSavePath)
        for p in newModel.processors:
            X=p.transform(X)
        metric=getMetric(metric)
        startEpoch=0
        optimizer = bayesianOpt(pbounds=newModel._getParameterRange(X,y,parameters={}),initParams=newModel.parameters)
        testAcc=newModel.testAcc
        next_point_to_probe = optimizer.next(target=testAcc,ifMax=metric.maximum())
        for i in range(0,len(newModel.paramHistory)):
            next_point_to_probe = optimizer.next(target=newModel.paramHistory[i]["score"],ifMax=metric.maximum(),params=newModel.paramHistory[i]["params"])
            startEpoch+=1

        for i in (tqdm(range(startEpoch,maxEpoch)) if newModel.verbose==1 else range(startEpoch,maxEpoch)):
            if newModel.verbose==2:
                print("Model: "+str(newModel)+" Epoch {0} ".format(i))
                print("Parameters: ",next_point_to_probe)
            trainAcc,testAcc,trainPred=newModel._trainModel(X,y,next_point_to_probe,metric)
            if ((testAcc>newModel.testAcc) if metric.maximum() else (testAcc<newModel.testAcc)):
                newModel.parameters=next_point_to_probe
                newModel.trainAcc=trainAcc
                newModel.testAcc=testAcc
                newModel.trainPred=trainPred
            if newModel.verbose==2:
                print("Score on train set: {0}".format(trainAcc)+" Score on test set: {0}".format(testAcc))
            if not (historySavePath is None):
                newModel.paramHistory.append({"params":next_point_to_probe,"score":testAcc,"best_params":newModel.parameters,"metric":metric})
                if (i+1)%historySaveFreq==0:
                    if not(os.path.exists(historySavePath)):
                        os.makedirs(historySavePath)
                    if newModel.verbose==2:
                        print("Save parameters to: ",historySavePath)
                    saveDict(newModel.paramHistory,os.path.join(historySavePath,str(newModel)+"-"+metric+"-history-epoch{0}".format(i)))
            if (not (modelSavePath is None)) and ((i+1)%modelSaveFreq==0):
                if not(os.path.exists(modelSavePath)):
                    os.makedirs(modelSavePath)
                if newModel.verbose==2:
                    print("Save model as: ",modelSavePath)
                newModel.model=newModel._fitModel(X,y,newModel._getModel(X,y,newModel.parameters,None,metric),newModel.parameters,metric)
                newModel.saveModel(os.path.join(modelSavePath,str(newModel)+"-model-epoch{0}".format(i)))
            next_point_to_probe = optimizer.next(target=testAcc,ifMax=metric.maximum())

        print("\nModel: "+str(newModel))
        print("Score on train set: {0}".format(newModel.trainAcc)+" Score on test set: {0}".format(newModel.testAcc))
        print("Parameters: ",newModel.parameters)
        newModel.model=newModel._fitModel(X,y,newModel._getModel(X,y,newModel.parameters,None,metric),newModel.parameters,metric)
        if not (historySavePath is None):
            if not(os.path.exists(historySavePath)):
                os.makedirs(historySavePath)
            saveDict(newModel.paramHistory,os.path.join(historySavePath,str(newModel)+"-"+metric+"-history-final"))
        if not (modelSavePath is None):
            if not(os.path.exists(modelSavePath)):
                os.makedirs(modelSavePath)
            newModel._saveModel(os.path.join(modelSavePath,str(newModel)+"-model-final"))
        return newModel
        
    def predict(self,X):
        for p in self.processors:
            X=p.transform(X)
        data=X
        return self._modelPredict(self.model,data)
    
    def _getClass(self):
        return base()

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
        trainAccs=[]
        testAccs=[]
        trainPred=pd.Series(index=y.index)
        indexs=X.index
        folds=5
        step=int(indexs.shape[0]/folds)
        kfold=[]
        for i in range(folds-1):
            kfold.append((np.concatenate([indexs[0:i*step],indexs[i*step+step:]]),indexs[i*step:i*step+step]))
        kfold.append((indexs[0:(folds-1)*step],indexs[(folds-1)*step:]))
        for train_index, test_index in kfold:
            X_train=X.loc[train_index]
            X_test=X.loc[test_index]
            y_train=y[train_index]
            y_test=y[test_index]
            model=self._getModel(X,y,parameters,None,metric)
            model=self._fitModel(X_train,y_train,model,parameters,metric)
            y_train_pred=self._modelPredict(model,X_train)
            y_test_pred=self._modelPredict(model,X_test)
            trainPred[y_test_pred.index]=y_test_pred
            trainAccs.append(metric.evaluate(y_train,y_train_pred))
            testAccs.append(metric.evaluate(y_test,y_test_pred))
        return np.mean(trainAccs),np.mean(testAccs),trainPred

    def _saveModel(self,path):
        dump(self.model, path)

    def __str__(self):
        return "base"