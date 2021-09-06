from convergencer.models import base,getModels
from convergencer.utils.io import saveDict,readDict
from convergencer.utils.metrics import getMetric
import pandas as pd
import os

class stacking(base):
    def __init__(self):
        self.metaModels=[]
        self.baseModels=[]
        super().__init__()

    def initialize(self,baseModels,y,metaModels=None,metaModelParams={},metaModelHistoryLoadPaths=None,verbose=1):
        newModel=self._getClass()
        newModel.verbose=verbose

        X=self._getX(baseModels)        
        
        models=[]
        if metaModels is None:
            models=getModels()

        if type(metaModels)==list:
            for metaModel in metaModels:
                models.append(getModels()[metaModel])
            for i,model in enumerate(models):
                metaModelParam={} if str(model) not in metaModelParams.keys() else metaModelParams[str(model)]
                metaModelHistoryLoadPath=None if (metaModelHistoryLoadPaths is None) or (str(model) not in metaModelHistoryLoadPaths.keys()) else metaModelHistoryLoadPaths[str(model)]
                models[i]=models[i].initialize(X,y,metaModelParam,metaModelHistoryLoadPath,newModel.verbose)
        else:
            models=getModels()[metaModels].initialize(X,y,metaModelParams,metaModelHistoryLoadPaths,newModel.verbose)

        newModel.metaModels=models

        if newModel.verbose>0:
            print("\n-------------------------Model: "+str(newModel)+" initialized-------------------------")

        return newModel

    def fit(self,baseModels,y,metric="r2",metaModelParaSearchEpoch=0,metaModelLoadPaths=None,metaModelSavePaths=None,metaModelSaveFreq=50,metaModelHistorySavePaths=None,metaModelHistorySaveFreq=50):
        newModel=self._getClass()
        newModel.metaModels=self.metaModels
        newModel.verbose=self.verbose
        newModel.baseModels=baseModels

        X=newModel._getX(baseModels)

        metric=getMetric(metric)
        if newModel.verbose>0:
            print("\n-------------------------Model: "+str(newModel)+" fitting-------------------------")

        if type(newModel.metaModels)==list:
            for i,model in enumerate(newModel.metaModels):
                metaModelLoadPath=None if (metaModelLoadPaths is None) or (str(model) not in metaModelLoadPaths.keys()) else metaModelLoadPaths[str(model)]
                metaModelSavePath=None if (metaModelSavePaths is None) or (str(model) not in metaModelSavePaths.keys()) else metaModelSavePaths[str(model)]
                metaModelHistorySavePath=None if (metaModelHistorySavePaths is None) or (str(model) not in metaModelHistorySavePaths.keys()) else metaModelHistorySavePaths[str(model)]
                if metaModelParaSearchEpoch==0:
                    newModel.metaModels[i]=model.fit(X,y,metric,metaModelLoadPath,metaModelSavePath)
                else:
                    newModel.metaModels[i]=model.parasearchFit(X,y,metric,metaModelParaSearchEpoch,None if metaModelSavePath is None else os.path.join(metaModelSavePath,"stacking"),metaModelSaveFreq,None if metaModelHistorySavePath is None else os.path.join(metaModelHistorySavePath,"stacking"),metaModelHistorySaveFreq)
        else:
            if metaModelParaSearchEpoch==0:
                newModel.metaModels=newModel.metaModels.fit(X,y,metric,metaModelLoadPaths,metaModelSavePaths)
            else:
                newModel.metaModels=newModel.metaModels.parasearchFit(X,y,metric,metaModelParaSearchEpoch,None if metaModelSavePaths is None else os.path.join(metaModelSavePaths,"stacking"),metaModelSaveFreq,None if metaModelHistorySavePaths is None else os.path.join(metaModelHistorySavePaths,"stacking"),metaModelHistorySaveFreq)

        return newModel
        
    def predict(self,X):
        X_stack=pd.DataFrame()
        if type(self.baseModels)==dict:
            for key in self.baseModels:
                X_stack[key]=self.baseModels[key].predict(X)
        if type(self.baseModels)==list:
            for model in self.baseModels:
                X_stack[str(model)]=model.predict(X)
        if type(self.metaModels)==list:
            return {str(model):model.predict(X_stack) for model in self.metaModels}
        else:
            return self.metaModels.predict(X_stack)
    
    def _getX(self,models):
        baseModels={}
        X=pd.DataFrame()
        if type(models)==list:
            for model in models:
                baseModels[str(model)]=model
        else:
            baseModels=models
        for key in baseModels:
            X[key]=baseModels[key].trainPred
        return X

    def _getClass(self):
        return stacking()

    def __str__(self):
        return "stacking"

'''
def baggingAnalyse(X,y,modelDict,modelNum=3,metric="r2",modelSelect=False,percentage=0.3):
    print("-------------------------------------Bagging analyse start-------------------------------------")
    metric=getMetric(metric)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    predictDict=pd.DataFrame()
    scoreDict={}
    for key in modelDict.keys():
        if type(modelDict[key])==pd.Series:
            predictDict[key]=modelDict[key][y_test.index]
        else:
            modelDict[key].fit(X_train,y_train,metric)
            predictDict[key]=modelDict[key].predict(X_test)
        scoreDict[key]=metric.evaluate(y_test,predictDict[key])
        print("Score on test set: ",scoreDict[key])
    if modelSelect:
        threshold=np.percentile(scoreDict.values,percentage*100)
        for key in scoreDict:
            if scoreDict[key]<threshold:
                predictDict.drop(key,axis=1)
    predictDict=predictDict.T
    clf = KMeans(n_clusters=modelNum)
    clf.fit(predictDict)
    result={}
    labels=clf.labels_
    for i,key in enumerate(predictDict.index):
        result[key]=(labels[i],scoreDict[key])
    return result
'''