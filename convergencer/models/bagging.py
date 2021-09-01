import numpy as np
from models.base import base

class baggingModel:
    def __init__(self,models,weights):
        self.models=models
        self.weights=weights

    def fit(self,X,y):
        return None

    def predict(self,X):
        predicts=[]
        for model in self.models:
            predict=model.inference(X)
            predicts.append(predict)
        predicts=np.array(predicts).T
        predicts=self.weights/np.sum(self.weights)*predicts
        predicts=np.sum(predicts,axis=1)
        return predicts


class baggingRegression(base):
    def __init__(self,X,y,models=[],parameters={},metric="r2",maxEpoch=1000,modelLoadPath=None,modelSavePath=None,modelSaveFreq=50,historyLoadPath=None,historySavePath=None,historySaveFreq=50,verbose=1):
        self.baggingModels=models
        super().__init__(X=X, y=y, parameters=parameters, metric=metric, maxEpoch=maxEpoch,modelLoadPath=modelLoadPath,modelSavePath=modelSavePath,modelSaveFreq=modelSaveFreq,historyLoadPath=historyLoadPath,historySavePath=historySavePath,historySaveFreq=historySaveFreq,verbose=verbose)

    def initParameter(self, X, y, parameters):
        for model in self.baggingModels:
            self.setParameter(str(model),1,parameters)
        return super().initParameter(X, y, parameters)

    def getParameterRange(self,X,y,parameters={}):
        for model in self.baggingModels:
            self.setParameter(str(model),(float,"uni",1e-6,1.0),parameters)
        return super().getParameterRange(X, y, parameters=parameters)
    
    def getModel(self,X,y,parameters,modelPath,metric):
        weights=[]
        for model in self.baggingModels:
            weights.append(parameters[str(model)])
        return baggingModel(self.baggingModels,weights)

    def saveModel(self,path):
        return None

    def __str__(self):
        return "bagging"