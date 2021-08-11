from utils.io import saveModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from base import base

class baggingModel:
    def __init__(self,models,weights):
        self.models=models
        self.weights=weights
    
    def fit(self,X,y):
        return self
    
    def score(self,X,y):
        predict=self.predict(X)
        return accuracy_score(y,predict)

    def predict(self,X):
        predicts=[]
        for model in self.models:
            predict=model.inference(X)
            predicts.append(predict)
        predicts=np.array(predicts).T
        predicts=self.weights*predicts
        predicts=np.sum(predicts,axis=1)
        return predicts

class bagging(base):
    def __init__(self,parameters={},models=None):
        if models is None:
            self.models=[]
            for key in parameters["models"]:

            super.__init__(parameters)
        else:
            self.models=models
            modelDict={}
            for model in models:
                modelDict[str(model)]=model.parameters
            super().__init__({"weights":1/len(models),"models":modelDict})
        self.paraLength=len(self.models)

    def getParameters(self, parameter):
        weights=parameter/np.sum(parameter)
        parameter=self.parameters
        parameter["weights"]=weights
        return parameter

    def getModel(self, parameter):
        return baggingModel(self.models,parameter["weights"])

    def __str__(self):
        return "BaggingModel"