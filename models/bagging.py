from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from base import base

class bagging(base):
    def __init__(self,models):
        self.models=models
        self.parameter=[]
        self.paraLength=len(self.models)

    def parameterToWeights(self,para):
    
    def test(self,X,y,parameter):
        weight=
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        trainPredicts=[]
        testPredicts=[]
        for model in self.models:
            print("Model: ",model)
            #X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_train, y_train, test_size=0.4)
            #trainAcc,testAcc=model.train(X_test_1,y_test_1)
            predict=model.inference(X_train)
            print("Accuracy on bagging train set: {0}".format(accuracy_score(y_train,predict)))
            trainPredicts.append(predict)
            predict=model.inference(X_test)
            print("Accuracy on bagging test set: {0}".format(accuracy_score(y_test,predict)))
            testPredicts.append(predict)
        predicts=np.array(trainPredicts).T
        predicts=self.weights*predicts
        predicts=np.sum(predicts,axis=1)
        trainAcc=accuracy_score(y_train,predicts)
        print("Accuracy of assemble model on bagging train set: {0}".format(trainAcc))
        predicts=np.array(testPredicts).T
        predicts=self.weights*predicts
        predicts=np.sum(predicts,axis=1)
        testAcc=accuracy_score(y_test,predicts)
        print("Accuracy of assemble model on bagging train set: {0}".format(testAcc))
        return trainAcc,testAcc
    
    def inference(self,X):
        predicts=[]
        for model in self.models:
            predict=model.inference(X)
            predicts.append(predict)
        predicts=np.array(predicts).T
        predicts=self.weights*predicts
        predicts=np.sum(predicts,axis=1)
        return predicts

    def __str__(self):
        return "Bagging model"

    def save():

    def load():