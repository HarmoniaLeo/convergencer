from utils.optimizing import bayesianOpt
from sklearn.model_selection import KFold
import numpy as np
from joblib import dump, load
from utils.io import saveModel
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_squared_log_error
from utils.metrics import mape,mspe

class base:
    def __init__(self,X=None,y=None,parameters={},metric="r2",modelPath=None,maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        self.parameters=parameters
        self.model=self.getModel(X,y,self.parameters,modelPath)
        if modelPath is None:
            self.model,self.trainAcc,self.testAcc=self.trainModel(X,y,self.model,self.parameters,metric)
            optimizer = bayesianOpt(pbounds=self.getParameterRange(X,y))
            next_point_to_probe = self.parameters
            model=self.model
            testAcc=self.testAcc
            for i in range(maxEpoch):
                next_point_to_probe = optimizer.next(params=next_point_to_probe,target=testAcc)
                model=self.getModel(next_point_to_probe)
                model,trainAcc,testAcc=self.trainModel(X,y,model,next_point_to_probe)
                if testAcc>self.testAcc:
                    self.model=model
                    self.parameters=next_point_to_probe
                    self.trainAcc=trainAcc
                    self.testAcc=testAcc
                print("Model: "+str(self)+" Epoch {0} ".format(i))
                print("Score on train set: {0}".format(trainAcc)+" Score on test set: {0}".format(testAcc))
                print("Best score on train set: {0}".format(self.trainAcc)+" Best score on test set: {0}".format(self.testAcc))
                print("Best parameters: ",self.parameters)
                if not(checkPointPath is None) and (i%checkPointFreq==0) and (i!=0):
                    self.saveModel(checkPointPath+"parasearch-{0}-".format(i)+str(self)+"-model")
                    saveModel(self.parameters,checkPointPath+"parasearch-{0}-".format(i)+str(self)+"-parameter")
            print("Model: "+str(self))
            print("Score on train set: {0}".format(self.trainAcc)+" Score on test set: {0}".format(self.testAcc))
        else:
            print("Model: "+str(self))
            print("Loaded from: "+modelPath)


    def setParameter(self,key,value,parameters):
        if key not in parameters:
            parameters[key]=value

    def getParameterRange(self,X,y,parameters={}):
        return parameters
    
    def getModel(self,X,y,parameters,modelPath):
        if modelPath is None:
            return None
        else:
            return load(modelPath)
    
    def fitModel(self,X_train,y_train,X_test,y_test,model,parameters,metric):
        model.fit(X_train,y_train)
        return model

    def trainModel(self,X,y,model,parameters,metric):
        kf = KFold(n_splits=5,shuffle=True)
        trainAccs=[]
        testAccs=[]
        for train_index, test_index in kf.split(X):
            X_train=X.loc[train_index]
            X_test=X.loc[test_index]
            y_train=y[train_index]
            y_test=y[test_index]
            model=self.fitModel(X_train,y_train,X_test,y_test,model,parameters,metric)
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
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
            trainAccs.append(score(y_train,y_train_pred))
            testAccs.append(score(y_test,y_test_pred))
        return model,np.mean(trainAccs),np.mean(testAccs)

    def inference(self,X):
        return self.model.predict(X)

    def saveModel(self,path):
        print("Save model as: ",path)
        dump(self.model, path)

    def __str__(self):
        return "base"