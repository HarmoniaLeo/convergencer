from utils.optimizing import generate,evolution
from sklearn.model_selection import train_test_split
from utils.io import saveModel

class base:
    def __init__(self,parameters={}):
        self.parameters=parameters
        self.model=self.getModel(parameters)
        self.paraLength=len(self.parameters.keys())

    def getModel(self,parameters):
        return self.model

    def getParameters(self,parameter):
        return {}
    
    def getParameter(self,key,default,parameters):
        return default if key not in parameters.keys() else parameters[key]
    
    def train(self,X,y,paraSelect=True,maxEpoch=1,maxSeed=20,checkPointPath=None):
        if paraSelect:
            parameters=generate(self.paraLength,maxSeed)
            bestTrainAcc=0
            bestTestAcc=0
            for i in range(maxEpoch):
                print("Parameter searching epoch {0}:".format(i+1))
                scores=[]
                for parameter in parameters:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    parameter=self.getParameters(parameter)
                    model=self.getModel(parameter)
                    model=model.fit(X_train,y_train)
                    trainAcc=model.score(X_train,y_train)
                    testAcc=model.score(X_test,y_test)
                    scores.append(testAcc)
                    if testAcc>bestTestAcc:
                        self.model=model
                        self.parameters=parameter
                        bestTrainAcc=trainAcc
                        bestTestAcc=testAcc
                print("The best parameters are: ",self.parameters)
                print("Best accuracy on train set: {0}".format(bestTrainAcc))
                print("Best accuracy on test set: {0}".format(bestTestAcc))
                if not(checkPointPath is None):
                    saveModel(self.parameters,checkPointPath+"para-search-{0}-".format(i)+str(self)+".pkl")
                parameters=evolution(scores,parameters)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            self.model=self.model.fit(X_train,y_train)
            bestTrainAcc=self.model.score(X_train,y_train)
            bestTestAcc=self.model.score(X_test,y_test)
        return bestTrainAcc,bestTestAcc

    def inference(self,X):
        return self.model.predict(X)
        
    def __str__(self):
        return "BaseModel"