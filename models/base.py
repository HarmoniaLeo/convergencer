from utils.optimizing import generate,evolution
from sklearn.model_selection import train_test_split
from utils.io import saveModel

class base:
    def __init__(self,parameters={}):
        self.model=None
        self.parameters=parameters
        self.paraLength=len(self.parameters.keys())

    def getModel(self,parameter):
        return
    
    def train(self,X,y,maxEpoch=1,checkPointPath=None):
        parameters=generate(self.paraLength)
        bestTrainAcc=0
        bestTestAcc=0
        for i in range(maxEpoch):
            print("Parameter searching epoch {0}:".format(i+1))
            for parameter in parameters:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                model,parameter=self.getModel(parameter)
                model=model.fit(X_train,y_train)
                trainAcc=model.score(X_train,y_train)
                testAcc=model.score(X_test,y_test)
                if testAcc>bestTestAcc:
                    self.model=model
                    self.parameters=parameter
                    bestTrainAcc=trainAcc
                    bestTestAcc=testAcc
            print("The best parameters are: ",self.parameters)
            print("Best accuracy on train set: {0}".format(bestTrainAcc))
            print("Best accuracy on test set: {0}".format(bestTestAcc))
            if not(checkPointPath is None):
                saveModel(self.parameters,checkPointPath+"para-search-{0}-".format(i)+str(self)+".json")
        return bestTrainAcc,bestTestAcc

    def inference(self,X):
        return self.model.predict(X)
    
    def read(self,parameters):
        self.parameters=parameters

    def save(self):
        return self.parameters
        
    def __str__(self):
        return "Basemodel"