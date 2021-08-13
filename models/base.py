from utils.optimizing import generate,evolution
from bayes_opt import BayesianOptimization,UtilityFunction
from utils.metric import cross_validation
from utils.io import saveModel

class base:
    def __init__(self,parameters={}):
        self.setParameters(parameters)
        self.model=self.getModel(self.parameters)

    def setParameters(self,parameters):
        for key in parameters:
            if key in self.parameters:
                self.parameters[key]=parameters[key]

    def getParameterRange(self):
        return {}
    
    def getModel(self):
        return base({})
    
    def train(self,X,y,paraSelect=True,maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        bestModel=None
        bestTrainAcc=0
        bestTestAcc=0
        if paraSelect:
            optimizer = BayesianOptimization(f=None,pbounds=self.getParameterRange(),verbose=2,random_state=1)
            utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
            next_point_to_probe = self.parameters
            for i in range(maxEpoch):
                model,trainAcc,testAcc=cross_validation(X,y,model)
                if testAcc>bestTestAcc:
                    bestModel=model
                    bestTrainAcc=trainAcc
                    bestTestAcc=testAcc
                print("Model: "+str(self)+" Epoch {0} ".format(i))
                print("Accuracy on train set: {0}".format(trainAcc)+" Accuracy on test set: {0}".format(testAcc))
                print("Best accuracy on train set: {0}".format(bestTrainAcc)+" Best accuracy on test set: {0}".format(bestTestAcc))
                if not(checkPointPath is None) and (i%checkPointFreq==0):
                    saveModel(bestModel.parameters,checkPointPath+"para-search-{0}-".format(i)+str(self)+".pkl")
                optimizer.register(params=next_point_to_probe,target=testAcc)
                next_point_to_probe = optimizer.suggest(utility)
                model=self.getModel(next_point_to_probe)
            return bestModel,bestTrainAcc,bestTestAcc
        else:
            model,trainAcc,testAcc=cross_validation(X,y,self.model)
            print("Model: "+str(self))
            print("Accuracy on train set: {0}".format(trainAcc)+" Accuracy on test set: {0}".format(testAcc))
            return model,trainAcc,testAcc

    def __str__(self):
        return "base"


'''
class base:
    def __init__(self,parameters={}):
        self.setParameters(parameters)
        self.model=self.getModel(self.parameters)
        self.paraLength=len(self.parameters.keys())

    def createFromGene(self,gene):
        model=base(self.parameters)
        for i,key in enumerate(model.parameters):
            model.parameters[key]=model.parameters[key]-1+gene[i]*2
        return model,model.parameters

    def getModel(self,parameters):
        return None

    def setParameters(self,parameters):
        for key in parameters:
            if key in self.parameters:
                self.parameters[key]=parameters[key]        
    
    def reduceParas(self,step,keys=[]):
        for key in keys:
            self.parameters[key]-=step*10
    
    def increaseParas(self,step,keys=[]):
        for key in keys:
            self.parameters[key]+=step*10
    
    def train(self,X,y):
        model,trainAcc,testAcc=cross_validation(X,y,self.model)
        newmodel=base(self.parameters)
        newmodel.model=model
        return newmodel,trainAcc,testAcc

    def train(self,X,y,paraSelect=True,maxEpoch=1,maxSeed=20,checkPointPath=None):
        if paraSelect:
            parameters=generate(self.paraLength,maxSeed)
            bestTrainAcc=0
            bestTestAcc=0
            for i in range(maxEpoch):
                print("Parameter searching epoch {0}:".format(i+1))
                scores=[]
                for parameter in parameters:
                    parameter=self.getParameters(parameter)
                    model=self.getModel(parameter)
                    trainAcc,testAcc=cross_validation(X,y,model)
                    scores.append(testAcc)
                    if testAcc>bestTestAcc:
                        self.model=model
                        self.setParameters(parameter)
                        bestTrainAcc=trainAcc
                        bestTestAcc=testAcc
                print("The best parameters are: ",self.parameters)
                print("Best accuracy on train set: {0}".format(bestTrainAcc))
                print("Best accuracy on test set: {0}".format(bestTestAcc))
                if not(checkPointPath is None):
                    saveModel(self.parameters,checkPointPath+"para-search-{0}-".format(i)+str(self)+".pkl")
                parameters=evolution(scores,parameters)
        else:
            trainAcc,testAcc=cross_validation(X,y,self.model)
        return bestTrainAcc,bestTestAcc

    def inference(self,X):
        return self.model.predict(X)
        
    def __str__(self):
        return "BaseModel"
'''