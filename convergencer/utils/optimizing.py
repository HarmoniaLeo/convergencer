import numpy as np
import random
from bayes_opt import BayesianOptimization,UtilityFunction

class bayesianOpt:
    def __init__(self,pbounds,initParams):
        tbounds={}
        self.floatToObject={}
        self.takeInt=[]
        self.takeExp=[]
        for key in pbounds.keys():
            if pbounds[key][0]==object:
                self.floatToObject[key]=[]
                for i in range(1,len(pbounds[key])):
                    self.floatToObject[key].append(pbounds[key][i])
                tbounds[key]=(0,len(self.floatToObject[key])-1e-5)
            else:
                if pbounds[key][0]==int:
                    self.takeInt.append(key)
                if pbounds[key][1]=="exp":
                    self.takeExp.append(key)
                    tbounds[key]=(np.log(pbounds[key][2]+1e-16),np.log(pbounds[key][3]))
                else:
                    tbounds[key]=(float(pbounds[key][2]),float(pbounds[key][3]))
        self.optimizer=BayesianOptimization(f=None,pbounds=tbounds,verbose=2,random_state=1)
        self.utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        self.params={}
        self.outer2inner(initParams)

    def inner2outer(self):
        params={}
        for key in self.params.keys():
            if key in self.floatToObject.keys():
                params[key]=self.floatToObject[key][int(self.params[key])]
            else:
                if (key in self.takeExp) and (key in self.takeInt):
                    params[key]=int(np.exp(self.params[key]))
                elif (key in self.takeExp) and (key not in self.takeInt):
                    params[key]=round(np.exp(self.params[key]),16)
                elif (key not in self.takeExp) and (key in self.takeInt):
                    params[key]=int(self.params[key])
                else:
                    params[key]=round(self.params[key],16)
        return params

    def outer2inner(self,params):
        for key in params.keys():
            if key in self.floatToObject.keys():
                for i in range(0,len(self.floatToObject[key])):
                    if self.floatToObject[key][i]==params[key]:
                        self.params[key]=float(i)
            elif key in self.takeExp:
                self.params[key]=np.log(params[key])
            else:
                self.params[key]=float(params[key])

    def next(self,target,ifMax,params=None):
        if not params is None:
            self.params=params
        if not ifMax:
            target=-target
        try:
            self.optimizer.register(params=self.params,target=target)
        except Exception:
            self.params=self.optimizer.suggest(self.utility)
            return self.inner2outer()
        self.params=self.optimizer.suggest(self.utility)
        return self.inner2outer()


def finetune(X,y,models,maxEpoch,accThr,maxModelNum,bestModels=[],bestTrainAccs=[],bestTestAccs=[],checkPointPath=None):
    for i in range(maxEpoch):
        print("Fine-tuning epoch {0}:".format(i+1))
        usedModels=[]
        if len(models)==0:
            print("No more model for training. ")
            break
        for model in models:
            print("Model: ",model,model.parameters)
            bestModels,bestTrainAccs,bestTestAccs=geneticSearch(X,y,model,1,30,maxModelNum,bestModels,bestTrainAccs,bestTestAccs,
            checkPointPath=None if checkPointPath is None else checkPointPath+"/fine-tune-{0}-".format(i+1))
        for bestModel in bestModels:
            print("Model: ",model,model.parameters)
            if (trainAcc>0.9*accThr) and (testAcc>1.9*trainAcc-1):
                if trainAcc<accThr:
                    print("Underfit",trainAcc)
                    model=model.underfit(accThr-trainAcc)
                elif testAcc<accThr:
                    print("Overfit".format())
                    model=model.overfit(trainAcc-testAcc)
                if (trainAcc>=accThr) and (testAcc>=accThr):
                    print("Try to make it better. ")
                    model=model.underfit(1-accThr)
                usedModels.append(model)
            else:
                print("Drop it in from next epoch. ")
        models=usedModels
    return bestModels,bestTrainAccs,bestTestAccs

def generate(length,num=50):
    if length==0:
        return [[]]
    parameters=[]
    for i in range(num):
        parameters.append(np.random.uniform(0,1,length))
    return parameters

def evolution(scores,parameters):
    scores=np.array(scores)
    scoresum=np.sum(scores)
    plist=scores/scoresum
    plist=np.cumsum(plist)
    plist=plist.tolist()
    plist.insert(0,0)
    reparameters=[]
    for i in range(len(parameters)):
        sample=random.random()
        for j in range(len(plist)-1):
            if (sample>plist[j])and(sample<plist[j]):
                reparameters.append(parameters[j])
                break
    parameters=reparameters
    if len(parameters)>=2:
        for i in range(0,len(parameters),2):
            w1=parameters[i]
            w2=parameters[i+1]
            splitPoint=random.randrange(1,len(parameters[i]))
            w3=np.concatenate([parameters[i][:splitPoint],parameters[i+1][splitPoint:]])
            w4=np.concatenate([parameters[i+1][:splitPoint],parameters[i][splitPoint:]])
            parameters[i]=w3
            parameters[i+1]=w4
    for weight in parameters:
        for j in range(len(weight)):
            sample=random.random()
            if sample<0.1:
                weight[j]=random.random()
    return parameters

def geneticSearch(X,y,model,maxEpoch,maxSeed,maxModelNum,bestModels=[],bestTrainAccs=[],bestTestAccs=[],checkPointPath=None):
    parameters=generate(model.paraLength,maxSeed)
    for i in range(maxEpoch):
        print("Parameter searching epoch {0}:".format(i+1))
        scores=[]
        for parameter in parameters:
            newmodel,parameters=model.createFromGene(parameter)
            newmodel,trainAcc,testAcc=newmodel.train(X,y)
            scores.append(testAcc)
            print("Parameters: ",parameters)
            print("Score on train set: ",trainAcc)
            print("Score on test set: ",testAcc)
            for j in range(len(bestModels)):
                if testAcc>bestTestAccs[j]:
                    bestModels.insert(j,newmodel)
                    bestTrainAccs.insert(j,trainAcc)
                    bestTestAccs.insert(j,testAcc)
                    print("Ranking {0}. ".format(j+1))
                    if len(bestModels)>maxModelNum:
                        bestModels.pop()
                        bestTrainAccs.pop()
                        bestTestAccs.pop()
                    break
            else:
                if len(bestModels)<maxModelNum:
                    bestModels.append(newmodel)
                    bestTrainAccs.append(trainAcc)
                    bestTestAccs.append(testAcc)
                    print("Ranking {0}. ".format(len(bestModels)))
        if not(checkPointPath is None):
            for j,model in enumerate(bestModels):
                saveModel(model.parameters,checkPointPath+"para-search-{0}-ranking-{1}".format(i+1,j+1)+str()+".pkl")
        parameters=evolution(scores,parameters)
    return bestModels,bestTrainAccs,bestTestAccs