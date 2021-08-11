import numpy as np
import random

def finetune(X,y,models,maxEpoch,accThr,maxModelNum,checkPointPath=None):
    bestModels=[]
    bestTrainAccs=[]
    bestTestAccs=[]
    for i in range(maxEpoch):
        print("Fine-tuning epoch {0}:".format(i+1))
        usedModels=[]
        if len(models)==0:
            print("No more model for training. ")
            break
        for model in models:
            print("Model: ",model)
            '''
            if len(bestModels)==0:
                bestModels.append(model)
                bestTrainAccs.append(trainAcc)
                bestTestAccs.append(testAcc)
                print("Ranking 1")
            '''
            trainAcc,testAcc=model.train(X,y,checkPointPath=checkPointPath+"/fine-tune-{0}-".format(i))
            for i in range(len(bestModels)):
                index=i
                if testAcc>bestTestAccs[i]:
                    bestModels.insert(index,model)
                    bestTrainAccs.insert(index,trainAcc)
                    bestTestAccs.insert(index,testAcc)
                    print("Ranking {0}. ".format(i+1))
                    if len(bestModels)>maxModelNum:
                        bestModels.pop()
                        bestTrainAccs.pop()
                        bestTestAccs.pop()
                    break
            else:
                if len(bestModels)<maxModelNum:
                    bestModels.append(model)
                    bestTrainAccs.append(trainAcc)
                    bestTestAccs.append(testAcc)
                    print("Ranking {0}. ".format(len(bestModels)))
            if trainAcc>0.9*accThr:
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