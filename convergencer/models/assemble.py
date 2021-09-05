import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from convergencer.utils.metrics import getMetric
from sklearn.model_selection import train_test_split

def stackingPredict(X_train,y_train,X_test,models,metaModel,metric="r2",paraSearch=True,maxEpoch=1000,modelLoadPath=None,modelSavePath=None,modelSaveFreq=50,historySavePath=None,historySaveFreq=50):
    print("-------------------------------------Stacking predict start-------------------------------------")
    kf = KFold(n_splits=5,shuffle=True)
    A_s={}
    B_s={}
    for key in models.keys():
        a_s=None
        b_s=pd.DataFrame()
        i=0
        for train_index, test_index in kf.split(X_train):
            X_train_train=X_train.iloc[train_index]
            y_train_train=y_train.iloc[train_index]
            X_train_test=X_train.iloc[test_index]
            models[key].fit(X_train_train,y_train_train,metric)
            if a_s is None:
                a_s=models[key].predict(X_train_test)
            else:
                a_s=a_s.append(models[key].predict(X_train_test))
            b_s[i]=models[key].predict(X_test)
            i+=1
        A_s[key]=a_s
        B_s[key]=b_s.mean(axis=1)
    A_s=pd.DataFrame(A_s,index=X_train.index)
    B_s=pd.DataFrame(B_s,index=X_test.index)
    if paraSearch:
        metaModel.parasearchFit(A_s,y_train,metric,maxEpoch,modelSavePath,modelSaveFreq,historySavePath,historySaveFreq)
    else:
        metaModel.fit(A_s,y_train,metric,modelLoadPath,modelSavePath)
    return metaModel.predict(B_s)

def baggingAnalyse(predictDict,modelNum=3,metric="r2",modelSelect=False,percentage=0.3):
    print("-------------------------------------Bagging analyse start-------------------------------------")
    metric=getMetric(metric)
    predictDict=pd.DataFrame(predictDict)
    predictDict=predictDict.T
    clf = KMeans(n_clusters=modelNum)
    clf.fit(predictDict)
    result={}
    labels=clf.labels_
    for i,key in enumerate(predictDict.index):
        result[key]=(labels[i])
    return result

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