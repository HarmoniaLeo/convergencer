from utils.processing import get_entropy
from base import base
import numpy as np
import pandas as pd
from sklearn import metrics

class singleSelector(base):
    def __init__(self, parameters):
        self.cols=self.getParameter("cols",None,parameters)
        self.threshold=self.getParameter("threshold",0.9,parameters)
    
    def fit(self,data):
        if self.cols is None:
            cols=[key for key in data.columns if data[key].dtype!=object]
        else:
            assert type(self.cols)==list
            cols=self.cols
        if len(cols)!=0:
            numCols=[key for key in data.columns if data[key].dtype!=object]
            vars = self.getTarget()
            vars = vars.sort_values()
            vars = vars.cumsum()
            varsSum = vars.sum()
            varsRate = vars/varsSum
            cols = [col for col in cols if varsRate[col]<(1-self.threshold)]
            data = data.drop(cols)
            self.dropCols=cols
            self.printMessage(cols)
        return data
    
    def transform(self, data):
        return data.drop(self.dropCols)

    def __str__(self):
        return "singleSelector"

class dropNa(singleSelector):    
    def fit(self,data):
        if self.cols is None:
            cols=[key for key in data.columns if data[key].dtype!=object]
        else:
            assert type(self.cols)==list
            cols=self.cols
        if len(cols)!=0:
            percent = (data[cols].isna().sum()/data[cols].isna().count()).sort_values(ascending=False)
            data = data.drop(percent[percent>self.dropNaThr].index,axis=1)
            print("Drop these cols since they high rate of nan values: ".format(self.dropNaThr),percent[percent>self.dropNaThr].index)
            self.dropCols=percent[percent>self.dropNaThr].index
        return data
    
    def __str__(self):
        return "dropNa"

class variationSelector(singleSelector):
    def getTarget(self,data,numCols):
        return pd.var(data[numCols],axis=0)
    
    def printMessage(self,cols):
        print("Drop these cols since they have low variance: ",cols)
    
    def __str__(self):
        return "variationSelector"

class entropySelector(singleSelector):
    def getTarget(self,data,numCols):
        return data[numCols].apply(get_entropy)
    
    def printMessage(self,cols):
        print("Drop these cols since they have low entropy: ",cols)
    
    def __str__(self):
        return "entropySelector"

class mutualSelector(singleSelector):
    def fit(self,data):
        if self.cols is None:
            cols=[key for key in data.columns if data[key].dtype!=object]
        else:
            assert type(self.cols)==list
            cols=self.cols
        self.dropCols=[]
        if len(cols)!=0:
            numCols=[key for key in data.columns if data[key].dtype!=object]
            for col in numCols:
                if col in self.dropCols: continue
                for col2 in cols:
                    if self.getTarget(data,col,col2)>self.threshold:
                        if col2 not in self.dropCols: self.dropCols.append(col2)
                        self.printMessage(col,col2)
            data=data.drop(self.dropCols)
        return data

    def __str__(self):
        return "mutualSelector"

class correlationSelector(mutualSelector):
    def getTarget(self,data,col,col2):
        return np.abs(np.correlate(data[col],data[col2]))
    
    def printMessage(self,col,col2):
        print("Drop col "+str(col2)+" since it highly correlated to "+str(col))

    def __str__(self):
        return "correlationSelector"

class mutInfoSelector(mutualSelector):
    def getTarget(self,data,col,col2):
        return metrics.normalized_mutual_info_score(data[col],data[col2])
    
    def printMessage(self,col,col2):
        print("Drop col "+str(col2)+" since it has high mutual infomation with "+str(col))

    def __str__(self):
        return "mutInfoSelector"