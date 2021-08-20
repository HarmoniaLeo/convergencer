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
            cols=self.getCols(data)
        else:
            assert type(self.cols)==list
            cols=self.cols
        if len(cols)!=0:
            vars = self.getTarget(data,cols)
            vars = vars.sort_values()
            vars = vars.cumsum()
            varsSum = vars.sum()
            varsRate = vars/varsSum
            cols = [col for col in cols if varsRate[col]<(1-self.threshold)]
            data = data.drop(cols,axis=1)
            self.dropCols=cols
            self.printMessage(cols)
        return data
    
    def transform(self, data):
        return data.drop(self.dropCols,axis=1)

    def __str__(self):
        return "singleSelector"

class naColFilter(singleSelector):    
    def fit(self,data):
        if self.cols is None:
            cols=data.columns
        else:
            assert type(self.cols)==list
            cols=self.cols
        if len(cols)!=0:
            percent = (data[cols].isna().sum()/data[cols].isna().count()).sort_values(ascending=False)
            data = data.drop(percent[percent>self.threshold].index,axis=1)
            print("Drop these cols since they high rate of nan values: ".format(self.threshold),percent[percent>self.threshold].index)
            self.dropCols=percent[percent>self.threshold].index
        return data
    
    def __str__(self):
        return "naColFilter"

class variationSelector(singleSelector):
    def getCols(self,data):
        ttn = data.select_dtypes(include=[np.number])
        cols=ttn.columns
        return cols

    def getTarget(self,data,numCols):
        return pd.var(data[numCols],axis=0)
    
    def printMessage(self,cols):
        print("Drop these cols since they have low variance: ",cols)
    
    def __str__(self):
        return "variationSelector"

def get_entropy(s):
    pe_value_array = s.unique()
    ent = 0.0
    for x_value in pe_value_array:
        p = float(s[s == x_value].shape[0]) / s.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent

class entropySelector(singleSelector):
    def getCols(self,data):
        ttn = data.select_dtypes(exclude=[np.number])
        cols=ttn.columns
        return cols

    def getTarget(self,data,numCols):
        return data[numCols].apply(get_entropy)
    
    def printMessage(self,cols):
        print("Drop these cols since they have low entropy: ",cols)
    
    def __str__(self):
        return "entropySelector"

class mutualSelector(singleSelector):
    def fit(self,data):
        if self.cols is None:
            cols=self.getCols(data)
        else:
            assert type(self.cols)==list
            cols=self.cols
        self.dropCols=[]
        if len(cols)!=0:
            targetData=data[cols]
            targetData=self.getTarget(data)
            indexs=np.argwhere(targetData>self.threshold)
            indexs=np.array(indexs)
            indexs=indexs[np.nonzero(indexs[:,0]<indexs[:,1])[0]]
            self.dropCols=indexs[:,1].tolist()
            self.printMessage(self.dropCols)
        return data

    def __str__(self):
        return "mutualSelector"

class correlationSelector(mutualSelector):
    def getCols(self,data):
        ttn = data.select_dtypes(include=[np.number])
        cols=ttn.columns
        return cols

    def getTarget(self,data):
        corrmat=data.corr()
        return np.abs(corrmat.values)
    
    def printMessage(self,cols):
        print("Drop these cols since they highly related to other cols: ",cols)

    def __str__(self):
        return "correlationSelector"

class mutInfoSelector(mutualSelector):
    def getCols(self,data):
        ttn = data.select_dtypes(exclude=[np.number])
        cols=ttn.columns
        return cols

    def getTarget(self,data,col,col2):
        number = data.shape[0]
        List = []
        for i in range(number):
            A = []
            X = data[data.columns[i]]
            for j in range(number):
                Y = data[data.columns[j]]
                A.append(metrics.normalized_mutual_info_score(X, Y))
            List.append(A)  # List是列表格式
        return np.array(List)
    
    def printMessage(self,cols):
        print("Drop these cols since they have high mutual infomation with other cols: ",cols)

    def __str__(self):
        return "mutInfoSelector"