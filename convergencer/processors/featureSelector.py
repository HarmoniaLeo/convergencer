from processors.base import base
import numpy as np
import pandas as pd
from sklearn import metrics,preprocessing

class singleSelector(base):
    def __init__(self, data, y=None, parameters={},verbose=1):
        '''
        parameters:
            {
                "cols": cols you want to process. Default=None means all cols. 
                "threshold": Default=0.9. Meaning is different in each selector. 
                    naColFilter: the col with nan > threshold will be removed
                    singleSelector: the col with value < 1-threshold will be removed 
                    mutualSelector: the cols with value > threshold will be removed one
            }
        '''
        self.verbose=verbose
        threshold=self.getParameter("threshold",0.9,parameters)
        cols=self.getParameter("cols",None,parameters)
        if cols is None:
            cols=self.getCols(data)
        if len(cols)!=0:
            self.fit(data,cols,threshold)
        else:
            self.dropCols=[]

    def getCols(self,data):
        return data.columns
    
    def fit(self,data,cols,threshold):
        vars = self.getTarget(data,cols)
        vars = vars.sort_values()
        vars = vars.cumsum()
        varsSum = vars.sum()
        varsRate = vars/varsSum
        cols = [col for col in cols if varsRate[col]<(1-threshold)]
        self.dropCols=cols
    
    def transform(self, data, y=None):
        if self.verbose==1:
            print("\n-------------------------Selcecting features with "+str(self)+"-------------------------")
            self.printMessage(self.dropCols)
        return super().transform(data.drop(self.dropCols,axis=1), y=y)        

    def __str__(self):
        return "singleSelector"

class naColFilter(singleSelector):

    def fit(self,data,cols,threshold):
        percent = (data[cols].isna().sum()/data[cols].isna().count()).sort_values(ascending=False)
        self.dropCols=percent[percent>=threshold].index
        
    def printMessage(self,cols):
        print("Drop "+str(len(cols))+" cols since they high rate of nan values: ",cols)
    
    def __str__(self):
        return "naColFilter"

class variationSelector(singleSelector):
    def getCols(self,data):
        ttn = data.select_dtypes(include=[np.number])
        cols=ttn.columns
        return cols

    def getTarget(self,data,numCols):
        return np.var(data[numCols],axis=0)
    
    def printMessage(self,cols):
        print("Drop "+str(len(cols))+" cols since they have low variance: ",cols)
    
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
        print("Drop "+str(len(cols))+" cols since they have low entropy: ",cols)
    
    def __str__(self):
        return "entropySelector"

class mutualSelector(singleSelector):
    def fit(self,data,cols,threshold):
        targetData=data[cols]
        targetData=self.getTarget(targetData)
        indexs=np.argwhere(targetData>threshold)
        indexs=np.array(indexs)
        indexs=indexs[np.nonzero(indexs[:,0]<indexs[:,1])[0]]
        self.dropCols=indexs[:,1].tolist()
        for i in range(len(self.dropCols)):
            self.dropCols[i]=cols[self.dropCols[i]]

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
        print("Drop "+str(len(cols))+" cols since they highly related to other cols: ",cols)

    def __str__(self):
        return "correlationSelector"

class mutInfoSelector(mutualSelector):
    def getCols(self,data):
        ttn = data.select_dtypes(exclude=[np.number])
        cols=ttn.columns
        return cols

    def getTarget(self,data):
        number = data.shape[1]
        List = []
        le = preprocessing.LabelEncoder()
        for i in range(number):
            A = []
            X = data[data.columns[i]]
            X = le.fit_transform(X.astype(str))
            for j in range(number):
                Y = data[data.columns[j]]
                Y = le.fit_transform(Y.astype(str))
                A.append(metrics.normalized_mutual_info_score(X, Y))
            List.append(A)  # List是列表格式
        return np.array(List)
    
    def printMessage(self,cols):
        print("Drop "+str(len(cols))+" cols since they have high mutual infomation with other cols: ",cols)

    def __str__(self):
        return "mutInfoSelector"