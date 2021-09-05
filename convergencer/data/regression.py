from convergencer.utils.io import readData,readLabel
import numpy as np
from convergencer.processors import naColSelector,fillNa,customFeatureEngineer,tsToNum,catToNum,numToCat,variationSelector,entropySelector,mutInfoSelector,correlationSelector,normalization,normalizeFilter,customFilter
from sklearn.model_selection import train_test_split

class convergencerRegressionData:
    def __init__(self,X_train,X_test=None,label=-1,id=None,labelId=None,delimiter=','):
        train=readData(X_train,delimiter,id)
        X,self.label=readLabel(train,label,delimiter,labelId)
        if not(X_test is None):
            test=readData(X_test,delimiter,id)
            self.data=X.append(test)
        else:
            self.data=X
        self.trainNum=X.shape[0]
        self.processors=[naColSelector(),fillNa(),customFeatureEngineer(),tsToNum(),catToNum(),numToCat(),variationSelector(),entropySelector(),mutInfoSelector(),correlationSelector(),normalization(),
        normalizeFilter(),customFilter()]
        self.processorsUsed=[]
    
    def summary(self):
        print("All columns: ",self.data.columns)
        ttn = self.data.select_dtypes(exclude=[np.number])
        print("Categorical columns: ",ttn.columns)
        ttn = self.data.select_dtypes(include=[np.number])
        print("Numerical columns: ",ttn.columns)
        percent = (self.data.isna().sum()/self.data.isna().count()).sort_values(ascending=False)
        print("Columns with nan values: ",percent[percent>0])
    
    def ifNan(self,col):
        print(self.data[col].isna().sum()/self.data[col].isna().count())

    def preprocess(self,processors=["naColSelector","fillNa","customFeatureEngineer","tsToNum","catToNum","numToCat","variationSelector","entropySelector","mutInfoSelector","correlationSelector",
    "normalization","normalizeFilter","customFilter","normalization"],params={}):
        '''
        params:
        {
            "naColSelector":
                {
                    "cols": cols you want to process. Default=None means all cols. 
                    "threshold": Default=0.9. The col with nan > threshold will be removed.
                }
            "fillNa":
                {
                    "strategy": filling strategy for each col
                        {
                            "col1":"mean"/"mode"/"auto"(mean for num, mode for cat)/"value"
                            "col2":"mean"/"mode"/"auto"(mean for num, mode for cat)/"value"
                            ...
                        }
                        nan in other cols will be filled with default method "knn"
                    "values": fill nan in cols with certain value
                        {
                            "col1":value
                            "col2":value
                            ...
                        }
                        cols above will automatically go to "strategy" as a new item: "col":"value"
                    "k": k of default method "knn". Default=5
                    "knn cols": use values in specific cols for knn method of each col
                        {
                            "col1":["col2","col3",...]
                            "col2":["col1","col3",...]
                            ...
                        }
                        other cols will use all values
                }
            "customFeatureEngineer":
                {
                    "fit": 
                        def fit(data,y):
                            ...
                            return params
                    "transform": 
                        def transform(data,y,params):
                            ...
                            return data,y
                    "reTransform":
                        def reTransform(data,y,params):
                            ...
                            return data,y
                }
            "tsToNum":
                {
                    "formats": formats of each timestamp col
                        {
                            "col1":"format1",
                            "col2":"format2",
                            ...
                        }
                }
            "catToNum":
                {
                    "orders": the catgory cols you want to turn to number cols and which category is which number. 
                        {
                            "col1":
                                {
                                    "cat1":num1,
                                    "cat2":num2,
                                    ...
                                }
                            "col2":
                                {
                                    "cat1":num1,
                                    "cat2":num2,
                                    ...
                                }
                            ...
                        }
                        or
                        {
                            "col1":["cat1","cat2",...]
                            "col2":["cat1","cat2",...]
                            ...
                        }
                        "cat1" will be 0, "cat2" will be 1, ...
                }
            "numToCat":
                {
                    "cols": list of number cols you want to turn to category cols
                }
            "variationSelector":
                {
                    "cols": cols you want to process. Default=None means all cols. 
                    "threshold": Default=0.9. The col with value < 1-threshold will be removed 
                }
            "entropySelector":
                {
                    "cols": cols you want to process. Default=None means all cols. 
                    "threshold": Default=0.9. The col with value < 1-threshold will be removed 
                }
            "mutInfoSelector":
                {
                    "cols": cols you want to process. Default=None means all cols. 
                    "threshold": Default=0.9. The cols with value > threshold will be removed one
                }
            "correlationSelector":
                {
                    "cols": cols you want to process. Default=None means all cols. 
                    "threshold": Default=0.9. The cols with value > threshold will be removed one
                }
            "normalization":
                {
                    "cols": cols to be turned to normal distribution. Default=None for all number cols. 
                }
            "normalizeFilter":
                {
                    "cols": cols to scan. Default=None for all normal distributed number cols. 
                }
            "customFilter":
                {
                    "fit": 
                        def fit(data,y):
                            ...
                            return params
                    "transform": 
                        def transform(data,y,params):
                            ...
                            return data,y
                    "reTransform":
                        def reTransform(data,y,params):
                            ...
                            return data,y
                }
        }   
        '''
        for p1 in processors:
            if type(p1)==str:
                for p2 in self.processors:
                    if str(p2)==p1:
                        if p1 in params.keys():
                            parameter=params[p1]
                        else:
                            parameter={}
                        self.processorsUsed.append(p2.initialize(parameter))
            else:
                self.processorsUsed.append(p1)
        for p in self.processorsUsed:
            if "Filter" in str(p):
                X=self.data.iloc[:self.trainNum]
                test=self.data.iloc[self.trainNum:]
                p.fit(X,self.label)
                X,self.label=p.transform(X,self.label)
                self.trainNum=X.shape[0]
                self.data=X.append(test)
            else:
                self.data,self.label=p.fit(self.data,self.label).transform(self.data,self.label)
        self.preprocessed=True

    def getXtrain(self):
        return self.data.iloc[:self.trainNum]
    
    def getXtest(self):
        return self.data.iloc[self.trainNum:]
    
    def gety(self):
        return self.label
    
    def trainValSplit(self,valRate=0.2):
        return train_test_split(self.data.iloc[:self.trainNum], self.label, test_size=0.2)

    def reProcessy(self,label):
        for p in self.processorsUsed[::-1]:
            _,label=p.reTransform(None,label)
            return label