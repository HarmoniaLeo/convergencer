from convergencer.utils.io import readData,readLabel
import numpy as np
from convergencer.processors import naColFilter,fillNa,custom,tsToNum,catToNum,numToCat,variationSelector,entropySelector,mutInfoSelector,correlationSelector,normalization,normalizeFilter
#from models import linear,ridge,lasso,elasticNet,SVMRegression,dtRegression,rfRegression,gbRegression,xgbRegression,lgbmRegression,lgbmRegression_goss,catBoostRegression

class convergencerRegressionData:
    def __init__(self,X_train,X_test,label=-1,id=None,labelId=None,delimiter=','):
        train=readData(X_train,delimiter,id)
        X,self.label=readLabel(train,label,delimiter,labelId)
        test=readData(X_test,delimiter,id)
        self.trainNum=X.shape[0]
        self.data=X.append(test)
    
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
    
    def process(self,name,func,processors,params):
        if name in processors:
            if name in params.keys():
                param=params[name]
            else:
                param={}
            p=func(self.data,self.label,parameters=param)
            self.data,self.label=p.transform(self.data,self.label)

    def preprocess(self,processors=["naColFilter","fillNa","custom","tsToNum","catToNum","numToCat","variationSelector","entropySelector","mutInfoSelector","correlationSelector","normalization","normalizeFilter"],params={}):
        '''
        params:
        {
            "naColFilter":
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
            "custom":
                {
                    "function": your custom processing function
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
        }   
        '''
        self.process("naColFilter",naColFilter,processors,params)
        self.process("fillNa",fillNa,processors,params)
        self.process("custom",custom,processors,params)
        self.process("tsToNum",tsToNum,processors,params)
        self.process("catToNum",catToNum,processors,params)
        self.process("numToCat",numToCat,processors,params)
        self.process("variationSelector",variationSelector,processors,params)
        self.process("entropySelector",entropySelector,processors,params)
        self.process("mutInfoSelector",mutInfoSelector,processors,params)
        self.process("correlationSelector",correlationSelector,processors,params)
        self.process("normalization",normalization,processors,params)
        if "normalizeFilter" in processors:
            if "normalizeFilter" in params.keys():
                param=params["normalizeFilter"]
            else:
                param={}
            X=self.data.iloc[:self.trainNum]
            test=self.data.iloc[self.trainNum:]
            p=normalizeFilter(X,self.label,parameters=param)
            X,self.label=p.transform(X,self.label)
            self.trainNum=X.shape[0]
            self.data=X.append(test)
        self.preprocessed=True

    def getXtrain(self):
        return self.data.iloc[:self.trainNum]
    
    def getXtest(self):
        return self.data.iloc[self.trainNum:]
    
    def gety(self):
        return self.label

'''
def regressionParaSearch(data,models=["linear","ridge","lasso","elasticNet","SVMRegression","dtRegression","rfRegression","gbRegression","xgbRegression","lgbmRegression","lgbmRegression_goss","catBoostRegression"],
    params={},historySavePath=None,historyLoadPath=None,modelSavePath=None,modelLoadPath=None):
    if not (data.preprocessed):
        data.preprocess()

def regressionSingleModel(data,model="linear",params={},modelSavePath=None,modelLoadPath=None):

def regressionBaggingModel(data,models=[],params={},modelSavePath=None,modelLoadPath=None):
'''