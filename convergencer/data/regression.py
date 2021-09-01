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
        self.X_train=X
        self.X_test=test
        self.ps=[]
    
    def summary(self):
        data=self.X_train.append(self.X_test)
        print("All columns: ",data.columns)
        ttn = data.select_dtypes(exclude=[np.number])
        print("Categorical columns: ",ttn.columns)
        ttn = data.select_dtypes(include=[np.number])
        print("Numerical columns: ",ttn.columns)
        percent = (data.isna().sum()/data.isna().count()).sort_values(ascending=False)
        print("Columns with nan values: ",percent[percent>0])
    
    def ifNan(self,col):
        data=self.X_train.append(self.X_test)
        print(data[col].isna().sum()/data[col].isna().count())
    
    def process(self,name,func,processors,params,data):
        if name in processors:
            if name in params.keys():
                param=params[name]
            else:
                param={}
            p=func(data,self.label,parameters=param)
            data,self.label=p.transform(data,self.label)
            self.ps.append(p)

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
        data=self.X_train.append(self.X_test)
        self.process("naColFilter",naColFilter,processors,params,data)
        self.process("fillNa",fillNa,processors,params,data)
        self.process("custom",custom,processors,params,data)
        self.process("tsToNum",tsToNum,processors,params,data)
        self.process("catToNum",catToNum,processors,params,data)
        self.process("numToCat",numToCat,processors,params,data)
        self.process("variationSelector",variationSelector,processors,params,data)
        self.process("entropySelector",entropySelector,processors,params,data)
        self.process("mutInfoSelector",mutInfoSelector,processors,params,data)
        self.process("correlationSelector",correlationSelector,processors,params,data)
        self.process("normalization",normalization,processors,params,data)
        if "normalizeFilter" in processors:
            if "normalizeFilter" in params.keys():
                param=params["normalizeFilter"]
            else:
                param={}
            p=normalizeFilter(self.X_train,self.label,parameters=param)
            self.X_train,self.label=p.transform(self.X_train,self.label)
        self.preprocessed=True

    def getXtrain(self):
        return self.X_train
    
    def getXtest(self):
        return self.X_test
    
    def gety(self):
        return self.label
    
    def reProcess(self,label):
        for p in self.ps:
            if str(p)=="normalization":
                _,label=p.reTransform(None,label)
                return label


'''
def regressionParaSearch(data,models=["linear","ridge","lasso","elasticNet","SVMRegression","dtRegression","rfRegression","gbRegression","xgbRegression","lgbmRegression","lgbmRegression_goss","catBoostRegression"],
    params={},historySavePath=None,historyLoadPath=None,modelSavePath=None,modelLoadPath=None):
    if not (data.preprocessed):
        data.preprocess()

def regressionSingleModel(data,model="linear",params={},modelSavePath=None,modelLoadPath=None):

def regressionBaggingModel(data,models=[],params={},modelSavePath=None,modelLoadPath=None):
'''