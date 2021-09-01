from processors.base import base
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

class fillNa(base):
    def __init__(self, data, y=None, parameters={},verbose=1):
        '''
        parameters:
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
        '''
        self.verbose=verbose
        self.fillNaStrg=self.getParameter("strategy",{},parameters)
        self.fillNaValue=self.getParameter("values",{},parameters)
        for key in self.fillNaValue:
            self.fillNaStrg[key]="value"
        self.fillNaK=self.getParameter("k",5,parameters)
        self.fillNaKCols=self.getParameter("knn cols",{},parameters)

    def impute_knn(self,df,fillNaK,fillNaKCols):
        cols_nan = df.columns[df.isna().any()].tolist()         # columns w/ nan 
        for col in cols_nan:
            nanCount = df[col].isna().sum()
            if self.verbose==1:
                print("Fill {0} nans in col ".format(nanCount)+str(col)+" with default {0}-nn strategy. ".format(fillNaK))
            #imp_test = df[df[col].isna()]   # indicies which have missing data will become our test set
            if col in fillNaKCols.keys():
                imp=df[fillNaKCols[col]]
            else:
                imp=df.drop(cols_nan,axis=1)
            imp=pd.get_dummies(imp)
            means=np.mean(imp,axis=0)
            vars=np.var(imp,axis=0)
            imp=(imp-means)/vars
            imp=imp.dropna(axis=1)
            imp_train= imp.loc[~df[col].isna()]
            imp_test = imp.loc[df[col].isna()]
            if df[col].dtype != np.number:
                model = KNeighborsClassifier(n_neighbors=fillNaK)
            else:
                model = KNeighborsRegressor(n_neighbors=fillNaK)  # KNR Unsupervised Approach
            knr = model.fit(imp_train, df.loc[~df[col].isna(),col])
            df.loc[df[col].isna(), col] = knr.predict(imp_test)
        #Parallel(n_jobs=-1)(delayed(impute_col)(col,df,fillNaK,fillNaKCols) for col in cols_nan)
        return df

    def transform(self, data, y=None):
        if self.verbose==1:
            print("\n-------------------------Try to fill nan values-------------------------")
        data=data.copy()
        for key in data.columns:
            nanCount = data[key].isna().sum()
            if (nanCount>0) and (key in self.fillNaStrg.keys()):
                if ((type(self.fillNaStrg)==str) and (self.fillNaStrg == "mode")) or ((type(self.fillNaStrg)==dict) and (self.fillNaStrg[key] == "mode")):
                    num = data[key].mode()[0]
                    if self.verbose==1:
                        print("Fill {0} nans in col ".format(nanCount)+str(key)+" with mode "+str(num))
                elif ((type(self.fillNaStrg)==str) and (self.fillNaStrg == "mean")) or ((type(self.fillNaStrg)==dict) and (self.fillNaStrg[key] == "mean")):
                    num = data[key].mean()[0]
                    if self.verbose==1:
                        print("Fill {0} nans in col ".format(nanCount)+str(key)+" with mean "+str(num))
                elif ((type(self.fillNaStrg)==str) and (self.fillNaStrg == "auto")) or ((type(self.fillNaStrg)==dict) and (self.fillNaStrg[key] == "auto")):
                    if data[key].dtype != np.number:
                        num = data[key].mode()[0]
                        if self.verbose==1:
                            print("Fill {0} nans in col ".format(nanCount)+str(key)+" with mode "+str(num))
                    else:
                        num = data[key].mean()[0]
                        if self.verbose==1:
                            print("Fill {0} nans in col ".format(nanCount)+str(key)+" with mean "+str(num))
                elif ((type(self.fillNaStrg)==str) and (self.fillNaStrg == "value")) or ((type(self.fillNaStrg)==dict) and (self.fillNaStrg[key] == "value")):
                    num = self.fillNaValue[key]
                    if self.verbose==1:
                        print("Fill {0} nans in col ".format(nanCount)+str(key)+" with "+str(num))
                else:
                    raise Exception("Unsupported filling strategy. ")
                data[key] = data[key].fillna(num)
        return super().transform(self.impute_knn(data,self.fillNaK,self.fillNaKCols), y=y)
    
    def __str__(self):
        return "fillNa"