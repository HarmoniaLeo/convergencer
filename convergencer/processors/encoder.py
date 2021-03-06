from convergencer.processors.base import base
from convergencer.processors.naProcess import fillNa
import pandas as pd
import numpy as np

class catToInt(base):
    def fit(self, data, y=None):
        ttn = data.select_dtypes(exclude=[np.number])
        self.uniques={}
        cols=ttn.columns
        for col in cols:
            self.uniques[col]=np.unique(data[col])  
        return self

    def transform(self, data, y=None):
        data=data.copy()
        for col in self.uniques.keys():
            uniques=np.unique(data[col])
            cat=data[col].dtype
            for uni in uniques:
                if uni not in self.uniques[col]:
                    data[col]=np.where(data[col]==uni,np.nan,data[col])
            data[col]=data[col].astype(cat)
        data=fillNa().initialize({},verbose=0).fit(data,y).transform(data)
        for col in self.uniques.keys():
            uniques=np.unique(data[col])
            for uni in uniques:
                data[col]=np.where(data[col]==uni,np.argwhere(self.uniques[col]==uni)[0],data[col])
            data[col]=data[col].astype(int)
        if self.verbose==1:
            print("\n-------------------------Encoding categorical data to ints-------------------------")
            print(self.uniques.keys()," have been encoded to ints. ")
        return super().transform(data, y=y)
    
    def __str__(self):
        return "catToInt"

class catToOneHot(catToInt):
    def transform(self, data, y=None):
        data=data.copy()
        for col in self.uniques.keys():
            for uni in self.uniques[col]:
                data[str(col)+"_"+str(uni)]=np.where(data[col]==uni,1,0)
            data=data.drop(col,axis=1)
        if self.verbose==1:
            print("\n-------------------------Encoding categorical data to one-hotted codes-------------------------")
            print(self.uniques.keys()," have been encoded to One-Hot code. ")
        if y is None:
            return data
        else:
            return data,y
    
    def __str__(self):
        return "catToOneHot"

class catToMean(base):
    def fit(self, data, y=None):
        ttn = data.select_dtypes(exclude=[np.number])
        self.uniques={}
        cols=ttn.columns
        for col in cols:
            self.uniques[col]={"unis":[],"values":[]}
            for uni in np.unique(data[col]):
                self.uniques[col]["unis"].append(uni)
                self.uniques[col]["values"].append(np.mean(y[data[col]==uni]))
        return self

    def transform(self, data, y=None):
        data=data.copy()
        for col in self.uniques.keys():
            uniques=np.unique(data[col])
            for uni in uniques:
                if uni not in self.uniques[col]["unis"]:
                    data[col]=np.where(data[col]==uni,np.nan,data[col])
        p=fillNa().initialize({},verbose=0).fit(data,y)
        data=p.transform(data)
        for col in self.uniques.keys():
            for uni,value in zip(self.uniques[col]["unis"],self.uniques[col]["values"]):
                data[col]=np.where(data[col]==uni,value,data[col])
            data[col]=data[col].astype(float)
        if self.verbose==1:
            print("\n-------------------------Encoding categorical data to mean of label-------------------------")
            print(self.uniques.keys()," have been encoded to mean value of y. ")
        return super().transform(data, y=y)
    
    def __str__(self):
        return "catToMean"

class catToPdCat(base):
    def fit(self, data, y=None):
        ttn = data.select_dtypes(exclude=[np.number])
        self.cols=ttn.columns
        return self
    
    def transform(self, data, y=None):
        data=data.copy()
        if len(self.cols)>0:
            data[self.cols]=data[self.cols].astype("category")
        return super().transform(data,y)
    
    def __str__(self):
        return "catToPdCat"

class catToIntPdCat(catToPdCat):
    def fit(self, data, y=None):
        self.intEncoder=catToInt().initialize({},verbose=0).fit(data,y)
        return super().fit(data, y=y)

    def transform(self, data, y=None):
        data=self.intEncoder.transform(data)
        return super().transform(data,y)
    
    def __str__(self):
        return "catToIntPdCat"

class catToNum(base):
    def initialize(self, parameters={},verbose=1):
        '''
        parameters:
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
        '''
        self.verbose=verbose
        self.orders=self._getParameter("orders",{},parameters)
        return self

    def transform(self, data, y=None):
        data=data.copy()
        if self.verbose==1:
            print("\n-------------------------Encoding categorical data to specific numbers-------------------------")
        for key in self.orders.keys():
            if type(self.orders[key])==list:
                for i in range(len(self.orders[key])):
                    data[key]=np.where(data[key]==k,i,data[key])
                    if self.verbose==1:
                        print(str(self.orders[key][i])+" in col "+str(key)+" has been set to "+str(i))
            elif type(self.orders[key])==dict:
                for k in self.orders[key].keys():
                    data[key]=np.where(data[key]==k,self.orders[key][k],data[key])
                    if self.verbose==1:
                        print(str(k)+" in col "+str(key)+" has been set to "+str(self.orders[key][k]))
                try:
                    data[key]=data[key].astype(int)
                except Exception:
                    raise Exception("Value "+str(k)+" in col "+str(key)+" didn't been specified a num. ")
            else:
                raise Exception("Unsupported orders type. ")
        return super().transform(data, y=y)
    
    def __str__(self):
        return "catToNum"

class numToCat(base):
    def initialize(self, parameters={},verbose=1):
        '''
        parameters:
            {
                "cols": list of number cols you want to turn to category cols
            }
        '''
        self.verbose=verbose
        self.cols=self._getParameter("cols",[],parameters)
        return self

    def transform(self, data, y=None):
        data=data.copy()
        if self.verbose==1:
            print("\n-------------------------Encoding specific numerical cols to categorical cols-------------------------")
        for col in self.cols:
            assert col in data.columns
            data[col]=data[col].astype("category")
            if self.verbose==1:
                print(str(col)+" has been set to category col. ")
        return super().transform(data, y=y)
    
    def __str__(self):
        return "numToCat"