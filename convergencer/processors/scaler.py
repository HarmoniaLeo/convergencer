from convergencer.processors.base import base
from sklearn.preprocessing import RobustScaler
import numpy as np

class normalizeScaler(base):
    def initialize(self, parameters={},verbose=1):
        '''
        params:
            parameters:
            {
                "cols": cols to scan. Default=None for all number cols. 
            }
        '''
        self.verbose=verbose
        self.processCols=self._getParameter("cols",None,parameters)
        return self
                
    def fit(self, data, y=None):
        if self.processCols is None:
            ttn = data.select_dtypes(include=[np.number])
            self.scaleCols=ttn.columns
        else:
            self.scaleCols=self.processCols
        if len(self.scaleCols)!=0:
            self.means=np.mean(data[self.scaleCols],axis=0)
            self.vars=np.var(data[self.scaleCols],axis=0)
        return self

    def transform(self, data, y=None):
        data=data.copy()
        if self.verbose==1:
            print("\n-------------------------Try to normalize data to mu=0, sigma=1-------------------------")
            print("The cols to screen are ",self.scaleCols)
            print("The means are: \n",self.means)
            print("The variations are: \n",self.vars)
        data[self.scaleCols]=(data[self.scaleCols]-self.means)/(self.vars+1e-16)
        return super().transform(data, y=y)
    
    def __str__(self):
        return "normalizeScaler"

class robustScaler(normalizeScaler):
    def fit(self, data, y=None):
        if self.processCols is None:
            ttn = data.select_dtypes(include=[np.number])
            self.scaleCols=ttn.columns
        else:
            self.scaleCols=self.processCols
        self.scaler=RobustScaler()
        self.scaler.fit(data[self.scaleCols])
        return self
                
    def transform(self, data, y=None):
        data=data.copy()
        if self.verbose==1:
            print("\n-------------------------Scaling data with robust scaler-------------------------")
            print("The cols to screen are ",self.scaleCols)
        data[self.scaleCols]=self.scaler.transform(data[self.scaleCols])
        if y is None:
            return data
        else:
            return data,y
    
    def __str__(self):
        return "robustScaler"