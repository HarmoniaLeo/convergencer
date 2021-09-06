from convergencer.processors.base import base
import numpy as np
from convergencer.utils.processing import normalTest
from convergencer.processors import customFeatureEngineer

class normalizeFilter(base):
    def initialize(self, parameters={},verbose=1):
        '''
        parameters:
            {
                "cols": cols to scan. Default=None for all normal distributed number cols. 
            }
        '''
        self.verbose=verbose
        self.processCols=self._getParameter("cols",None,parameters)
        return self
    
    def fit(self, data, y=None):
        if self.processCols is None:
            ttn = data.select_dtypes(include=[np.number])
            cols=ttn.columns
        else:
            cols=self.processCols
        self.filterCols,_=normalTest(data,cols)
        if len(self.filterCols)!=0:
            self.means=data[self.filterCols].mean()
            self.stds=data[self.filterCols].std()
        return self
    
    def transform(self, data, y):
        data=data.copy()
        if len(self.filterCols)!=0:
            if self.verbose==1:
                print("\n-------------------------Try to filter data with normal distribution-------------------------")
                print("Scanning cols: ",self.filterCols)
                print("The means are: \n",self.means)
                print("The standard deviations are: \n",self.stds)
            indexs=(data[self.filterCols]>(self.means+3*self.stds))|(data[self.filterCols]<(self.means-3*self.stds))
            index=indexs.any(axis=1)
            data=data.loc[~index]
            if self.verbose==1:
                print("Dropped rows: ",np.where(index))
        return super().transform(data, y=y[data.index])

    def __str__(self):
        return "normalizeFilter"

class naRowFilter(base):
    def transform(self, data, y=None):
        data=data.copy()
        if self.verbose==1:
            print("\n-------------------------Try to drop rows with nan values-------------------------")
        data = data.dropna()
        return super().transform(data, y=y[data.index])
    
    def __str__(self):
        return "naRowFilter"

def defaultFunc(data):
    return data

class customFilter(base):
    def initialize(self,parameters={},verbose=1):
        '''
        parameters:
            {
                "transform": 
                    def transform(data):
                        ...
                        return data
            }
        '''
        self.verbose=verbose
        self.transFunction=self._getParameter("transform",defaultFunc,parameters)
        return self
    
    def transform(self, data, y=None):
        if self.verbose==1:
            print("\n-------------------------Using custom fuction to filter samples-------------------------")
        if not (y is None):
            y=y.copy()
        data=data.copy()
        data=self.transFunction(data)
        y=y.loc[data.index]
        return super().transform(data, y=y)

    def __str__(self):
        return "customFilter"