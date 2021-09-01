from convergencer.processors.base import base
import numpy as np
from convergencer.utils.processing import normalTest

class normalizeFilter(base):
    def __init__(self, data, y=None, parameters={},verbose=1):
        '''
        parameters:
            {
                "cols": cols to scan. Default=None for all normal distributed number cols. 
            }
        '''
        self.verbose=verbose
        cols=self.getParameter("cols",None,parameters)
        if cols is None:
            ttn = data.select_dtypes(include=[np.number])
            cols=ttn.columns
        self.filterCols,_=normalTest(data,cols)
        if len(self.filterCols)!=0:
            self.means=data[self.filterCols].mean()
            self.stds=data[self.filterCols].std()
    
    def transform(self, data, y):
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
        if self.verbose==1:
            print("\n-------------------------Try to drop rows with nan values-------------------------")
        data = data.dropna()
        return super().transform(data, y=y[data.index])
    
    def __str__(self):
        return "naRowFilter"