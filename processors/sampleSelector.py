from base import base
import numpy as np

class normalizeFilter(base):
    def __init__(self, parameters):
        self.filtercols=self.getParameter("cols",None,parameters)
    
    def fit(self,data):
        if self.filterCols is None:
            ttn = data.select_dtypes(include=[np.number])
            cols=ttn.columns
        else:
            assert type(self.filterCols)==list
            cols=self.filterCols
        if len(cols)!=0:
            self.means=data[cols].mean()
            self.stds=data[cols].std()
            print("The means are: ",self.means)
            print("The standard deviations are: ",self.stds)
            data=self.transform(data)
        return data
    
    def transform(self, data):
        print("Try to filter data with normal distribution. ")
        print("The cols to screen are ",self.filtercolscols)
        indexs=(data[self.filtercols]>(self.means+3*self.stds))|(data[self.filtercols]<(self.means-3*self.stds))
        index=indexs.any(axis=1)
        data=data.loc[~index]
        print("Dropped rows: ",np.argwhere(index==1))
        return data
    
    def __str__(self):
        return "normalizeFilter"

class naRowFilter(base):
    def transform(self, data):
        data = data.dropna()
        return data
    
    def __str__(self):
        return "naRowFilter"