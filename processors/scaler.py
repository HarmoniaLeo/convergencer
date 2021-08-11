from base import base
import numpy as np

class normalizeScaler(base):
    def __init__(self, parameters):
        self.scaleCols=self.getParameter("cols",None,parameters)
    
    def fit(self, data):
        if self.scaleCols is None:
            cols=[key for key in data.columns if data[key].dtype!=object]
        else:
            assert type(self.scaleCols)==list
            cols=self.scaleCols
        if len(cols)!=0:
            self.means=np.mean(data[cols],axis=0)
            self.vars=np.var(data[cols],axis=0)
            data=self.transform(data)
            print("The means are: ",self.means)
            print("The variations are: ",self.vars)
        return data
    
    def transform(self, data):
        print("Try to normalize data to mu=0, sigma=1. ")
        print("The cols to screen are ",self.scaleCols)
        data[self.scaleCols]=(data[self.scaleCols]-self.means)/self.vars
        return data
    
    def __str__(self):
        return "normalizeScaler"