from processors.base import base
import numpy as np

class normalizeScaler(base):
    def __init__(self, data, y=None, parameters={},verbose=1):
        '''
        params:
            parameters:
            {
                "cols": cols to scan. Default=None for all number cols. 
            }
        '''
        self.verbose=verbose
        self.scaleCols=self.getParameter("cols",None,parameters)
        if self.scaleCols is None:
            ttn = data.select_dtypes(include=[np.number])
            self.scaleCols=ttn.columns
        if len(self.scaleCols)!=0:
            self.means=np.mean(data[self.scaleCols],axis=0)
            self.vars=np.var(data[self.scaleCols],axis=0)
                
    def transform(self, data, y=None):
        if self.verbose==1:
            print("\n-------------------------Try to normalize data to mu=0, sigma=1-------------------------")
            print("The cols to screen are ",self.scaleCols)
            print("The means are: \n",self.means)
            print("The variations are: \n",self.vars)
        data=data.copy()
        data[self.scaleCols]=(data[self.scaleCols]-self.means)/self.vars
        return super().transform(data, y=y)
    
    def __str__(self):
        return "normalizeScaler"