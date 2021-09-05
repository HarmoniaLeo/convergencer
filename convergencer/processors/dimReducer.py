from convergencer.processors.base import base
from sklearn.decomposition import PCA
import pandas as pd

class PCAReducer(base):
    def initialize(self, parameters={},verbose=1):
        '''
        parameters:
            {
                "threshold": How much amount of variation you want to maintain. Default=0.9
            }
        '''
        self.verbose=verbose
        threshold=self._getParameter("threshold",0.9,parameters)
        self.transformer=PCA(threshold)
        return self

    def fit(self, data, y=None):
        self.transformer.fit(data)
        return self
    
    def transform(self, data, y=None):
        if self.verbose==1:
            print("\n-------------------------Using PCA to transform data-------------------------")
        res=self.transformer.transform(data)
        return super().transform(pd.DataFrame(res,index=data.index), y=y)
    
    def __str__(self):
        return "PCA"