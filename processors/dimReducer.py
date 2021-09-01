from processors.base import base
from sklearn.decomposition import PCA
import pandas as pd

class PCAReducer(base):
    def __init__(self, data, y=None, parameters={},verbose=1):
        '''
        parameters:
            {
                "threshold": How much amount of variation you want to maintain. Default=0.9
            }
        '''
        self.verbose=verbose
        threshold=self.getParameter("threshold",0.9,parameters)
        self.transformer=PCA(threshold)
        self.transformer.fit(data)
    
    def transform(self, data, y=None):
        print("\n-------------------------Using PCA to transform data-------------------------")
        res=self.transformer.transform(data)
        return super().transform(pd.DataFrame(res,index=data.index), y=y)
    
    def __str__(self):
        return "PCA"