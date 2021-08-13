from base import base
from sklearn.decomposition import PCA
import pandas as pd

class PCA(base):
    def __init__(self, parameters):
        threshold=self.getParameter("threshold",0.9,parameters)
        self.transformer=PCA(threshold)
    
    def fit(self,data):
        self.transformer.fit(data)
        return pd.DataFrame(self.transformer.transform(data))
    
    def transform(self, data):
        return pd.DataFrame(self.transformer.transform(data))
    
    def __str__(self):
        return "PCA"