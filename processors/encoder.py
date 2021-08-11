from base import base
import pandas as pd

class dummies(base):
    def transform(self, data):
        return pd.get_dummies(data)
    
    def __str__(self):
        return "dummies"