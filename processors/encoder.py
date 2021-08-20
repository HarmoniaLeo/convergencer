from base import base
import pandas as pd
from sklearn import preprocessing
import numpy as np

class dummies(base):
    def transform(self, data):
        ttn = data.select_dtypes(exclude=[np.number])
        print("Get dummies for category cols: ",ttn.columns)
        return pd.get_dummies(data)
    
    def __str__(self):
        return "dummies"

class intEncoder(base):
    def transform(self,data):
        le = preprocessing.LabelEncoder()
        ttn = data.select_dtypes(exclude=[np.number])
        ttn=le.fit_transform(ttn.astype(str))
        print(ttn.columns," has been encoded to int codes. ")
        data[ttn.columns]=ttn
        return data