import numpy as np
import pandas as pd
from scipy import stats

def normalTest(data,cols=None,threshold=0.05):
    if type(data)==pd.Series:
        p=stats.kstest(data, 'norm', (data.mean(), data.std()))[1]
        if p<threshold:
            #print("The col failed to pass k-s test with p-value={0}. ".format(p))
            return False
        else:
            #print("The col passed k-s test with p-value={0}. ".format(p))
            return True
    else:
        passCols=[]
        failedCols=[]
        if cols is None:
            ttn = data.select_dtypes(include=[np.number])
            cols=ttn.columns
        for col in cols:
            p=stats.kstest(data[col], 'norm', (data[col].mean(), data[col].std()))[1]
            if p<threshold:
                #print("Col "+str(col)+" failed to pass k-s test with p-value={0}. ".format(p))
                failedCols.append(col)
            else:
                #print("Col "+str(col)+" passed k-s test with p-value={0}. ".format(p))
                passCols.append(col)
        return passCols,failedCols