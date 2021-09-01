from convergencer.processors.base import base
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import inv_boxcox
from convergencer.utils.processing import normalTest

class normalization(base):
    def __init__(self, data, y=None, parameters={},verbose=1):
        '''
        parameter:
            {
                "cols": cols to be turned to normal distribution. Default=None for all number cols. 
            }
        '''
        self.verbose=verbose
        cols=self.getParameter("cols",None,parameters)
        self.l=None
        self.cols=[]
        if cols is None:
            ttn = data.select_dtypes(include=[np.number])
            cols=ttn.columns
        _,cols=normalTest(data,cols)
        for col in cols:
            t, l = stats.boxcox(data[col]+1, lmbda=None, alpha=None)
            target=pd.Series(t,index=data.index)
            if normalTest(target,threshold=0.03):
                self.cols.append(col)
        if not (y is None):
            if not normalTest(y):
                t, l = stats.boxcox(y+1, lmbda=None, alpha=None)
                target=pd.Series(t,index=y.index)
                if normalTest(target,threshold=0.03):
                    self.l=l

    def transform(self, data, y=None):
        '''
        params:
            data: the data to transform
        return:
            data whose number cols are turned to normal distribution
        '''
        print("\n-------------------------Try to normalize cols-------------------------")
        data=data.copy()
        for col in self.cols:
            if self.verbose==1:
                print("Try to normalize col: ",col)
            t, l = stats.boxcox(data[col]+1, lmbda=None, alpha=None)
            target=pd.Series(t,index=data.index)
            if self.verbose==1:
                print("Skew is {0} now. ".format(target.skew()))
            data[col]=target
        if not (y is None):
            if not (self.l is None):
                if self.verbose==1:
                    print("Try to normalize label")
                t, l = stats.boxcox(y+1, lmbda=None, alpha=None)
                y=pd.Series(t,index=y.index)
                if self.verbose==1:
                    print("Skew is {0} now. ".format(y.skew()))
        return super().transform(data, y=y)
    
    def reTransform(self,data, y=None):
        '''
        params:
            target: label to re-transform
        return:
            original label
        '''
        if not (y is None):
            if not (self.l is None):
                if self.verbose==1:
                    print("\n-------------------------Try to de-normalize label-------------------------")
                t=inv_boxcox(y, self.l)-1
                y=pd.Series(t,index=y.index)
                if self.verbose==1:
                    print("Skew is {0} now. ".format(y.skew()))
        return super().reTransform(data, y=y)

    def __str__(self):
        return "normalization"