from convergencer.processors.base import base
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import inv_boxcox
from convergencer.utils.processing import normalTest

class normalization(base):
    def initialize(self, parameters={},verbose=1):
        '''
        parameter:
            {
                "cols": cols to be turned to normal distribution. Default=None for all number cols. 
            }
        '''
        self.verbose=verbose
        self.processCols=self._getParameter("cols",None,parameters)
        self.l=None
        return self

    def fit(self, data, y=None):
        if self.processCols is None:
            ttn = data.select_dtypes(include=[np.number])
            cols=ttn.columns
        else:
            cols=self.processCols
        _,cols=normalTest(data,cols)
        self.cols=[]
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
        return self

    def transform(self, data, y=None):
        '''
        params:
            data: the data to transform
        return:
            data whose number cols are turned to normal distribution
        '''
        if self.verbose==1:
            print("\n-------------------------Try to normalize cols-------------------------")
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

class simpleNormalization(base):
    def initialize(self, parameters={},verbose=1):
        '''
        parameter:
            {
                "threshold": col with skew>threshold will be transformed with box-cox. Default=0.5
            }
        '''
        self.verbose=verbose
        self.threshold=self._getParameter("threshold",0.5,parameters)
        self.normalizey=False
        return self

    def fit(self, data, y=None):
        ttn = data.select_dtypes(include=[np.number])
        cols=ttn.columns
        self.cols = [col for col in cols if data[col].skew()>self.threshold]
        if not(y is None):
            self.normalizey=(y.skew()>self.threshold)
        return self

    def transform(self, data, y=None):
        '''
        params:
            data: the data to transform
        return:
            data whose number cols are turned to normal distribution
        '''
        data=data.copy()
        if self.verbose==1:
            print("\n-------------------------Try to normalize cols-------------------------")
        for col in self.cols:
            if self.verbose==1:
                print("Try to normalize col: ",col)
            t, l = stats.boxcox(data[col]+1, lmbda=None, alpha=None)
            target=pd.Series(t,index=data.index)
            if self.verbose==1:
                print("Skew is {0} now. ".format(target.skew()))
            data[col]=target
        if not (y is None):
            y=y.copy()
            if self.normalizey:
                if self.verbose==1:
                    print("Try to normalize label")
                t, self.l = stats.boxcox(y+1, lmbda=None, alpha=None)
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
            y=y.copy()
            if self.normalizey:
                if self.verbose==1:
                    print("\n-------------------------Try to de-normalize label-------------------------")
                t=inv_boxcox(y, self.l)-1
                y=pd.Series(t,index=y.index)
                if self.verbose==1:
                    print("Skew is {0} now. ".format(y.skew()))
        return super().reTransform(data, y=y)

    def __str__(self):
        return "simpleNormalization"