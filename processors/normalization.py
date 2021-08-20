from base import base
import numpy as np
from scipy import stats

class normalization(base):
    def __init__(self, parameters):
        self.normalizeCols=self.getParameter("cols",None,parameters)
        self.normalizeDict={}

    def fit(self,data):
        if self.normalizeCols is None:
            ttn = data.select_dtypes(include=[np.number])
            columns=ttn.columns
        elif type(self.normalizeCols)==list:
            columns=self.normalizeCols
        if len(columns)!=0:
            print("Try to normalize numerical values. ")
            for col in columns:
                self.normalizeDict[col]=0
                p=stats.kstest(data[col], 'norm', (data[col].mean(), data[col].std())).pvalue()
                i=0
                while p<0.05:
                    if i>=2:
                        break
                    print("Col "+str(col)+"failed to pass k-s test with p-value={0}. Try to normalize it. ".format(p))
                    if data[col].skew()>0:
                        data[col]=np.log1p(data[col])
                        self.normalizeDict[col]-=1
                    else:
                        data[col]=np.exmp1(data[col])
                        self.normalizeDIct[col]+=1
                    p=stats.kstest(data[col], 'norm', (data[col].mean(), data[col].std())).pvalue()
                    i+=1
                else:
                    print("Col "+str(col)+" passed k-s test with p-value={0}".format(p))
        return data
    
    def transform(self, data):
        print("Try to normalize numerical values. ")
        for col in self.normalizeDict.keys():
            if self.normalizeDict[col]>0:
                for i in range(self.normalizeDict[col]):
                    data[col]=np.exmp1(data[col])
            elif self.normalizeDict[col]<0:
                for i in range(-self.normalizeDict[col]):
                    data[col]=np.log1p(data[col])
        return data
    
    def __str__(self):
        return "normalization"