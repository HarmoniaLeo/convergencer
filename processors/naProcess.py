from base import base
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

def impute_knn(df,fillNaK,fillNaKCols):

    cols_nan = df.columns[df.isna().any()].tolist()         # columns w/ nan 

    for col in cols_nan:
        nanCount = df[col].isna().sum()
        print("Fill {0} nans in col ".format(nanCount)+str(col)+" with default {0}-nn strategy. ".format(fillNaK))
        #imp_test = df[df[col].isna()]   # indicies which have missing data will become our test set
        if col in fillNaKCols.keys():
            imp=df[fillNaKCols[col]]
        else:
            imp=df
        imp=imp.dropna()
        imp=pd.get_dummies(imp)
        means=np.mean(imp,axis=0)
        vars=np.var(imp,axis=0)
        imp=(imp-means)/vars
        imp_train= imp.loc[~df[col].isna()]
        imp_test = imp.loc[df[col].isna()]
        model = KNeighborsRegressor(n_neighbors=fillNaK)  # KNR Unsupervised Approach
        knr = model.fit(imp_train, df.loc[~df[col].isna(),col])
        df.loc[df[col].isna(), col] = knr.predict(imp_test)
    
    return df

class fillNa(base):
    def __init__(self, parameters):
        self.fillNaStrg=self.getParameter("strategy",None,parameters)
        self.fillNaValue=self.getParameter("values",None,parameters)
        self.fillNaK=self.getParameter("k",5,parameters)
        self.fillNaCols=self.getParameter("knn cols",{},parameters)
    
    def transform(self,data):
        print("Try to fill nan values. ")
        for key in data.columns:
            nanCount = data[key].isna().sum()
            if nanCount==0:
                continue
            if (self.fillNaStrg is None) or (key not in self.fillNaStrg.keys()):
                continue
            else:
                if ((type(self.fillNaStrg)==str) and (self.fillNaStrg == "mode")) or ((type(self.fillNaStrg)==dict) and (self.fillNaStrg[key] == "mode")):
                    num = data[key].mode()
                    print("Fill {0} nans in col ".format(nanCount)+str(key)+" with mode "+str(num))
                elif ((type(self.fillNaStrg)==str) and (self.fillNaStrg == "mean")) or ((type(self.fillNaStrg)==dict) and (self.fillNaStrg[key] == "mean")):
                    num = data[key].mean()
                    print("Fill {0} nans in col ".format(nanCount)+str(key)+" with mean "+str(num))
                elif ((type(self.fillNaStrg)==str) and (self.fillNaStrg == "auto")) or ((type(self.fillNaStrg)==dict) and (self.fillNaStrg[key] == "auto")):
                    if data[key].dtype == object:
                        num = data[key].mode()
                        print("Fill {0} nans in col ".format(nanCount)+str(key)+" with mode "+str(num))
                    else:
                        num = data[key].mean()
                        print("Fill {0} nans in col ".format(nanCount)+str(key)+" with mean "+str(num))
                elif ((type(self.fillNaStrg)==str) and (self.fillNaStrg == "value")) or ((type(self.fillNaStrg)==dict) and (self.fillNaStrg[key] == "value")):
                    num = self.fillNaValue[key]
                    print("Fill {0} nans in col ".format(nanCount)+str(key)+" with "+str(num))
                else:
                    raise Exception("Unsupported filling strategy. ")
                data[key] = data[key].fillna(num)
        return impute_knn(data,self.fillNaK,self.fillNaKCols)
    
    def __str__(self):
        return "fillNa"