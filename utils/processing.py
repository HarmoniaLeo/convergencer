import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

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

'''
def encoding(data,orderEncode):
    if not(orderEncode is None):
        print("Try to encode category values in order. ")
        if type(orderEncode)==list:
            le = preprocessing.LabelEncoder()
            for col in [col for col in orderEncode]:
                data[col]=le.fit_transform(data[col].astype(str))
                print("Encode col "+str(col)+" in an order. ")
        elif type(orderEncode)==dict:
            for col in [col for col in orderEncode.keys()]:
                for value in np.unique(data[col]):
                    data[col][data[col]==value]=orderEncode[col][value]
                    print("In col "+str(col)+", replace "+str(value)+" with "+str(orderEncode[col][value]))
        else:
            raise Exception("Unsupported orderEncode. ")
    return data
'''

def get_entropy(s):
    pe_value_array = s.unique()
    ent = 0.0
    for x_value in pe_value_array:
        p = float(s[s == x_value].shape[0]) / s.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent