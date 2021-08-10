import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from scipy import stats
from scipy.spatial.distance import pdist,squareform
from sklearn import metrics

def removeNa(data,dropNaThr):
    print("Try to remove nan features with a threshold. ")
    percent = (data.isna().sum()/data.isna().count()).sort_values(ascending=False)
    print("The percentage of nan features in each col: ",percent)
    data = data.drop(percent[percent>dropNaThr].index,axis=1)
    print("Since the threshold is {0}, we drop cols: ".format(dropNaThr),percent[percent>dropNaThr].index)
    return data,percent[percent>dropNaThr].index
    

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
        imp=scaling(imp,None)
        imp_train= imp.loc[~df[col].isna()]
        imp_test = imp.loc[df[col].isna()]
        model = KNeighborsRegressor(n_neighbors=fillNaK)  # KNR Unsupervised Approach
        knr = model.fit(imp_train, df.loc[~df[col].isna(),col])
        df.loc[df[col].isna(), col] = knr.predict(imp_test)
    
    return df

def fillNa(data,fillNaStrg,fillNaValue,fillNaK,fillNaKCols):
    print("Try to fill nan values. ")
    for key in data.columns:
        nanCount = data[key].isna().sum()
        if nanCount==0:
            continue
        if (fillNaStrg is None) or (key not in fillNaStrg.keys()):
            continue
        else:
            if ((type(fillNaStrg)==str) and (fillNaStrg == "mode")) or ((type(fillNaStrg)==dict) and (fillNaStrg[key] == "mode")):
                num = data[key].mode()
                print("Fill {0} nans in col ".format(nanCount)+str(key)+" with mode "+str(num))
            elif ((type(fillNaStrg)==str) and (fillNaStrg == "mean")) or ((type(fillNaStrg)==dict) and (fillNaStrg[key] == "mean")):
                num = data[key].mean()
                print("Fill {0} nans in col ".format(nanCount)+str(key)+" with mean "+str(num))
            elif ((type(fillNaStrg)==str) and (fillNaStrg == "auto")) or ((type(fillNaStrg)==dict) and (fillNaStrg[key] == "auto")):
                if data[key].dtype == object:
                    num = data[key].mode()
                    print("Fill {0} nans in col ".format(nanCount)+str(key)+" with mode "+str(num))
                else:
                    num = data[key].mean()
                    print("Fill {0} nans in col ".format(nanCount)+str(key)+" with mean "+str(num))
            elif ((type(fillNaStrg)==str) and (fillNaStrg == "value")) or ((type(fillNaStrg)==dict) and (fillNaStrg[key] == "value")):
                num = fillNaValue[key]
                print("Fill {0} nans in col ".format(nanCount)+str(key)+" with "+str(num))
            else:
                raise Exception("Unsupported filling strategy. ")
            data[key] = data[key].fillna(num)
    return impute_knn(data,fillNaK,fillNaKCols)

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

def normalization(data,normalizeCols):
    if normalizeCols is None:
        columns=[key for key in data.columns if data[key].dtype!=object]
    elif type(normalizeCols)==list:
        columns=normalizeCols
    else:
        assert type(normalizeCols)==dict
        print("Try to normalize numerical values. ")
        for col in normalizeCols.keys():
            if normalizeCols[col]>0:
                for i in range(normalizeCols[col]):
                    data[col]=np.exmp1(data[col])
            elif normalizeCols[col]<0:
                for i in range(-normalizeCols[col]):
                    data[col]=np.log1p(data[col])
        return data,normalizeCols
    if len(columns)!=0:
        print("Try to normalize numerical values. ")
        normalizeCols={}
        for col in columns:
            normalizeCols[col]=0
            p=stats.kstest(data[col], 'norm', (data[col].mean(), data[col].std())).pvalue()
            i=0
            while p<0.05:
                if i>=2:
                    break
                print("Col "+str(col)+"failed to pass k-s test with p-value={0}. Try to normalize it. ".format(p))
                if data[col].skew()>0:
                    data[col]=np.log1p(data[col])
                    normalizeCols[col]-=1
                else:
                    data[col]=np.exmp1(data[col])
                    normalizeCols[col]+=1
                p=stats.kstest(data[col], 'norm', (data[col].mean(), data[col].std())).pvalue()
                i+=1
            else:
                print("Col "+str(col)+" passed k-s test with p-value={0}".format(p))
    return data,normalizeCols

def filtering(data,filterCols,means=None,stds=None):
    if filterCols is None:
        cols=[key for key in data.columns if data[key].dtype!=object]
    else:
        assert type(filterCols)==list
        cols=filterCols
    if len(cols)!=0:
        print("Try to filter data with normal distribution. ")
        print("The cols to screen are ",cols)
        if means is None:
            means=data[cols].mean()
        if stds is None:
            stds=data[cols].std()
        indexs=(data[cols]>(means+3*stds))|(data[cols]<(means-3*stds))
        index=indexs.any(axis=1)
        data=data.loc[~index]
        print("Dropped rows: ",np.argwhere(index==1))
    return data,means,stds

def scaling(data,scaleCols,means=None,vars=None):
    if scaleCols is None:
        cols=[key for key in data.columns if data[key].dtype!=object]
    else:
        assert type(scaleCols)==list
        cols=scaleCols
    if len(cols)!=0:
        print("Try to normalize data to mu=0, sigma=1. ")
        print("The cols to screen are ",cols)
        means=np.mean(data[cols],axis=0)
        vars=np.var(data[cols],axis=0)
        data[cols]=(data[cols]-means)/vars
        print("The means are: ",means)
        print("The variations are: ",vars)
    return data,means,vars

def varianceSelection(data,varianceSelectionCols,varianceSelectionThr):
    if varianceSelectionCols is None:
        cols=[key for key in data.columns if data[key].dtype!=object]
    else:
        assert type(varianceSelectionCols)==list
        cols=varianceSelectionCols
    if len(cols)!=0:
        numCols=[key for key in data.columns if data[key].dtype!=object]
        vars = pd.var(data[numCols],axis=0)
        vars = vars.sort_values()
        vars = vars.cumsum()
        varsSum = vars.sum()
        varsRate = vars/varsSum
        cols = [col for col in cols if varsRate[col]<(1-varianceSelectionThr)]
        data = data.drop(cols)
        print("Drop these cols since they have low variance: ",cols)
    return data,cols

def get_entropy(s):
    pe_value_array = s.unique()
    ent = 0.0
    for x_value in pe_value_array:
        p = float(s[s == x_value].shape[0]) / s.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent

def entropySelection(data,entropySelectionCols,entropySelectionThr):
    if entropySelectionCols is None:
        cols=[key for key in data.columns if data[key].dtype==object]
    else:
        assert type(entropySelectionCols)==list
        cols=entropySelectionCols
    if len(cols)!=0:
        numCols=[key for key in data.columns if data[key].dtype==object]
        vars = data[numCols].apply(get_entropy)
        vars = vars.sort_values()
        vars = vars.cumsum()
        varsSum = vars.sum()
        varsRate = vars/varsSum
        cols = [col for col in cols if varsRate[col]<(1-entropySelectionThr)]
        data = data.drop(cols)
        print("Drop these cols since they have low entropy: ",cols)
    return data,cols

def corrSelection(data,corrSelectionCols,corrSelectionThr):
    if corrSelectionCols is None:
        cols=[key for key in data.columns if data[key].dtype!=object]
    else:
        assert type(corrSelectionCols)==list
        cols=corrSelectionCols
    dropCols=[]
    if len(cols)!=0:
        numCols=[key for key in data.columns if data[key].dtype!=object]
        for col in numCols:
            if col in dropCols: continue
            for col2 in cols:
                if np.abs(np.correlate(data[col],data[col2]))>corrSelectionThr:
                    if col2 not in dropCols: dropCols.append(col2)
                    print("Drop col "+str(col2)+" since it highly correlated to "+str(col))
        data=data.drop(dropCols)
    return data,dropCols
    
def mutInfoSelection(data,mutInfoSelectionCols,mutInfoSelectionThr):
    if mutInfoSelectionCols is None:
        cols=[key for key in data.columns if data[key].dtype==object]
    else:
        assert type(mutInfoSelectionCols)==list
        cols=mutInfoSelectionCols
    dropCols=[]
    if len(cols)!=0:
        numCols=[key for key in data.columns if data[key].dtype==object]
        for col in numCols:
            if col in dropCols: continue
            for col2 in cols:
                if metrics.normalized_mutual_info_score(data[col],data[col2])>mutInfoSelectionThr:
                    if col2 not in dropCols: dropCols.append(col2)
                    print("Drop col "+str(col2)+" since it has high mutual infomation with "+str(col))
        data=data.drop(dropCols)
    return data,dropCols