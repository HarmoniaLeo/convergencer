import numpy as np
import pandas as pd
import logging
from utils.io import readData,readLabel
from utils.processing import removeNa,fillNa,normalization,filtering,scaling,varianceSelection,corrSelection,entropySelection,mutInfoSelection
from sklearn.decomposition import PCA
import pandas_profiling as pp

class regressor:

    def __init__(self,data,label=-1,id=None,labelId=None,timestamp=None,labelTimeStamp=None,
    delimiter=',',ifEDA=False,ifPreProcess=True,dropNaThr=0.15,fillNaStrg=None,
    fillNaValue={},fillNaK=5,fillNaKCols={},varianceSelectionCols=None,varianceSelectionThr=0.9,
    entropySelectionCols=None,entropySelectionThr=0.9,corrSelectionCols=None,corrSelectionThr=0.9,
    mutInfoSelectionCols=None,mutInfoSelectionThr=0.9,normalizeCols=None,normalizey=True,
    filterCols=None,scaleCols=None,ifPCA=True,PCAThr=0.9):
        
        data = readData(data,delimiter,id)
        data,label = readLabel(data,label,delimiter,labelId)        
            
        self.y = label
        self.X = data

        if ifEDA:
            pp.ProfileReport(data.join(label))

        if ifPreProcess:
            self.preProcess(dropNaThr=dropNaThr,fillNaStrg=fillNaStrg,fillNaK=fillNaK,
            fillNaValue=fillNaValue,fillNaKCols=fillNaKCols,
            varianceSelectionCols=varianceSelectionCols,varianceSelectionThr=varianceSelectionThr,
            entropySelectionCols=entropySelectionCols,entropySelectionThr=entropySelectionThr,
            corrSelectionCols=corrSelectionCols,corrSelectionThr=corrSelectionThr,
            mutInfoSelectionCols=mutInfoSelectionCols,mutInfoSelectionThr=mutInfoSelectionThr,
            normalizeCols=normalizeCols,normalizey=normalizey,filterCols=filterCols,
            scaleCols=scaleCols,ifPCA=ifPCA,PCAThr=PCAThr)

    
        

    def preProcess(self,dropNaThr=0.15,fillNaStrg=None,fillNaValue={},fillNaK=5,fillNaKCols={},
    varianceSelectionCols=None,varianceSelectionThr=0.9,entropySelectionCols=None,
    entropySelectionThr=0.9,corrSelectionCols=None,corrSelectionThr=0.9,mutInfoSelectionCols=None,
    mutInfoSelectionThr=0.9,normalizeCols=None,normalizey=True,filterCols=None,scaleCols=None,
    ifPCA=True,PCAThr=0.9):

        #nan processing
        self.naProcess(dropNaThr=dropNaThr,fillNaStrg=fillNaStrg,fillNaK=fillNaK,
        fillNaValue=fillNaValue,fillNaKCols=fillNaKCols)

        #filtering cols
        self.colSelection(varianceSelectionCols=varianceSelectionCols,varianceSelectionThr=varianceSelectionThr,
            entropySelectionCols=entropySelectionCols,entropySelectionThr=entropySelectionThr,
            corrSelectionCols=corrSelectionCols,corrSelectionThr=corrSelectionThr,
            mutInfoSelectionCols=mutInfoSelectionCols,mutInfoSelectionThr=mutInfoSelectionThr)

        #normalization
        self.normalize(normalizeCols=normalizeCols,normalizey=normalizey)

        #filtering rows
        self.filtering(filterCols=filterCols)

        #encoding
        self.encoding()
        
        #scale
        self.scaling(scaleCols=scaleCols)

        #PCA
        if ifPCA:
            self.PCA(PCAThr=PCAThr)
    
    def naProcess(self,dropNaThr=0.15,fillNaStrg=None,fillNaValue={},fillNaK=5,fillNaKCols={}):
        self.dropNaThr=dropNaThr
        self.fillNaStrg=fillNaStrg

        #remove nan y
        self.X = self.X.loc[~(self.y.iloc[:,0].isna())]
        self.y = self.y.dropna()

        #remove nan X with a threshold
        self.X = removeNa(self.X,dropNaThr)

        #fill nan X within the threshold
        self.X = fillNa(self.X,fillNaStrg,fillNaValue,fillNaK,fillNaKCols)
    
    def colSelection(self,varianceSelectionCols=None,varianceSelectionThr=0.9,
    entropySelectionCols=None,entropySelectionThr=0.9,corrSelectionCols=None,corrSelectionThr=0.9,
    mutInfoSelectionCols=None,mutInfoSelectionThr=0.9):
        self.X,self.varianceSelectionCols=varianceSelection(self.X,varianceSelectionCols,varianceSelectionThr)
        self.X,self.entropySelectionCols=entropySelection(self.X,entropySelectionCols,entropySelectionThr)
        self.X,self.corrSelectionCols=corrSelection(self.X,corrSelectionCols,corrSelectionThr)
        self.X,self.mutInfoSelectionCols=mutInfoSelection(self.X,mutInfoSelectionCols,mutInfoSelectionThr)
    
    def normalize(self,normalizeCols=None,normalizey=True):
        self.normalization=True
        self.normalizeCols=normalizeCols
        self.normalizey=normalizey
        self.X = normalization(self.X,normalizeCols)
        if normalizey:
            self.y = normalization(self.y,None)
    
    def filtering(self,filterCols=None):
        self.filterCols=filterCols
        self.X,self.filterMeans,self.filterStds = filtering(self.X,filterCols)
    
    def encoding(self):
        self.X = pd.get_dummies(self.X)
        print("Get dummies for category values. ")
    
    def scaling(self,scaleCols=None):
        self.scaleCols=scaleCols
        self.X,self.scaleMeans,self.scaleVars = scaling(self.X,scaleCols)
    
    def PCA(self,PCAThr=0.9):
        self.ifPCA = True
        self.pcaTransformer = PCA(PCAThr).fit(self.X)
        print("Proportion of variance of each dimension after PCA: ",self.pcaTransformer.explained_variance_ratio_)
        self.X = self.pcaTransformer.transform(self.X)

    def __getitem__(self,key):
        return self.X[key]
    
    def __setitem__(self,key,value):
        self.X[key]=value