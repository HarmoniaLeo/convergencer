import numpy as np
import pandas as pd
import logging
from utils.io import readData,readLabel,saveModel,readModel
from utils.processing import removeNa,fillNa,normalization,filtering,scaling,varianceSelection,corrSelection,entropySelection,mutInfoSelection
from utils.optimizing import finetune
from models import bagging
from sklearn.decomposition import PCA
import pandas_profiling as pp

class regressPipeline:

    def __init__(self,data,label=-1,id=None,labelId=None,timestamp=None,labelTimeStamp=None,
    delimiter=',',ifEDA=False,ifPreProcess=True,dropNaThr=0.15,fillNaStrg=None,
    fillNaValue={},fillNaK=5,fillNaKCols={},customProcessFunction=lambda x:x,
    varianceSelectionCols=None,varianceSelectionThr=0.9,
    entropySelectionCols=None,entropySelectionThr=0.9,
    corrSelectionCols=None,corrSelectionThr=0.9,
    mutInfoSelectionCols=None,mutInfoSelectionThr=0.9,
    normalizeCols=None,normalizey=True,filterCols=None,scaleCols=None,ifPCA=True,PCAThr=0.9,
    modelPath=None,checkPointPath=None,modelTypes=[],ifTrain=True,accThr=0.98,
    maxEpochFinetuning=100,maxBaggingModelNum=5,maxEpochBagging=100):

        data = readData(data,delimiter,id)
        data,label = readLabel(data,label,delimiter,labelId)        
            
        self.y = label.iloc[:,0]
        self.X = data

        if ifEDA:
            pp.ProfileReport(data.join(label))
    
        self.ifNaProcess=False
        self.ifCustomProcess=False
        self.ifColSelection=False
        self.ifFiltering=False
        self.ifEncoding=False
        self.ifScaling=False
        self.ifPCA=False

        self.model=None
        
        self.dropCols=[]

        if ifPreProcess:
            self.preProcess(dropNaThr=dropNaThr,fillNaStrg=fillNaStrg,fillNaK=fillNaK,
            fillNaValue=fillNaValue,fillNaKCols=fillNaKCols,customProcessFunction=customProcessFunction,
            varianceSelectionCols=varianceSelectionCols,varianceSelectionThr=varianceSelectionThr,
            entropySelectionCols=entropySelectionCols,entropySelectionThr=entropySelectionThr,
            corrSelectionCols=corrSelectionCols,corrSelectionThr=corrSelectionThr,
            mutInfoSelectionCols=mutInfoSelectionCols,mutInfoSelectionThr=mutInfoSelectionThr,
            normalizeCols=normalizeCols,normalizey=normalizey,filterCols=filterCols,
            scaleCols=scaleCols,ifPCA=ifPCA,PCAThr=PCAThr)

        if ifTrain:
            self.train(modelPath=modelPath,checkPointPath=None,modelTypes=modelTypes,accThr=accThr,maxEpochFinetuning=maxEpochFinetuning,maxBaggingModelNum=maxBaggingModelNum,maxEpochBagging=maxEpochBagging)
        

    def preProcess(self,dropNaThr=0.15,fillNaStrg=None,fillNaValue={},fillNaK=5,fillNaKCols={},
    customProcessFunction=lambda x:x,varianceSelectionCols=None,varianceSelectionThr=0.9,
    entropySelectionCols=None,entropySelectionThr=0.9,corrSelectionCols=None,corrSelectionThr=0.9,
    mutInfoSelectionCols=None,mutInfoSelectionThr=0.9,normalizeCols=None,normalizey=True,
    filterCols=None,scaleCols=None,ifPCA=True,PCAThr=0.9):

        #nan processing
        self.naProcess(dropNaThr=dropNaThr,fillNaStrg=fillNaStrg,fillNaK=fillNaK,
        fillNaValue=fillNaValue,fillNaKCols=fillNaKCols)

        self.customProcess(customProcessFunction=customProcessFunction)

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
        self.ifNaProcess=True
        self.dropNaThr=dropNaThr
        self.fillNaStrg=fillNaStrg
        self.fillNaValue=fillNaValue
        self.fillNaK=fillNaK
        self.fillNaKCols=fillNaKCols

        #remove nan y
        self.X = self.X.loc[~(self.y.isna())]
        self.y = self.y.dropna()

        #remove nan X with a threshold
        self.X,dropNaCols = removeNa(self.X,dropNaThr)
        self.dropCols=np.union1d(self.dropCols,dropNaCols)

        #fill nan X within the threshold
        self.X = fillNa(self.X,fillNaStrg,fillNaValue,fillNaK,fillNaKCols)
    
    def customProcess(self,customProcessFunction=lambda x:x):
        self.ifCustomProcess=True
        self.func = customProcessFunction
        self.X = customProcessFunction(self.X)
    
    def colSelection(self,varianceSelectionCols=None,varianceSelectionThr=0.9,
    entropySelectionCols=None,entropySelectionThr=0.9,corrSelectionCols=None,corrSelectionThr=0.9,
    mutInfoSelectionCols=None,mutInfoSelectionThr=0.9):
        self.X,varianceSelectionCols=varianceSelection(self.X,varianceSelectionCols,varianceSelectionThr)
        self.X,entropySelectionCols=entropySelection(self.X,entropySelectionCols,entropySelectionThr)
        self.X,corrSelectionCols=corrSelection(self.X,corrSelectionCols,corrSelectionThr)
        self.X,mutInfoSelectionCols=mutInfoSelection(self.X,mutInfoSelectionCols,mutInfoSelectionThr)
        self.dropCols = np.union1d(self.dropCols,entropySelectionCols)
        self.dropCols = np.union1d(self.dropCols,corrSelectionCols)
        self.dropCols = np.union1d(self.dropCols,mutInfoSelectionCols)

    def normalize(self,normalizeCols=None):
        self.ifNormalization=True
        self.X,self.normalizeCols = normalization(self.X,normalizeCols)
    
    def filtering(self,filterCols=None):
        self.ifFiltering=True
        self.filterCols=filterCols
        self.X,self.filterMeans,self.filterStds = filtering(self.X,filterCols)
    
    def encoding(self):
        self.ifEncoding=True
        self.X = pd.get_dummies(self.X)
        print("Get dummies for category values. ")
    
    def scaling(self,scaleCols=None):
        self.ifScaling=True
        self.scaleCols=scaleCols
        self.X,self.scaleMeans,self.scaleVars = scaling(self.X,scaleCols)
    
    def PCA(self,PCAThr=0.9):
        self.ifPCA = True
        self.pcaTransformer = PCA(PCAThr).fit(self.X)
        print("Proportion of variance of each dimension after PCA: ",self.pcaTransformer.explained_variance_ratio_)
        self.X = self.pcaTransformer.transform(self.X)
        print("PCA applied. ")
    
    def train(self,modelPath=None,checkPointPath=None,modelTypes=[],accThr=0.98,maxEpochFinetuning=100,maxBaggingModelNum=5,maxEpochBagging=100):
        if modelPath is None:
            self.models=[

            ]
            print("Start Training. ")
            bestModels,bestTrainAcc,bestTestAccs=finetune(self.X,self.y,self.models,maxEpochFinetuning,accThr,maxBaggingModelNum,checkPointPath)
            self.model=bestModels[0]
            bestTrainAcc=bestTrainAcc[0]
            bestTestAcc=bestTestAccs[0]
            baggingModel=bagging(bestModels)
            print("Start Bagging. ")
            baggingTrainAcc,baggingTestAcc=bagging.train(self.X,self.y,maxEpochBagging,checkPointPath)
            if baggingTestAcc>bestTestAcc:
                self.model=baggingModel
                bestTrainAcc=baggingTrainAcc
                bestTestAcc=baggingTestAcc
            print("Found the best model: ",self.model)
            print("Accuracy on train set: {0}".format(bestTrainAcc))
            print("Accuracy on test set: {0}".format(bestTestAcc))
        else:
            self.model=readModel(modelPath)

    def inference(self,data,id=None,timestamp=None,delimiter=','):
        assert not(self.model is None)
        data = readData(data,delimiter,id)
        if self.ifNaProcess:
            data = fillNa(data,self.fillNaStrg,self.fillNaValue,self.fillNaK,self.fillNaKCols)
        if self.ifCustomProcess:
            data = self.func(data)
        if len(self.dropCols)>0:
            print("Drop these cols: ",self.dropCols)
            data = data.drop(self.dropCols)
        if self.ifNormalization:
            data,self.normalizeCols = normalization(data,self.normalizeCols)
        if self.ifFiltering:
            data,self.filterMeans,self.filterStds = filtering(data,self.filterCols,self.filterMeans,self.filterStds)
        if self.ifEncoding:
            data = pd.get_dummies(data)
            print("Get dummies for category values. ")
        if self.ifScaling:
            data,self.scaleMeans,self.scaleVars = scaling(data,self.scaleCols,self.scaleMeans,self.scaleVars)
        if self.ifPCA:
            data = self.pcaTransformer.transform(data)
            print("PCA applied. ")
        return self.model.inference(data)

    def saveModel(self,path=""):
        assert not(self.model is None)
        saveModel(self.model.save(),path)