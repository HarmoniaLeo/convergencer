from convergencer.models.base import base
from convergencer.models.linearRegression import linear,ridge,lasso,elasticNet
from convergencer.models.SVR import SVMRegression,NuSVMRegression
from convergencer.models.treeRegression import dtRegression,rfRegression,gbRegression
from convergencer.models.xgb import xgbRegression,xgbRegression_dart
from convergencer.models.lgbm import lgbmRegression,lgbmRegression_dart,lgbmRegression_goss
from convergencer.models.catboost import catBoostRegression
from convergencer.models.ANNRegression import ANNRegression
from convergencer.models.TabNet import TabNetRegression
from convergencer.models.assemble import stackingPredict,baggingAnalyse

def getModels():
    models=[linear(),ridge(),lasso(),elasticNet(),SVMRegression(),NuSVMRegression(),dtRegression(),rfRegression(),gbRegression(),xgbRegression(),lgbmRegression(),lgbmRegression_goss(),
    catBoostRegression(),TabNetRegression()]
    modelDict={}
    for model in models:
        modelDict[str(model)]=model
    return modelDict