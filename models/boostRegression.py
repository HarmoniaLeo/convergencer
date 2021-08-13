from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from base import base

class gbRegression(base):
    def __init__(self, parameters):
        self.parameters={"subsample":0.5,"min_samples_split":0.0001,"min_samples_leaf":0.0001,
        "max_depth":10.0,}
        super().__init__(parameters=parameters)

    def getParameterRange(self):
        return {"subsample":(0.5,0.8),"min_samples_split":(1e-6,0.1),"min_samples_leaf":(1e-6,0.1),
        "max_depth":(1.0,200.0)}

    def getModel(self, parameters):
        return GradientBoostingRegressor(
            subsample=parameters["subsample"],
            min_samples_split=parameters["min_samples_split"],
            max_depth=int(parameters["max_depth"])
            )
        
    def __str__(self):
        return "gbRegressor"

class xgbRegression(base):
    