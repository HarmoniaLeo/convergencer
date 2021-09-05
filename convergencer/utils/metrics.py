import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_squared_log_error
from pytorch_tabnet.metrics import Metric

def mean_absolute_percentage_error(y_true,y_pred):
    loss = np.mean(np.abs((y_true - y_pred) / y_true),axis=-1)
    return loss

def mean_squared_percentage_error(y_true,y_pred):
    loss = np.mean(np.square(((y_true - y_pred) / y_true)),axis=-1)
    return loss

class tabnet_r2(Metric):
    def __init__(self):
        self._name = "r2"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return r2_score(y_true[:, 0], y_score[:, 0])

class tabnet_msle(Metric):
    def __init__(self):
        self._name = "msle"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return mean_squared_log_error(y_true[:, 0], y_score[:, 0])

class tabnet_mape(Metric):
    def __init__(self):
        self._name = "mape"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return mean_absolute_percentage_error(y_true[:, 0], y_score[:, 0])

class tabnet_mspe(Metric):
    def __init__(self):
        self._name = "mspe"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return mean_squared_percentage_error(y_true[:, 0], y_score[:, 0])

class metric:
    def evaluate(self,y_true,y_pred):
        return 0

    def maximum(self):
        return False

    def __str__(self):
        return "metric"

class r2(metric):
    def evaluate(self,y_true,y_pred):
        return r2_score(y_true,y_pred)
    
    def xgb(self):
        def score(preds,xgtrain):
            label = xgtrain.get_label()
            return "r2",r2_score(label,preds)
        return score
    
    def lgbm(self):
        def score(preds, train_data):
            label = train_data.get_label()
            return "r2",r2_score(label,preds),True
        return score

    def tabnet(self):
        return tabnet_r2
    
    def catboost(self):
        return "R2"
    
    def ann(self):
        return "mse"

    def maximum(self):
        return True

    def __str__(self):
        return "r2"

class mse(metric):
    def evaluate(self,y_true,y_pred):
        return mean_squared_error(y_true,y_pred)
    
    def xgb(self):
        def score(preds,xgtrain):
            label = xgtrain.get_label()
            return "mse",mean_squared_error(label,preds)
        return score

    def lgbm(self):
        return "mse"

    def tabnet(self):
        return "mse"

    def ann(self):
        return "mse"
    
    def catboost(self):
        return "RMSE"

    def __str__(self):
        return "mse"

class rmse(metric):
    def evaluate(self,y_true,y_pred):
        return np.sqrt(mean_squared_error(y_true,y_pred))
    
    def xgb(self):
        def score(preds,xgtrain):
            label = xgtrain.get_label()
            return "rmse",np.sqrt(mean_squared_error(label,preds))
        return score
    
    def lgbm(self):
        return "mse"
    
    def tabnet(self):
        return "mse"
    
    def ann(self):
        return "mse"
    
    def catboost(self):
        return "RMSE"

    def __str__(self):
        return "rmse"

class mae(metric):
    def evaluate(self,y_true,y_pred):
        return mean_absolute_error(y_true,y_pred)
    
    def xgb(self):
        def score(preds,xgtrain):
            label = xgtrain.get_label()
            return "mae",mean_absolute_error(label,preds)
        return score
    
    def lgbm(self):
        return "mae"

    def tabnet(self):
        return "mae"
    
    def ann(self):
        return "mae"
    
    def catboost(self):
        return "MAE"

    def __str__(self):
        return "mae"

class msle(metric):
    def evaluate(self,y_true,y_pred):
        return mean_squared_log_error(y_true,y_pred)
    
    def xgb(self):
        def score(preds,xgtrain):
            label = xgtrain.get_label()
            return "msle",mean_squared_log_error(label,preds)
        return score

    def lgbm(self):
        def score(preds, train_data):
            label = train_data.get_label()
            return "msle",mean_squared_log_error(label,preds),False
        return score

    def tabnet(self):
        return tabnet_msle
    
    def ann(self):
        return "msle"
    
    def catboost(self):
        return "MSLE"

    def __str__(self):
        return "msle"

class rmsle(metric):
    def evaluate(self,y_true,y_pred):
        return np.sqrt(mean_squared_log_error(y_true,y_pred))
    
    def xgb(self):
        def score(preds,xgtrain):
            label = xgtrain.get_label()
            return "rmsle",np.sqrt(mean_squared_log_error(label,preds))
        return score

    def lgbm(self):
        def score(preds, train_data):
            label = train_data.get_label()
            return "rmsle",np.sqrt(mean_squared_log_error(label,preds)),False
        return score

    def tabnet(self):
        return tabnet_msle
    
    def ann(self):
        return "msle"
    
    def catboost(self):
        return "MSLE"

    def __str__(self):
        return "rmsle"

class mape(metric):
    def evaluate(self,y_true,y_pred):
        return mean_absolute_percentage_error(y_true,y_pred)
    
    def xgb(self):
        def score(preds,xgtrain):
            label = xgtrain.get_label()
            return "mape",mean_absolute_percentage_error(label,preds)
        return score

    def lgbm(self):
        def score(preds, train_data):
            label = train_data.get_label()
            return "mape",mean_absolute_percentage_error(label,preds),False
        return score

    def tabnet(self):
        return tabnet_mape
    
    def ann(self):
        return "mape"
    
    def catboost(self):
        return "MAPE"
    
    def __str__(self):
        return "mape"

class mspe(metric):
    def evaluate(self, y_true, y_pred):
        return mean_squared_percentage_error(y_true,y_pred)
    
    def xgb(self):
        def score(preds,xgtrain):
            label = xgtrain.get_label()
            return "mspe",mean_squared_percentage_error(label,preds)
        return score

    def lgbm(self):
        def score(preds, train_data):
            label = train_data.get_label()
            return "mspe",mean_squared_percentage_error(label,preds),False
        return score

    def tabnet(self):
        return tabnet_mspe
    
    def ann(self):
        return "mape"
    
    def catboost(self):
        return "MAPE"
    
    def __str__(self):
        return "mspe"

class rmspe(metric):
    def evaluate(self, y_true, y_pred):
        return np.sqrt(mean_squared_percentage_error(y_true,y_pred))
    
    def xgb(self):
        def score(preds,xgtrain):
            label = xgtrain.get_label()
            return "rmspe",np.sqrt(mean_squared_percentage_error(label,preds))
        return score
    
    def lgbm(self):
        def score(preds, train_data):
            label = train_data.get_label()
            return "rmspe",np.sqrt(mean_squared_percentage_error(label,preds)),False
        return score

    def tabnet(self):
        return tabnet_mspe
    
    def ann(self):
        return "mape"
    
    def catboost(self):
        return "MAPE"

    def __str__(self):
        return "rmspe"

def getMetric(target):
    metrics=[r2(),mse(),rmse(),mae(),msle(),rmsle(),mape(),mspe(),rmspe()]
    if type(target)==str:
        for m in metrics:
            if target==str(m):
                return m
        raise Exception("Unsupported metric. ")
    else:
        return target