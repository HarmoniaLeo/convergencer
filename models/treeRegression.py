from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from base import base
import multiprocessing

class dtRegression(base):
    def __init__(self, parameters):
        self.parameters={"max_depth":10.0,"min_samples_split":0.0001,"min_samples_leaf":0.0001}
        super().__init__(parameters=parameters)

    def getParameterRange(self):
        return {"max_depth":(1.0,200.0),"min_samples_split":(1e-6,0.1),"min_samples_leaf":(1e-6,0.1)}

    def getModel(self, parameters):
        return DecisionTreeRegressor(
            max_depth=int(parameters["max_depth"]),
            min_samples_split=parameters["min_samples_split"],
            min_samples_leaf=parameters["min_samples_split"]
            )
    
    def overfit(self,step):
        model=dtRegression(self.parameters)
        model.increaseParas(step,["min_samples_split","min_samples_leaf"])
        model.reduceParas(step,["max_depth"])
        return model
    
    def underfit(self,step):
        model=dtRegression(self.parameters)
        model.reduceParas(step,["min_samples_split","min_samples_leaf"])
        model.increaseParas(step,["max_depth"])
        return model
        
    def __str__(self):
        return "dtRegression"

class rfRegression(base):
    def __init__(self, parameters):
        self.parameters={"num":100.0,"max_depth":10.0,"min_samples_split":0.0001,"min_samples_leaf":0.0001}
        super().__init__(parameters=parameters)

    def getParameterRange(self):
        return {"num":(10.0,200.0),"max_depth":(1.0,200.0),"min_samples_split":(1e-6,0.1),"min_samples_leaf":(1e-6,0.1)}

    def getModel(self, parameters):
        return RandomForestRegressor(
            n_estimators=int(parameters["num"]),
            n_jobs=multiprocessing.cpu_count(),
            max_depth=int(parameters["max_depth"]),
            min_samples_split=parameters["min_samples_split"],
            min_samples_leaf=parameters["min_samples_split"]
            )
        
    def __str__(self):
        return "rfRegression"