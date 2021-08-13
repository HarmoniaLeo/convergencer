from base import base
from sklearn.linear_model import Ridge,Lasso,ElasticNet,MultiTaskLasso,MultiTaskElasticNet
import numpy as np

class ridge(base):
    def __init__(self, parameters):
        self.parameters={"alpha":1.0}
        super().__init__(parameters=parameters)

    def getParameterRange(self):
        return {"alpha":(0.0,1.0)}

    def getModel(self, parameters):
        return Ridge(alpha=parameters["alpha"])
    
    def overfit(self,step):
        model=ridge(self.parameters)
        model.increaseParas(step,["alpha"])
        return model
    
    def underfit(self,step):
        model=ridge(self.parameters)
        model.reduceParas(step,["alpha"])
        return model
        
    def __str__(self):
        return "ridge"

class lasso(ridge):
    def getModel(self, parameters):
        return Lasso(alpha=np.exp(self.parameters["alpha"]))

    def overfit(self,step):
        model=Lasso(self.parameters)
        model.increaseParas(step,["alpha"])
        return model
    
    def underfit(self,step):
        model=Lasso(self.parameters)
        model.reduceParas(step,["alpha"])
        return model
    
    def __str__(self):
        return "lasso"

class elasticNet(base):
    def __init__(self, parameters):
        self.parameters={"alpha":1.0,"l1Rate":0.5}
        super().__init__(parameters=parameters)

    def getParameterRange(self):
        return {"alpha":(0.0,1.0),"l1Rate":(0.0,1.0)}

    def getModel(self, parameters):
        return ElasticNet(alpha=parameters["alpha"],l1_ratio=parameters["l1Rate"])
    
    def overfit(self,step):
        model=elasticNet(self.parameters)
        model.increaseParas(step,["alpha","l1Rate"])
        return model
    
    def underfit(self,step):
        model=elasticNet(self.parameters)
        model.reduceParas(step,["alpha","l1Rate"])
        return model
    
    def __str__(self):
        return "elasticNet"

class multiTaskLasso(lasso):
    def getModel(self, parameters):
        return MultiTaskLasso(alpha=parameters["alpha"])

    def overfit(self,step):
        model=multiTaskLasso(self.parameters)
        model.increaseParas(step,["alpha"])
        return model
    
    def underfit(self,step):
        model=multiTaskLasso(self.parameters)
        model.reduceParas(step,["alpha"])
        return model
    
    def __str__(self):
        return "multiTaskLasso"

class multiTaskElasticNet(elasticNet):
    def getModel(self, parameters):
        return MultiTaskElasticNet(alpha=parameters["alpha"],l1_ratio=parameters["l1Rate"])
    
    def overfit(self,step):
        model=multiTaskElasticNet(self.parameters)
        model.increaseParas(step,["alpha","l1Rate"])
        return model
    
    def underfit(self,step):
        model=multiTaskElasticNet(self.parameters)
        model.reduceParas(step,["alpha","l1Rate"])
        return model

    def __str__(self):
        return "multiTaskElasticNet"