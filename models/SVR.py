from sklearn.svm import SVR
from base import base

class SVMRegression(base):
    def __init__(self, parameters):
        self.parameters={"C":1.0,"kernel":2.0,"degree":3.0}
        super().__init__(parameters=parameters)
    
    def getParameterRange(self):
        return {"C":(0.0,20.0),"kernel":(0.0,5.0),"degree":(2.0,9.0-1e-6)}

    def getModel(self, parameters):
        kernels=["linear","poly","rbf","sigmoid","precomputed"]
        return SVR(kernel=kernels[int(parameters["kernel"])],
        degree=int(parameters["degree"]),C=parameters["C"])
    
    def overfit(self,step):
        model=SVMRegression(self.parameters)
        model.increaseParas(step,["C"])
        model.reduceParas(step,["degree"])
        return model
    
    def underfit(self,step):
        model=SVMRegression(self.parameters)
        model.reduceParas(step,["C"])
        model.increaseParas(step,["degree"])
        return model
        
    def __str__(self):
        return "SVRegression"