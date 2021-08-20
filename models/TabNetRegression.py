from pytorch_tabnet.tab_model import TabNetRegressor
from torch.optim import optimizer
from torch.optim.adam import Adam
from base import base
import multiprocessing
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim

class TabNetRegression(base):
    def __init__(self,X,y,parameters={},maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        self.setParameter("n_d",8,parameters)
        self.setParameter("n_steps",3,parameters)
        self.setParameter("gamma",1.3,parameters)
        self.setParameter("n_independent",2,parameters)
        self.setParameter("n_shared",2,parameters)
        self.setParameter("learning_rate",0.1,parameters)
        self.setParameter("momentum",0.02,parameters)
        self.setParameter("lambda_sparse",1e-3,parameters)
        self.setParameter("batch_size",32,parameters)
        self.setParameter("iterations",500,parameters)
        self.setParameter("earlystop",5,parameters)
        super().__init__(X, y, parameters=parameters, maxEpoch=maxEpoch, checkPointPath=checkPointPath, checkPointFreq=checkPointFreq)

    def getParameterRange(self, X, y, parameters={}):
        self.setParameter("n_d",(int,"uni",8,64),parameters)
        self.setParameter("n_steps",(int,"uni",3,10),parameters)
        self.setParameter("gamma",(float,"uni",1.0,2.0),parameters)
        self.setParameter("n_independent",(int,"uni",1,5),parameters)
        self.setParameter("n_shared",(int,"uni",1,5),parameters)
        self.setParameter("learning_rate",(float,"exp",0.0,0.1),parameters)
        self.setParameter("momentum",(float,"exp",0.01,0.4),parameters)
        self.setParameter("lambda_sparse",(float,"exp",0.0,1.0),parameters)
        self.setParameter("batch_size",(object,128,256,512,1024,2048),parameters)
        self.setParameter("iterations",(object,100,200,500,1000,2000,5000),parameters)
        self.setParameter("earlystop",(object,5,10,15),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def getModel(self, X, y, parameters, modelPath):
        if modelPath is None:
            optimizer=torch.optim.Adam(dict(lr=parameters["learning_rate"]))
            return TabNetRegressor(
                n_d=parameters["n_d"],
                n_a=parameters["n_d"],
                n_steps=parameters["n_steps"],
                gamma=parameters["gamma"],
                n_independent=parameters["n_independent"],
                n_shared=parameters["n_shared"],
                momentum=parameters["momentum"],
                lambda_sparse=parameters["lambda_sparse"],
                optimizer_fn=optimizer,
                scheduler_fn=ReduceLROnPlateau(optimizer, mode='min'),
            )
        else:
            model=TabNetRegressor()
            return model.load_model(modelPath)

    def fitModel(self, X_train, y_train, X_test, y_test, model, parameters):
        parameters=parameters.copy()
        model.fit(X_train,y_train.values,eval_set=[(X_test, y_test.values)],patience=parameters["earlystop"],
        batch_size=parameters["batch_size"],max_epochs=parameters["iterations"],num_workers=multiprocessing.cpu_count())
    
    def saveModel(self, path):
        self.model.save_model(path)
        
    def __str__(self):
        return "ANNRegression"