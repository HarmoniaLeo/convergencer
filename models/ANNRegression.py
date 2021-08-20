from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from base import base

class ANNRegression(base):
    def __init__(self,X,y,parameters={},maxEpoch=1000,checkPointPath=None,checkPointFreq=50):
        self.setParameter("layers",3,parameters)
        self.setParameter("hidden_count",64,parameters)
        self.setParameter("dropout",0.2,parameters)
        self.setParameter("learning_rate",0.1,parameters)
        self.setParameter("batch_size",32,parameters)
        self.setParameter("iterations",500,parameters)
        self.setParameter("earlystop",5,parameters)
        super().__init__(X, y, parameters=parameters, maxEpoch=maxEpoch, checkPointPath=checkPointPath, checkPointFreq=checkPointFreq)

    def getParameterRange(self, X, y, parameters={}):
        self.setParameter("layers",(int,"uni",3,12),parameters)
        self.setParameter("hidden_count",(object,16,32,64,128,256),parameters)
        self.setParameter("dropout",(float,"uni",0.2,0.8),parameters)
        self.setParameter("learning_rate",(float,"exp",0.0,0.1),parameters)
        self.setParameter("batch_size",(object,128,256,512,1024,2048),parameters)
        self.setParameter("iterations",(object,100,200,500,1000,2000,5000),parameters)
        self.setParameter("earlystop",(object,5,10,15),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def getModel(self, X, y, parameters, modelPath):
        if modelPath is None:
            model = Sequential()
            model.add(Dense(X.shape[1],activation='relu'))
            for i in range(parameters["layers"]):
                model.add(Dense(parameters["hidden_count"],activation='relu'))
                model.add(Dropout(parameters["dropout"]))
            model.add(Dense(1))
            model.compile(optimizer=Adam(parameters["learning_rate"]), loss="mse")
            return model
        else:
            return load_model(modelPath)

    def fitModel(self, X_train, y_train, X_test, y_test, model, parameters):
        es = callbacks.EarlyStopping(monitor='val_loss', patience=parameters["earlystop"], verbose=1, restore_best_weights=True)
        rlp = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-10, mode='min', verbose=1)
        model.fit(X_train, y_train.values,
        validation_data=(X_test,y_test.values),
        batch_size=parameters["batch_size"],
        epochs=parameters["iterations"],
        callbacks=[es, rlp])
    
    def saveModel(self, path):
        self.model.save(path)
        
    def __str__(self):
        return "ANNRegression"