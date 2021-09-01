from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
import tensorflow as tf
from convergencer.models import base
import tensorflow.keras.backend as K
from convergencer.processors import normalizeScaler,catToMean

def fr2(y_true, y_pred):
    st=K.sum(K.square(y_true-K.mean(y_true)))
    se=K.sum(K.square(y_true-y_pred))
    return 1-se/st

def fmspe(y_true, y_pred):
    return K.mean(K.square(((y_true - y_pred) / y_true)))

class ANNRegression(base):
    def initParameter(self, X, y, parameters):
        self.setParameter("layers",3,parameters)
        self.setParameter("hidden_count",64,parameters)
        self.setParameter("dropout",0.2,parameters)
        self.setParameter("learning_rate",0.1,parameters)
        self.setParameter("batch_size",32,parameters)
        self.setParameter("iterations",500,parameters)
        self.setParameter("earlystop",5,parameters)
        return super().initParameter(X, y, parameters)

    def getParameterRange(self, X, y, parameters={}):
        self.setParameter("layers",(int,"uni",3,12),parameters)
        self.setParameter("hidden_count",(object,16,32,64,128,256),parameters)
        self.setParameter("dropout",(float,"uni",0.2,0.8),parameters)
        self.setParameter("learning_rate",(float,"exp",0.001,0.1),parameters)
        self.setParameter("batch_size",(object,128,256,512,1024,2048),parameters)
        self.setParameter("iterations",(object,100,200,500,1000,2000,5000),parameters)
        self.setParameter("earlystop",(object,5,10,15),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            model = Sequential()
            model.add(Dense(X.shape[1],activation='relu'))
            for i in range(parameters["layers"]):
                model.add(Dense(parameters["hidden_count"],activation='relu'))
                model.add(Dropout(parameters["dropout"]))
            model.add(Dense(1))
            if metric=="r2":
                score=fr2
            elif metric=="mse":
                score="mse"
            elif metric=="mae":
                score="mae"
            elif metric=="msle":
                score="msle"
            elif metric=="mape":
                score="mape"
            elif metric=="mspe":
                score=fmspe
            model.compile(optimizer=Adam(parameters["learning_rate"]), loss="mse",metrics=score)
            return model
        else:
            return load_model(modelPath)

    def fitModel(self, X_train, y_train, X_test, y_test, model, parameters, metric):
        es = callbacks.EarlyStopping(monitor='val_loss', patience=parameters["earlystop"], verbose=0, restore_best_weights=True)
        rlp = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-5, mode='min', verbose=0)
        #trainSet=tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(parameters["batch_size"])
        #testSet=tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(1000).batch(parameters["batch_size"])
        '''
        model.fit(trainSet,
        validation_data=testSet,
        batch_size=parameters["batch_size"],
        epochs=parameters["iterations"],
        callbacks=[es, rlp])
        '''
        #X_train=tf.convert_to_tensor(X_train)
        #X_test=tf.convert_to_tensor(X_test)
        model.fit(X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=parameters["batch_size"],
        epochs=parameters["iterations"],
        callbacks=[es, rlp],verbose=0)
        return model
    
    def getProcessors(self,X,y):
        return [catToMean(X,verbose=0),normalizeScaler(X,verbose=0)]

    def inference(self, X):
        #X=tf.convert_to_tensor(X)
        return super().inference(X)
    
    def saveModel(self, path):
        self.model.save(path)
        
    def __str__(self):
        return "ANNRegression"