from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from convergencer.models import base
from convergencer.processors import normalizeScaler,catToMean
from sklearn.model_selection import train_test_split

class ANNRegression(base):
    def _initParameter(self, X, y, parameters):
        self._setParameter("layers",3,parameters)
        self._setParameter("hidden_count",64,parameters)
        self._setParameter("dropout",0.2,parameters)
        self._setParameter("learning_rate",0.1,parameters)
        self._setParameter("batch_size",32,parameters)
        self._setParameter("iterations",500,parameters)
        self._setParameter("earlystop",5,parameters)
        return super().initParameter(X, y, parameters)

    def _getParameterRange(self, X, y, parameters={}):
        self._setParameter("layers",(int,"uni",3,12),parameters)
        self._setParameter("hidden_count",(object,16,32,64,128,256),parameters)
        self._setParameter("dropout",(float,"uni",0.2,0.8),parameters)
        self._setParameter("learning_rate",(float,"exp",0.001,0.1),parameters)
        self._setParameter("batch_size",(object,128,256,512,1024,2048),parameters)
        self._setParameter("iterations",(object,100,200,500,1000,2000,5000),parameters)
        self._setParameter("earlystop",(object,5,10,15),parameters)
        return super().getParameterRange(X, y, parameters=parameters)

    def _getModel(self, X, y, parameters, modelPath,metric):
        if modelPath is None:
            model = Sequential()
            model.add(Dense(X.shape[1],activation='relu'))
            for i in range(parameters["layers"]):
                model.add(Dense(parameters["hidden_count"],activation='relu'))
                model.add(Dropout(parameters["dropout"]))
            model.add(Dense(1))
            model.compile(optimizer=Adam(parameters["learning_rate"]), loss="mse",metrics=metric.ann())
            return model
        else:
            return load_model(modelPath)

    def _fitModel(self, X,y, model, parameters, metric):
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
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
        #X_train=tf.convert_to_tensor(X_train)
        #X_test=tf.convert_to_tensor(X_test)
        model.fit(X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=parameters["batch_size"],
        epochs=parameters["iterations"],
        callbacks=[es, rlp],verbose=0)
        return model
    
    def _getProcessors(self):
        return [catToMean().initialize({},verbose=0),normalizeScaler().initialize({},verbose=0)]
    
    def _saveModel(self, path):
        self.model.save(path)
        
    def __str__(self):
        return "ANNRegression"