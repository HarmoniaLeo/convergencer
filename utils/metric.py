from sklearn.model_selection import KFold
import numpy as np

def cross_validation(X,y,model):
    kf = KFold(n_splits=5,shuffle=True)
    trainAccs=[]
    testAccs=[]
    for train_index, test_index in kf.split(X):
        X_train=X.loc[train_index]
        X_test=X.loc[test_index]
        y_train=y[train_index]
        y_test=y[test_index]
        model=model.fit(X_train,y_train)
        trainAccs.append(model.score(X_train,y_train))
        testAccs.append(model.score(X_test,y_test))
    return model,np.mean(trainAccs),np.mean(testAccs)