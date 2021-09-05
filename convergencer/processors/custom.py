from convergencer.processors.base import base

def defaultFunc(data,y,params):
    return data,y

def defaultFit(data,y):
    return None

class customFeatureEngineer(base):
    def initialize(self,parameters={},verbose=1):
        '''
        parameters:
            {
                "fit": 
                    def fit(data,y):
                        ...
                        return params
                "transform": 
                    def transform(data,y,params):
                        ...
                        return data,y
                "reTransform":
                    def reTransform(data,y,params):
                        ...
                        return data,y
            }
        '''
        self.verbose=verbose
        self.fitFunction=self._getParameter("fit",defaultFit,parameters)
        self.transFunction=self._getParameter("transform",defaultFunc,parameters)
        self.reTransFunction=self._getParameter("reTransform",defaultFunc,parameters)
        return self

    def fit(self, data, y=None,parameters={}):
        self.params=self.fitFunction(data,y)
        return self
    
    def transform(self, data, y=None):
        if self.verbose==1:
            print("\n-------------------------Using custom fuction to transform data-------------------------")
        if not (y is None):
            y=y.copy()
        data=data.copy()
        data,y=self.transFunction(data,y,self.params)
        return super().transform(data, y=y)
    
    def reTransform(self, data, y):
        if self.verbose==1:
            print("\n-------------------------Using custom fuction to re-transform data-------------------------")
        if not (y is None):
            y=y.copy()
        data=data.copy()
        data,y=self.reTransFunction(data,y,self.params)
        return super().reTransform(data, y=y)
    
    def __str__(self):
        return "customFeatureEngineer"