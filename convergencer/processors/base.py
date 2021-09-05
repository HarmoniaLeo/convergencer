class base:
    def initialize(self,parameters={},verbose=1):
        self.verbose=verbose
        return self

    def fit(self,data,y=None):
        return self

    def transform(self,data,y=None):
        if y is None:
            return data
        else:
            return data,y
    
    def reTransform(self,data,y=None):
        if y is None:
            return data
        else:
            return data,y
    
    def _getParameter(self,key,default,parameters):
        return default if key not in parameters.keys() else parameters[key]

    def __str__(self):
        return "BaseProcessor"