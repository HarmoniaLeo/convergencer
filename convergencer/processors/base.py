class base:
    def __init__(self,data,y=None,parameters={},verbose=1):
        self.verbose=verbose
    
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
    
    def getParameter(self,key,default,parameters):
        return default if key not in parameters.keys() else parameters[key]

    def __str__(self):
        return "BaseProcessor"