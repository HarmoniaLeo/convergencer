from convergencer.processors.base import base

class custom(base):
    def __init__(self, data, y=None, parameters={},verbose=1):
        '''
        parameters:
            {
                "function": your custom processing function
            }
        '''
        self.verbose=verbose
        self.function=self.getParameter("function",lambda x:x,parameters)
    
    def transform(self, data, y=None):
        print("\n-------------------------Using custom fuction to transform data-------------------------")
        return super().transform(self.function(data), y=y)
    
    def __str__(self):
        return "custom"