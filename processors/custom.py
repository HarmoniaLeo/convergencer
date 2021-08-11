from base import base

class custom(base):
    def __init__(self, parameters):
        self.function=self.getParameter("function",lambda x:x,parameters)
    
    def transform(self, data):
        return self.function(data)
    
    def __str__(self):
        return "custom"