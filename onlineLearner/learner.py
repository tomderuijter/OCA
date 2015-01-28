'''
Created on Mar 14, 2012
@author: Tom de Ruijter

An abstract class for implementation of specialised online learners.
'''

class OnlineLearner(object):
    '''Defines a new online learner.
    Note: weights are defined on implementation level.'''
    
    def __init__(self, Loss, LearnRule, Task, **params):
        self.Loss = Loss            #Loss function handle
        self.LearnRule = LearnRule  #Learn parameter update handle
        self.Task = Task            #Model task, e.g. 'classification'
        
        #self.Predict = getattr($pred_class, Task)
        
        self.params = params        #Model & data parameter dictionary
        
        #self._w = w                #Implementation dependent

    # Weight property
    def getw(self):
        return self._w
    def setw(self, value):
        self._w = value
    def delw(self):
        del self._w
    w = property(getw, setw, delw, "Model weights")
    
    def norm(self):
        raise NotImplementedError("No appropriate norm implemented.")
    
    def train_one(self, *args):
        raise NotImplementedError(self.__name__ + \
                                  ": no train method specified.")

    def train_model(self, *args):
        raise NotImplementedError(self.__name__ + \
                                  ": no train procedure specified.")
    
    def predict_set(self, *args):
        raise NotImplementedError(self.__name__ + \
                                  ": no test procedure specified.")
