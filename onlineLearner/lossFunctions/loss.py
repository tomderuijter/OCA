'''
Created on Dec 11, 2011

@author: Tom de Ruijter

Due to the lack of abstract classes and  interfaces in
Python (or I haven't found them), new Loss classes should 
inherit this class and implement at least the following.

WARNING: only computes derivative with respect to y.
         Multiply it with derivative of y to w.
'''

import math

#y := predicted value
#t := target value
class Loss(object):
    '''Random Loss function class, including Loss gradient'''
    
    @staticmethod
    def Loss(y, t):
        '''Computes Loss L(y, t)'''
        
        raise NotImplementedError("No Loss function implemented")
    
    @staticmethod
    def dloss(y, t):
        '''Computes gradient Loss dL(y, t)/dy'''
        
        raise NotImplementedError("No gradient Loss function implemented")
    

class LogLoss(Loss):
    '''Implements the log Loss and its derivative to y.'''

    # 0, 1 labeling
    #logLoss(y,t) = -(t * log(y) + (1 - t) * log(1-y))        
#    @staticmethod
#    def Loss(y, t):
#        return -(t * math.log(y) + (1 - t) * math.log(1-y))
        
    #dloss(y,t) / dy     
    @staticmethod
    def dloss(y, t):
#        if -100 < y:
#            return (1. / (1. + math.exp(-y))) - t
#        else:
#            return - t
        return ((-100 < y) and (1. / (1. + math.exp(-y))) - t) or -t


class HingeLoss(Loss):
    '''Implements the hinge Loss and its derivative to y.'''

    # 0, 1 labeling
    #hingeLoss(y,t) = max(0, 1-y*t)        
    @staticmethod
    def Loss(y, t):
        t2 = 2*t-1
        return 1 - y*t2 if y*t2 < 1 else 0
        
    #dloss(y,t) / dy     
    @staticmethod
    def dloss(y, t):
        t2 = 2*t-1
        return -t2 if y*t2 < 1 else 0
    
    
class SquaredLoss(Loss):
    '''Implements the quadratic Loss and its derivative to y.'''

    # 0, 1 labeling
    #quadraticLoss(y,t) = 0.5 * (t - y)^2        
    @staticmethod
    def Loss(y, t):
        t2 = 2*t-1
        return 0.5 * (y - t2) ** 2
        
    #dloss(y,t) / dy     
    @staticmethod
    def dloss(y, t):
        t2 = 2*t-1
        return y-t2
    
        
class SquaredHingeLoss(Loss):
    '''Implements the squared hinge Loss and its derivative to y.'''
    
    # 0, 1 labeling
    #squaredHingeLoss(y,t) = 0.5 * max(0, 1-y*t)^2
    @staticmethod
    def Loss(y, t):
        t2 = 2*t-1
        return 0.5 * (1 - y*t2)**2 if y*t2 < 1 else 0

    @staticmethod
    def dloss(y, t):
        t2=2*t-1
        return y*t2*t2+t2 if y*t2 < 1 else 0
    
