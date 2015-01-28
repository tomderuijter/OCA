'''
Created on Feb 19, 2012

@author: Tom de Ruijter

Implements different kinds of learning rate updates.
To use, just include the right learn-rate when initializing a classifier.
'''

class StdLearnRate(object):
    '''Learn rate interface. The bare minimum of each learning rate.'''
    @staticmethod
    def update(eta0 = 0.1, lambda_ = 0, t = 1, N = 1):
        '''Computes the learning rate'''
        
        raise NotImplementedError("No learning rate function implemented")
  
  
class EvgeniLearnRate(StdLearnRate):
    '''Computes the learning rate as Evgeni Tsivtsivadze did
    for logistic regression.'''

    @staticmethod
    def update(eta0 = 0.1, lambda_ = 0, t = 1, N = 1):
        #arguments: eta_0, t, N
        #eta_t = eta_0 / (1 + (t/N))
        return eta0 / (1 + (float(t)/N))


class PegasosLearnRate(StdLearnRate):
    '''Computes the learning rate as done by Shalev Schwartz
    for Pegasos.'''
    @staticmethod
    def update(eta0 = 0.1, lambda_ = 0, t = 1, N = 1):
        #arguments: lambda, t
        #eta_t = 1 / (lambda * t)
        return 1 / (lambda_ * t)

class NewPegasosLearnRate(StdLearnRate):
    '''Computes the learning rate as done by Shalev Schwartz
    for Pegasos.'''
    @staticmethod
    def update(eta0 = 0.1, lambda_ = 0, t = 1, N = 1):
        #arguments: lambda, t
        #eta_t = 1 / (lambda * t)
        return 1. / (lambda_ *  (t + eta0))

class BottouLearnRate(StdLearnRate):
    '''Computes the learning rate as done by Leon Bottou for
    SGD-based SVMs'''
    @staticmethod
    def update(eta0 = 0.1, lambda_ = 0, t = 1, N = 1):
        #arguments: eta_0, lambda, t
        #eta_t = eta_0 / (1 + lambda * eta_0 * t)
        return eta0 / (1 + lambda_ * eta0 * t)