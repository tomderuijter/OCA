'''
Created on Mar 14, 2012

@author: Tom de Ruijter

When implementing a new learner, make sure that in its prediction class
each method has exactly the same arguments. Else interfacing is not possible.
'''



class Prediction(object):
    '''Abstract class for the implementation of prediction functions for 
    models. Depending on the task, a different prediction function can be 
    implemented.'''

    @staticmethod
    def regression(*args):
        raise NotImplementedError(  "Predictive function not implemented: \
                                    regression.")
    
    @staticmethod
    def classification(*args):
        raise NotImplementedError(  "Predictive function not implemented: \
                                    classification.")


class LinearPrediction(Prediction):
    
    @staticmethod
    def regression(w, features):
        return sum([w[f] * features[f] for f in features])
    
    @staticmethod
    def classification(w, features):
        return 1 if 0. < sum([w[f] for f in features]) else 0
    
    
class CoregularizedPrediction(Prediction):
    
    @staticmethod
    def regression(w, D, features):
        return sum([w[f] * features[f] for f in features]) * 0.5 \
            + sum([w[f+D] * features[f] for f in features]) * 0.5
    
    @staticmethod
    def classification(w, D, features):
        P = CoregularizedPrediction.regression(w, D, features)
        return 1 if 0. < P else 0
    
class FeatureCoregularizedPrediction(Prediction):
    
    @staticmethod
    def regression(w, features1, features2):
        return sum([w[f] * features1[f] for f in features1]) * 0.5 \
            + sum([w[f] * features2[f] for f in features2]) * 0.5
    
    @staticmethod
    def classification(w, features1, features2):
        P = CoregularizedPrediction.regression(w, features1, features2)
        return 1 if 0. < P else 0