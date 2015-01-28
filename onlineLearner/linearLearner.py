'''
Created on Jan 13, 2012

@author: Tom de Ruijter
@author: Evgeni Tsivtsivadze
'''

from learner import OnlineLearner
from predictionFunctions.prediction import LinearPrediction
import collections

class LinearLearner(OnlineLearner):
    '''A class containing a general learner for online training.'''
    
    def __init__(self, Loss, LearnRule, Task, **params):
        '''Model parameters: eta0, lambda_, N, nrBatches'''
        super(LinearLearner, self).__init__(Loss, LearnRule, Task, **params)
        
        #Prediction function
        self.Predict = getattr(LinearPrediction, Task)
        
        #Weight definition
        self._w = collections.defaultdict(float)
        self._ws = 1.
        
    def norm(self):
        return sum([w*w for w in self._w])
    
    def renorm(self):
        for f in self._w:
            self._w[f] = self._w[f] / self._ws
        self._ws = 1.
    
    def train_one(self, features, target, lrate):
        sumterm = sum([(self._w[x] / self._ws) * features[x] for x in features])
        gradFactor = lrate * self._ws * self.Loss.dloss(sumterm, target)
        
        for f in features:
            self._w[f] -= gradFactor * features[f]
            
        self._ws = self._ws / (1 - self.params['lambda_'] * lrate)
        if self._ws > 1e3:
            self.renorm()
    
    def train_model(self, trainData, trainLabels):
        '''Trains the entire model'''
        # Save function handles & variables for speed
        train = self.train_one
        lrule = self.LearnRule.update
        eta0 = self.params['eta0']
        reg = self.params['lambda_']
        N = self.params['N']
        
        t = 1
        for i in xrange(len(trainData)):           
            train(trainData[i], trainLabels[i], lrule(eta0, reg, t, float(N)))
            t += 1
        return self._w
    
    
    def predict_set(self, testData):
        '''Classifies an entire dataset'''
        predict_one = self.Predict
        predictions = [0] * len(testData)
        for i in xrange(len(testData)):
            predictions[i] = predict_one(self._w, testData[i])
        return predictions