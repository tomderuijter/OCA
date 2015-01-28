'''
Created on Jan 13, 2012

@author: Tom de Ruijter
@author: Evgeni Tsivtsivadze
'''

from learner import OnlineLearner
from predictionFunctions.prediction import CoregularizedPrediction as CoregPred
import collections
import random

class LinearCoregressionLearner(OnlineLearner):
    '''A class containing a general learner for online training.
    Fit for large binary class (encoding [0,1]) datasets with an
    arbitrary amount of binary features.'''
    
    def __init__(self, Loss, LearnRule, Task, **params):
        '''Model parameters: eta0, lambda, nu, k, N, D'''
        super(LinearCoregressionLearner, self).__init__(Loss, LearnRule, Task, 
                                                        **params)
        
        #Prediction function
        self.Predict = getattr(CoregPred, Task)
        
        #Weight definition
        self._w = collections.defaultdict(float)
        
        
    def norm(self):
        return sum([w*w for w in self._w])


    def split_data(self, data, labels):
        '''Splits data & labels in two equal parts'''
        N = int(self.params['N'] / 2)
        data_1 = data[:N]
        labels_1 = labels[:N]
        data_2 = data[N:]
        labels_2 = labels[N:]
        return N, data_1, labels_1, data_2, labels_2
    
    
    def train_one(self,
                  features1, target1,
                  features2, target2,
                  unlFeatures, unlabSet, lrate):
        '''Trains the model by presenting two labeled samples
        and an arbitrary number of unlabled samples.'''
        #unlFeatures := all unlabeled samples.
        #unlabSet    := indices of chosen unlabeled samples.
        
        D = self.params['D']
        
        # Model predictions
        sumterm1 = sum([self._w[x] * features1[x] for x in features1])
        sumterm2 = sum([self._w[x+D] * features2[x] for x in features2])

        # Model disagreement on unlabeled data.
        diff = sum([
                    sum([
                         (self._w[x] * y[x]) - \
                         (self._w[x+D] * y[x]) for x in y
                    ]) 
                    for y in [unlFeatures[z] for z in unlabSet]
                ])

        # Compute the gradient.
        gradient1 = self.Loss.dloss(sumterm1, target1)
        gradient2 = self.Loss.dloss(sumterm2, target2)
        
        # Update model with gradient
        for f in features1:
            g = f
            self._w[g] = (1 - self.params['lambda_'] * lrate) * self._w[g] \
                - lrate * gradient1 * features1[f] \
                + lrate * self.params['nu'] * diff
        
        for f in features2:
            g = f + self.params['D']
            self._w[g] = (1 - self.params['lambda_'] * lrate) * self._w[g] \
                - lrate * gradient2 * features2[f] \
                + lrate * self.params['nu'] * diff  
    
    
    def train_model(self, data, labels, unlabData):
        '''Trains the entire model on all data with k randomly sampled
        unlabeled points per pass.'''
        
        # Save function handles & variables for speed
        train = self.train_one
        sample = random.sample
        lrule = self.LearnRule.update
        eta0 = self.params['eta0']
        lambda_ = self.params['lambda_']
        k = self.params['k']
        t = 1
        
        N, data_1,labels_1,data_2,labels_2 = self.split_data(data, labels)
        
        for i in xrange(N):
            unlabSet = sample(xrange(len(unlabData)), k)                    
            train(data_1[i], labels_1[i],
                  data_2[i], labels_2[i],
                  unlabData, unlabSet,
                  lrule(eta0, lambda_, t, N))
            t += 1
        return self._w
    
    
    def predict_set(self, testData):
        '''Classifies an entire dataset'''
        predict_one = self.Predict
        predictions = [0] * len(testData)
        for i in xrange(len(testData)):
            predictions[i] = predict_one(self._w, self.params['D'], testData[i])
        return predictions