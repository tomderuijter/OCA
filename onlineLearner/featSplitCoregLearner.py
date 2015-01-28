'''
Created on Mar 25, 2012

@author: Tom de Ruijter
@author: Evgeni Tsivtsivadze
'''

from learner import OnlineLearner
import numpy
from predictionFunctions.prediction import FeatureCoregularizedPrediction as FeatCoregPred
import collections

class LinearFeatureCoregressionLearner(OnlineLearner):
    '''A class containing a general learner for online training.
    Fit for large binary class (encoding [0,1]) datasets with an
    arbitrary amount of binary features.'''
    
    def __init__(self, Loss, LearnRule, Task, **params):
        '''Model parameters: eta0, lambda1, lamda2, nu, k, N, D'''
        super(LinearFeatureCoregressionLearner, self).__init__(Loss, LearnRule,
                                                               Task, **params)
        if float(self.params['k']) == 0:
            self.params['nu/k'] = self.params['nu']
        else:
            self.params['nu/k'] = self.params['nu'] / float(self.params['k'])
        #Prediction function
        self.Predict = getattr(FeatCoregPred, Task)
        
        #Weight definition
        self._w = collections.defaultdict(float)
        self._ws1 = 1.0
        self._ws2 = 1.0
    
    def norm(self):
        return sum([w*w for w in self._w])        

    def train_one(self,
                  features1, features2, target,
                  unlFeatures1, unlFeatures2, 
                  lrate1, lrate2):
        '''Trains the model by presenting two labeled samples
        and an arbitrary number of unlabled samples.'''
        # First view
        sumterm1 = sum([self._w[x] * features1[x] for x in features1])
        gradient1 = self.Loss.dloss(sumterm1, target)
                
        unlSumterm1 = sum([sum([self._w[x] * y[x] for x in y]) for y in unlFeatures1])
        unlSumterm2 = sum([sum([self._w[x] * y[x] for x in y]) for y in unlFeatures2])
        
        diff1 = unlSumterm1 - unlSumterm2

        lrateFactor1 = (1 - self.params['lambda1'] * lrate1)
        gradientFactor1 = lrate1 * gradient1
        diffFactor1 = lrate1 * self.params['nu/k'] * diff1 
        
        for f in features1:
            self._w[f] =  lrateFactor1 * self._w[f] \
                - gradientFactor1 * features1[f] \
                + diffFactor1 * features1[f]
        
        # Second view
        sumterm2 = sum([self._w[x] * features2[x] for x in features2])
        gradient2 = self.Loss.dloss(sumterm2, target)
        diff2 = unlSumterm2 - unlSumterm1
        
        lrateFactor2 = (1 - self.params['lambda2'] * lrate2)
        gradientFactor2 = lrate2 * gradient2
        diffFactor2 = lrate2 * self.params['nu/k'] * diff2
        
        for f in features2:
            self._w[f] = lrateFactor2 * self._w[f] \
                - gradientFactor2 * features2[f] \
                + diffFactor2 * features2[f]

    def renorm(self, modelNo):
        if modelNo == 1:
            s = self._ws1
            features = self.params['f1']
        else:
            s = self._ws2
            features = self.params['f2']
        
        for f in features:
            self._w[f] = self._w[f] / s
        
        if modelNo == 1:
            self._ws1 = 1.
        else:
            self._ws2 = 1.
    
    def train_one_ex2(self,
                  features1, features2, target,
                  unlFeatures1, unlFeatures2,
                  lrate1, lrate2):
        '''Trains the model by presenting two labeled samples
        and an arbitrary number of unlabeled samples.'''
        #It's important to do all W calculations prior to updating it.
        #Labeled sumterms
        sumterm1 = sum([(self._w[x] / self._ws1) * features1[x] for x in features1])
        gradFactor1 = lrate1 * self._ws1 * self.Loss.dloss(sumterm1, target)
        sumterm2 = sum([(self._w[x] / self._ws2) * features2[x] for x in features2])
        gradFactor2 = lrate2 * self._ws2 * self.Loss.dloss(sumterm2, target)
        
        #Unlabeled sumterms
        diffs1 = [0] * len(unlFeatures1)
        diffs2 = [0] * len(unlFeatures2)
        for i in xrange(len(unlFeatures1)):
            u1 = unlFeatures1[i]
            u2 = unlFeatures2[i]
            unlSum1 = sum([(self._w[x] / self._ws1) * u1[x] for x in u1])
            unlSum2 = sum([(self._w[x] / self._ws2) * u2[x] for x in u2])
            diffs1[i] = unlSum1 - unlSum2
            diffs2[i] = unlSum2 - unlSum1
        
        #Labeled updates
        for f in features1:
            self._w[f] -= gradFactor1 * features1[f]
        for f in features2:
            self._w[f] -= gradFactor2 * features2[f]
        
        #Unlabeled updates
        for i in xrange(len(unlFeatures1)):
            u1 = unlFeatures1[i]
            u2 = unlFeatures2[i]
            diffFactor1 = 4 * self._ws1 * diffs1[i] * lrate1 * self.params['nu/k']
            diffFactor2 = 4 * self._ws2 * diffs2[i] * lrate2 * self.params['nu/k']
            for f in u1:
                self._w[f] -= diffFactor1 * u1[f]
            for f in u2:
                self._w[f] -= diffFactor2 * u2[f]
                
                #Regularization updates    
        self._ws1 = self._ws1 / (1 - self.params['lambda1'] * lrate1)
        self._ws2 = self._ws2 / (1 - self.params['lambda2'] * lrate2)
        if self._ws1 > 1e3:
            self.renorm(1)
        if self._ws2 > 1e3:
            self.renorm(2)

    def train_one_ex(self,
                  features1, features2, target,
                  unlFeatures1, unlFeatures2,
                  lrate1, lrate2):
        '''Trains the model by presenting two labeled samples
        and an arbitrary number of unlabeled samples.'''
        #It's important to do all W calculations prior to updating it.
        #Labeled sumterms
        sumterm1 = sum([(self._w[x]) * features1[x] for x in features1])
        gradFactor1 = lrate1 * self.Loss.dloss(sumterm1, target)
        sumterm2 = sum([(self._w[x]) * features2[x] for x in features2])
        gradFactor2 = lrate2 * self.Loss.dloss(sumterm2, target)
        
        #Unlabeled sumterms
        diffs1 = [0] * len(unlFeatures1)
        diffs2 = [0] * len(unlFeatures2)
        for i in xrange(len(unlFeatures1)):
            u1 = unlFeatures1[i]
            u2 = unlFeatures2[i]
            unlSum1 = sum([self._w[x] * u1[x] for x in u1])
            unlSum2 = sum([self._w[x] * u2[x] for x in u2])
            diffs1[i] = unlSum1 - unlSum2
            diffs2[i] = unlSum2 - unlSum1
        
        #Regularization updates    
        lrateFactor1 = (1 - self.params['lambda1'] * lrate1)
        lrateFactor2 = (1 - self.params['lambda2'] * lrate2)
        for f in self.params['f1']:
            self._w[f] = lrateFactor1 * self._w[f]
        for f in self.params['f2']:
            self._w[f] = lrateFactor2 * self._w[f]
        
        #Labeled updates
        for f in features1:
            self._w[f] -= gradFactor1 * features1[f]
        for f in features2:
            self._w[f] -= gradFactor2 * features2[f]
        
        #Unlabeled updates
        for i in xrange(len(unlFeatures1)):
            u1 = unlFeatures1[i]
            u2 = unlFeatures2[i]
            diffFactor1 = diffs1[i] * lrate1 * self.params['nu/k']
            diffFactor2 = diffs2[i] * lrate2 * self.params['nu/k']
            for f in u1:
                self._w[f] -= diffFactor1 * u1[f]
            for f in u2:
                self._w[f] -= diffFactor2 * u2[f]
        

    
    def train_model(self, trainData1, trainData2, trainLabels, unlabData1, unlabData2):
        '''Trains the entire model on all data with k randomly sampled
        unlabeled points per pass.'''
        
        # Save function handles & variables for speed
        train = self.train_one_ex2
        sample = numpy.random.randint
        lrule = self.LearnRule.update
        eta0 = self.params['eta0']
        lambda1 = self.params['lambda1']
        lambda2 = self.params['lambda2']
        k = self.params['k']
        N = self.params['N']
        t = 1
        
        for i in xrange(len(trainData1)):
            if k != 0:
                unlabSet = sample(len(unlabData1), size=k)
                unlab_features1 = [unlabData1[u] for u in unlabSet]
                unlab_features2 = [unlabData2[u] for u in unlabSet]
            else:
                unlab_features1 = []
                unlab_features2 = []
                
            lrate1 = lrule(eta0, lambda1, t, N)        
            lrate2 = lrule(eta0, lambda2, t, N)
            
            train(trainData1[i], trainData2[i], trainLabels[i], 
                  unlab_features1, unlab_features2, 
                  lrate1, lrate2)
            t += 1
        return self._w
    
    
    def predict_set(self, testData1, testData2):
        '''Classifies an entire dataset'''
        predict_one = self.Predict
        predictions = [0] * len(testData1)
        for i in xrange(len(testData1)):
            predictions[i] = predict_one(self._w, testData1[i], testData2[i])
        return predictions