'''
Created on Jan 13, 2012

@author: Tom de Ruijter
@author: Evgeni Tsivtsivadze
'''

import IO
import random
import cProfile
import data
import numpy
from time import time
from os.path import basename
import measures as ms
import crossValidation as cv
from onlineLearner.lossFunctions import loss
from onlineLearner.learnRates import learnrate as lr
from onlineLearner.linearLearner import LinearLearner as LinLearn
from onlineLearner.featSplitCoregLearner import LinearFeatureCoregressionLearner as featCoregLearn    

def main(classifier, Loss, LRule, measure, task, dataset, k=0, trainFolds=1,
         doCV=True, copy=False, folds=5, splitFactor=0.8,
         lambda1Params=[1.0], lambda2Params=[1.0], 
         lRates=[0.1], nuParams=[1.0]):
        
    print('Learning %s %s model' % (classifier, task))
    print('Using %s and %s.' % (Loss.__name__, LRule.__name__))
    print('Performance measure: %s.' % measure.__name__)
    
    random.seed(100)                    #Fix random seed
    numpy.random.seed(100)
    args, N, D = data.get_set(dataset)  #prepares data for loading
    
    #default parameters
    eta0 = lRates[0]
    lambda1 = lambda1Params[0]
    lambda2 = lambda2Params[0]
    nu = nuParams[0]

    # Regular Linear Classifier
    if classifier == 'linear':
        traData, traLabels = IO.load_samples(args[0])
        
        if doCV:
            eta0, lambda1 = cv.cross_validate_2(LinLearn, Loss, LRule, task, traData, traLabels, N, D,
                                                folds, lambda1Params, lRates,
                                                splitFactor, measure)
        print('Using parameters: LRate = %f, Lambda = %f' % (eta0, 
                                                             lambda1))

        trainTime = time()
        model = LinLearn(Loss, LRule, task, 
                          eta0=eta0, lambda_=lambda1,
                          N=N, D=D)
        for _ in xrange(trainFolds):
            model.train_model(traData, traLabels)
        print('Finished training (%fs).' % round(time()-trainTime,4))
        
        teData, teLabels = IO.load_samples(args[1])
        predictions = model.predict_set(teData)
        score = cv.check_predictions(teLabels, predictions, measure)
        print('Model performance: %f.' % score)
        IO.write_results(args[1], classifier, Loss.__name__, measure.__name__, predictions)
    
    #Feature split co-regularized model
    elif classifier =='coreg':
        _, _, f1, f2 = IO.split_features_big(D)
        traData1, traData2, traLabels = IO.load_samples_split(args[0], task, f1)
        if k != 0:
            unlabData1, unlabData2, _ = IO.load_samples_split(args[2], task, f1)
        else:
            unlabData1, unlabData2 = ([], [])
        
        if doCV:
            if copy:
                #TODO: Implement this to do cv3.
                pass
            else:
                eta0, lambda1, lambda2, nu = cv.cross_validate_4(featCoregLearn, Loss, LRule, task,
                                                                 traData1, traData2, traLabels, unlabData1, unlabData2, f1, f2, D, N,
                                                                 folds, lambda1Params, lambda2Params, nuParams, lRates,
                                                                 splitFactor, measure, k)
        evalTime = time()
        model = featCoregLearn(Loss, LRule, task, eta0=eta0, 
                               lambda1=lambda1, lambda2=lambda2, nu=nu,
                               k=k, N=N, D=D, f1=f1, f2=f2)     
        for _ in xrange(trainFolds):
            model.train_model(traData1, traData2, traLabels, unlabData1, unlabData2)
        
        teData1, teData2, teLabels = IO.load_samples_split(args[1], task, f1)
        predictions = model.predict_set(teData1, teData2)
        
        score = cv.check_predictions(teLabels, predictions, measure)

        print('Finished evaluating (%fs/p).' % round(time()-evalTime,4))
        print('Using %s.' % Loss.__name__)
        print('Trained on %s. Tested on %s.' % (basename(args[0]), basename(args[1])))
        print('Using parameters: LRate = %f, Lambda1 = %f, Lambda2 = %f, Nu = %f, k = %d' % 
              (eta0, lambda1, lambda2, nu, k))
        print('Actual Model performance: %g.' % score)
        IO.write_results(args[1], classifier, Loss.__name__, measure.__name__, predictions, k)
        
if __name__ == "__main__":
    #General poperties
    classifier = 'coreg'
    Loss = loss.HingeLoss
    LRule = lr.NewPegasosLearnRate
    measure = ms.AUC
    task = 'regression'    #Choose 'regression' when using AUC 
    dataset = 'epsilon'
    trainFolds = 2

    #Cross Validation parameters
    doCV = True
    copy = False           #Use same lambda for multiple views
    folds = 5
    splitFactor = 0.8
    
    #Linear Model parameters
    if classifier == 'linear':
        lambda1Params = [2**(-8), 2**(-7), 2**(-6), 2**(-5), 2**(-4), 2**(-3), 2**(-2), 2**(-1),2**(0),
                         2**(1), 2**(2), 2**(3), 2**(4), 2**(5)]
        lRates = [0.0625, 0.125, 0.25, 0.5, 0.75, 1., 1.25, 2.5, 5, 10., 100.]

    #Feature co-regression model parameters
    elif classifier == 'coreg':     
        lRates = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        lambda1Params = [2**(-8), 2**(-7), 2**(-6), 2**(-5),2**(-4), 2**(-3), 2**(-2), 2**(-1), 2**(0), 2**(1), 2**(2), 2**(3)]
    lambda2Params = [2**(-8), 2**(-7), 2**(-6), 2**(-5),2**(-4), 2**(-3), 2**(-2), 2**(-1), 2**(0), 2**(1), 2**(2), 2**(3)]
    nuParams = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    
#    lRates = [1.5]
#    lambda1Params = [2**(-10), 2**(-9), 2**(-8), 2**(-7), 2**(-6), 2**(-5), 2**(-4)]
#    lambda2Params = [2**(-10), 2**(-9), 2**(-8), 2**(-7), 2**(-6), 2**(-5), 2**(-4)]
#
    lRates = [0.125]    
    lambda1Params = [2**(-10)]
    lambda2Params = [2**(-10)]
    nuParams = [0.005]
    k = 3
    #End of user input
    if k == 0:
        nuParams = [0.]    
    
    main(classifier, Loss, LRule, measure, task, dataset, k, trainFolds,
                  doCV, copy, folds, splitFactor,
                  lambda1Params, lambda2Params, lRates, nuParams)
    
#    cProfile.run('main(classifier, Loss, LRule, measure, task, dataset, k,\
#                  trainFolds, doCV, copy, folds, splitFactor, \
#                  lambda1Params, lambda2Params, lRates, nuParams)')