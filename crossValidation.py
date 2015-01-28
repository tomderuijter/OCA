'''
Created on Feb 23, 2012

@author: Tom de Ruijter

Module containing a cross validation function and simple evaluation methods.
'''

import numpy
import measures as ms
from time import time
from numpy import array
from arrayFunctions import *


def check_predictions(labels, predictions, measure):
    '''Applies accuracy metric to predictions'''
    return measure(labels, predictions)


def make_train_data(data, perm, splitFactor):
    '''Splits available data horizontally in a training partition and a 
    validation partition'''
    count = int(len(data) * splitFactor)
    trainPerm = perm[:count]
    evalPerm = perm[count:]

    return trainPerm, evalPerm


def cross_validate_2(Classifier, Loss, LRule, task,
                     features, labels, N, D,
                     kFolds, regs, lRates, splitFactor = 0.9,
                     measure = ms.accuracy):
    '''Takes a classifier class, a loss funciton and learn rate rule,
    data and lists of parameters and applies cross validation on the 
    specified class of classifiers with given parameters.'''

    split = make_train_data
    cvTime = time()
    results = [[0] * len(regs) for _ in lRates]
    
    bestLRate = None
    bestReg = None
    
    for fold in xrange(kFolds):
        print('==Starting fold %d==' % (fold+1))
        perm = range(len(features))
        numpy.random.shuffle(perm)
        trainPerm, evalPerm = split(features, perm, splitFactor)        
        
        for i in xrange(len(lRates)):
            for j in xrange(len(regs)):
                # Create model
                cls = Classifier(Loss, LRule, task, eta0=lRates[i], lambda_=regs[j], N=int(N*splitFactor), D=D)
                trainData = [features[p] for p in trainPerm]
                trainLabels = [labels[p] for p in trainPerm]
                cls.train_model(trainData, trainLabels)                
                
                # Evaluate performance
                evalData = [features[p] for p in evalPerm]
                evalLabels = [labels[p] for p in evalPerm]
                pred = cls.predict_set(evalData)
                score = check_predictions(evalLabels, pred, measure)
#                print('LRate: %f, Lambda: %f, Score: %g.' % (lRates[i],
#                                                          regs[j],
#                                                          score))
                
                results[i][j] += score
            print '----'
        tmpRes = (array(results)/(fold+1)).tolist()
        k,l = arg_max_2(tmpRes, measure)
        bestLRate = lRates[k]
        bestReg = regs[l]
        print('Best parameters: LRate = %f, Lambda = %f' % (bestLRate, 
                                                            bestReg))
        print('Best average score: %f.' % tmpRes[k][l])
    print('Finished cross validation (%fs).' % round(time()-cvTime,4))
    return bestLRate, bestReg

#TODO: WARNING, NOT UP TO DATE.
def cross_validate_3(Classifier, Loss, LRule, task, 
                     data1, data2, labels, unlabData1, unlabData2, D, N, 
                   kFolds, lambdaParams, nuParams, lRates, splitFactor = 0.9, 
                   measure = ms.accuracy, K = 1):
    '''Takes a classifier class, a loss funciton and learn rate rule,
    data and lists of parameters and applies cross validation on the 
    specified class of classifiers with given parameters.'''

    split = make_train_data
    cvTime = time()
    
    results = [[[0] * len(nuParams) for _ in lambdaParams] for _ in lRates]
    bestLambda = None
    bestNu = None
    bestLRate = None
    
    for fold in xrange(kFolds):
        print('==Starting fold %d==' % (fold+1))
        perm = range(len(data1))
        numpy.random.shuffle(perm)
        trainPerm, evalPerm = split(data1, perm, splitFactor)        
        
        for i in xrange(len(lRates)):
            for j in xrange(len(lambdaParams)):
                for k in xrange(len(nuParams)):
                # Create classifier
                    cls = Classifier(Loss, LRule, task, 
                                     eta0=lRates[i], lambda_=lambdaParams[j], 
                                     nu=nuParams[k], k=K, N=N*splitFactor, 
                                     D=D)

                    traData1 = [data1[p] for p in trainPerm]
                    traData2 = [data2[p] for p in trainPerm]
                    trainLabels = [labels[p] for p in trainPerm]

                    cls.train_model(traData1, traData2, trainLabels, unlabData1, unlabData2)
                    
                    # Evaluate performance                        
                    evalData1 = [data1[p] for p in evalPerm]
                    evalData2 = [data2[p] for p in evalPerm]
                    evalLabels = [labels[p] for p in evalPerm]
                    
                    pred = cls.predict_set(evalData1, evalData2)
                    
                    score = check_predictions(evalLabels, pred, measure)

                    results[i][j][k] += score
            print '----'
        tmpResults = (array(results) / (fold+1)).tolist()
        a, b, c = arg_max_3(tmpResults, measure)                 
        bestLRate = lRates[a]
        bestLambda = lambdaParams[b]
        bestNu = nuParams[c]
        print('Best parameters: LRate = %f, Lambda = %f, Nu = %f' % (bestLRate, 
                                                                     bestLambda,
                                                                     bestNu))
        print('Best average score: %f.' % results[a][b][c])
    
    print('Finished cross validation (%fs).\n' % round(time()-cvTime,4))
    return bestLRate, bestLambda, bestNu

def cross_validate_4(Classifier, Loss, LRule, task,
                     data1, data2, labels, unlabData1, unlabData2, f1, f2, D, N,
                   kFolds, lambda1Params, lambda2Params, nuParams, lRates,
                   splitFactor = 0.5, measure = ms.accuracy, K = 1):
    '''Takes a classifier class, a loss funciton and learn rate rule,
    data and lists of parameters and applies cross validation on the 
    specified class of classifiers with given parameters.'''

    split = make_train_data
    cvTime = time()
    
    results = [[[[0] * len(nuParams) for _ in lambda2Params] for _ in lambda1Params] for _ in lRates]
    bestLRate = None
    bestLambda1 = None
    bestLambda2 = None
    bestNu = None
    
    for fold in xrange(kFolds):
        print('==Starting fold %d==' % (fold+1))
        perm = range(len(data1))
        numpy.random.shuffle(perm)
        trainPerm, evalPerm = split(data1, perm, splitFactor)   
        
        for i in xrange(len(lRates)):
            for j in xrange(len(lambda1Params)):
                for k in xrange(len(lambda2Params)):
                    for l in xrange(len(nuParams)):

                # Create classifier
                        cls = Classifier(Loss, LRule, task, 
                                         eta0=lRates[i], lambda1=lambda1Params[j], 
                                         lambda2=lambda2Params[k], nu=nuParams[l], k=K, N=int(N*splitFactor), 
                                         D=D, f1=f1, f2=f2)

                        traData1 = [data1[p] for p in trainPerm]
                        traData2 = [data2[p] for p in trainPerm]
                        trainLabels = [labels[p] for p in trainPerm]

                        cls.train_model(traData1, traData2, trainLabels, unlabData1, unlabData2)
                        
                        # Evaluate performance                        
                        evalData1 = [data1[p] for p in evalPerm]
                        evalData2 = [data2[p] for p in evalPerm]
                        evalLabels = [labels[p] for p in evalPerm]
                        
                        pred = cls.predict_set(evalData1, evalData2)
                        
                        score = check_predictions(evalLabels, pred, measure) 
                        results[i][j][k][l] += score
                        print('LRate: %f, Lambda1: %f, Lambda2: %f, Nu: %f - Score: %g.' % (lRates[i],
                                                                     lambda1Params[j],
                                                                     lambda2Params[k],
                                                                     nuParams[l],
                                                                     score))
            print '----'
        tmpResults = (array(results) / (fold+1)).tolist()
        a, b, c, d = arg_max_4(tmpResults, measure)                 
        bestLRate = lRates[a]
        bestLambda1 = lambda1Params[b]
        bestLambda2 = lambda2Params[c]
        bestNu = nuParams[d]
        print('Best parameters: LRate = %f, Lambda1 = %f, Lambda2 = %f, Nu = %f' % (bestLRate, 
                                                                                    bestLambda1,
                                                                                    bestLambda2,
                                                                                    bestNu))
        print('Best average score: %f.' % tmpResults[a][b][c][d])
    
    print('Finished cross validation (%fs).\n' % round(time()-cvTime,4))
    return bestLRate, bestLambda1, bestLambda2, bestNu