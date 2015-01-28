'''
Created on May 14, 2012

@author: Tom de Ruijter
'''

import numpy

def arg_max_2(results, measure):
    bestRow = None
    bestColumn = None
    bestScore = 0
    for i in xrange(len(results)):
        for j in xrange(len(results[0])):
            if measure.__name__ == 'RMSE':
                score = 1. / results[i][j]
            else:
                score = results[i][j]
            if score > bestScore:
                bestRow = i
                bestColumn = j
                bestScore = score
    return bestRow, bestColumn

def arg_max_3(results, measure):
    bestLayer = None
    bestRow = None
    bestColumn = None
    bestScore = 0.
    for i in xrange(len(results)):
        for j in xrange(len(results[0])):
            for k in xrange(len(results[0][0])):
                if measure.__name__ == 'RMSE':
                    score = 1. / results[i][j][k]
                else:
                    score = results[i][j][k]
                if score > bestScore:
                    bestRow = i
                    bestColumn = j
                    bestLayer = k
                    bestScore = score
    return bestRow, bestColumn, bestLayer

def arg_max_4(results, measure):
    bestLayer = None
    bestRow = None
    bestColumn = None
    bestNiche = None
    bestScore = 0.
    
    for i in xrange(len(results)):
        for j in xrange(len(results[0])):
            for k in xrange(len(results[0][0])):
                for l in xrange(len(results[0][0][0])):
                    if measure.__name__ == 'RMSE':
                        score = 1. / results[i][j][k][l]
                    else:
                        score = results[i][j][k][l]
                    if score > bestScore:
                        bestRow = i
                        bestColumn = j
                        bestLayer = k
                        bestNiche = l
                        bestScore = score
        
    return bestRow, bestColumn, bestLayer, bestNiche

def sample_array(data, k):
    '''Given an array, uniquely selects k items from it.'''
    if k >= len(data):
        return range(len(data))
    perm = range(len(data))
    numpy.random.shuffle(perm)
    perm = perm[:k]
    return perm