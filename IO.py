'''
Created on Mar 20, 2012

@author: Tom de Ruijter
'''

import collections
import random
import cPickle as pickle
from time import time
from os.path import basename, splitext

def load_data(filePath):
    """Reads raw data from a file and converts each sample into a list.
    File format: Label [featurenumber:featurevalue]*"""
    
    dataFile = open(filePath)
    
    data = []
    labels = []
    for sample in dataFile:
        fields = sample.strip('\n').split(' ')
        #enforce [0,1] class encoding
        if float(fields[0]) < 0.:
            labels.append(0.0)
        else:
            labels.append(float(fields[0]))
        
        fields = [x.split(':') for x in fields[1:]]
        sample = collections.defaultdict(float)
        
        if fields:    
            if fields[-1] == ['']:
                fields.pop(-1)
        if fields:
            if fields[0] == ['']:
                fields.pop(0)

        for i in xrange(len(fields)):
            sample[int(fields[i][0])] = float(fields[i][1])
            
        data.append(sample)
    dataFile.close()
    return data, labels


def load_samples(filePath):
    """Loads and returns train and test data for a standard linear model."""
    
    s = splitext(basename(filePath))[0]
    picklePath = 'tmp/' + s + '.pkl'
    loadTime = time()
    try:
        pickleFile = open(picklePath, 'rb')
        print 'Loading pickled data...'
        data = pickle.load(pickleFile)
        labels = pickle.load(pickleFile)
        pickleFile.close()
    except:
        print 'No pickled data could be read, recomputing features.'
        data, labels = load_data(filePath)
        pickleFile = open(picklePath, 'wb')
        pickle.dump(data, pickleFile, -1)
        pickle.dump(labels, pickleFile, -1)
        pickleFile.close()
    print('Finished loading %s (%fs).' % (s, round(time()-loadTime,4)))
    
    return data, labels

def split_features(D):
    x  = range(1, D+1)
    random.shuffle(x)
    D1 = int(D) / 2
    D2 = D - D1
    f1 = sorted(x[:D1])
    f2 = sorted(x[D1:])
    return D1, D2, f1, f2

def split_features_big(D):
    x = int(D / 2)
    D1 = int(D) / 2
    D2 = D - D1
    f1 = xrange(1, x)
    f2 = xrange(x, D+1)
    return D1, D2, f1, f2

def split_sample(fields, L1):
    s1 = collections.defaultdict(float)
    s2 = collections.defaultdict(float)
    
    for i in xrange(len(fields)):
        try:
            if L1[fields[i][0]]:
                s1[int(fields[i][0])] = float(fields[i][1])
        except KeyError:
            s2[int(fields[i][0])] = float(fields[i][1])
    return s1,s2
            
            
def load_data_split(filePath, task, f1):
    """Reads raw data from a file and converts each sample into a list."""
    """File format: Label [featurenumber:featurevalue]*"""
    
    dataFile = open(filePath)
    
    data1 = []
    data2 = []
    labels = []
    L1 = dict(zip(f1,f1))
    for sample in dataFile:
        fields = sample.strip('\n').split(' ')
        #enforce [0,1] class encoding
        if float(fields[0]) < 0.:
            labels.append(0.0)
        else:
            labels.append(float(fields[0]))
        #Remove label from list
        fields = [x.split(':') for x in fields[1:]]
        if fields:    
            if fields[-1] == ['']:
                fields.pop(-1)
        if fields:
            if fields[0] == ['']:
                fields.pop(0)

        s1 = collections.defaultdict(float)
        s2 = collections.defaultdict(float)
    
        for i in xrange(len(fields)):
            if L1.has_key(int(fields[i][0])):
                s1[int(fields[i][0])] = float(fields[i][1])
            else:
                s2[int(fields[i][0])] = float(fields[i][1])

        data1.append(s1)
        data2.append(s2)
    dataFile.close()
    return data1, data2, labels


def load_samples_split(filePath, task, f1):
    s = splitext(basename(filePath))[0]
    picklePath = 'tmp/' + s + '_split.pkl'
    loadTime = time()
    try:
        pickleFile = open(picklePath, 'rb')
        print 'Loading pickled data...'
        data1 = pickle.load(pickleFile)
        data2 = pickle.load(pickleFile)
        labels = pickle.load(pickleFile)
        pickleFile.close()
    except:
        print 'No pickled data could be read, recomputing features.'
        data1, data2, labels = load_data_split(filePath, task, f1)
        pickleFile = open(picklePath, 'wb')
        pickle.dump(data1, pickleFile, -1)
        pickle.dump(data2, pickleFile, -1)
        pickle.dump(labels, pickleFile, -1)
        pickleFile.close()
    print('Finished loading %s (%fs).' % (s, round(time()-loadTime,4)))
    
    return data1, data2, labels

    
def write_results(fileName, classifier, loss, measure, predictions, k = 0):
    '''Writes model predictions to a file, with one prediction per line.'''
    
    name = splitext(basename(fileName))[0]
    filePath = 'predictions/' + name + '_' + classifier + '_' + loss \
                            +'_' + measure +'_k' + str(k) + '.txt' 
                            
    writeFile = open(filePath, 'w')
    for p in xrange(len(predictions)):
        line = str(predictions[p]) + '\n'
        writeFile.write(line)
    writeFile.close()