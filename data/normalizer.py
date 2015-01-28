'''
Created on Mar 9, 2012

@author: Tom de Ruijter

Linearly normalizes a feature file so all features are in range [-1,1]
and all labels are in range [0, 100].
'''

import collections
import sys
import random

def load_data(filePath):
    """Reads raw data from a file and converts each sample into a list."""
    """File format: Label [featurenumber:featurevalue]*"""
    
    dataFile = open(filePath)
    
    data = []
    labels = []
    for sample in dataFile:
        fields = sample.strip('\n').split(' ')
        
#        if float(fields[0]) == 2:
#            labels.append(1.0)
#        else:
#            labels.append(0.0)
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

def write_file(data, labels, filePath):
    
    assert len(data) == len(labels)
    
    writeFile = open(filePath, 'w')
    
    for l in xrange(len(data)):
        line = str(labels[l])
        for f in (data[l]):
            line += ' ' + str(f) + ':' + str(data[l][f])
        line += '\n'
        writeFile.write(line)
        
    writeFile.close()

def load_data_bin(filePath):
    """Reads binary feature data from a file and converts each sample 
    into a list."""
    
    dataFile = open(filePath)
    
    data = []
    labels = []
    for sample in dataFile:
        fields = sample.strip('\n').split('\t')
        fields = [int(x) for x in fields]    
        labels.append(fields[0])
        data.append(fields[1:])
    dataFile.close()
    return data, labels

def write_file_bin(data, labels, filePath):
    
    assert len(data) == len(labels)
    
    writeFile = open(filePath, 'w')
    
    for l in xrange(len(data)):
        line = str(labels[l])
        for f in (data[l]):
            line += ' ' + str(f) + ':' + '1.0'
        line += '\n'
        writeFile.write(line)
        
    writeFile.close()
    
def normalize_labels(labels):
    lst = min(labels)
    mst = max(labels)    
    
#    labels = [l * 100 for l in labels]
    
    # Ground labels
    if lst < 0:
        mst += abs(lst)
        labels = map(lambda x: ((x+abs(lst))/mst)*100., labels)
        lst = 0.0
    elif lst > 0:
        mst -= lst
        labels = map(lambda x: ((x-abs(lst))/mst)*100, labels)
        lst = 0.0

    return labels

def update_ends(mst, lst, x):
    if x > mst:
        mst = x
    if x < lst:
        lst = x
    return mst, lst
        
def ground_data(mst, lst, f, data):
    if lst < 0:
        mst += abs(lst)
        if mst == 0.0:
            mst = 1.0
        for d in data:
            if d[f] == 0.0:
                del d[f]
                continue
            d[f] = ((d[f]+abs(lst))/mst)*2. -1
        lst = 0
    elif lst >= 0:
        mst -= lst
        if mst == 0.0:
            mst = 1.0
        for d in data:
            if d[f] == 0.0:
                del d[f]
                continue
            d[f] = ((d[f]-abs(lst))/mst)*2. -1
        lst = 0
    return mst,lst, data

def normalize_features(data, featureDim):
    
    for f in xrange(1, featureDim+1):
        lst = data[0][f]
        mst = data[0][f]
        for d in data:
            if d[f] == 0.0:
                del d[f]
                continue
            mst, lst = update_ends(mst, lst, d[f])
        print 'Feature:', f, 'Most:', mst, 'Least:', lst
        if mst == 0.0:
#            print('Feature %d does not occur.' % f)
            continue
        mst,lst,data = ground_data(mst,lst, f, data)
#        print 'feature: ', f, ' mst: ', mst, ' lst: ', lst
        if f % 100 == 0:
            print 'Doing feature', f
    
    return data  

def randomize(data, labels):
    
    perm = range(len(data))
    random.shuffle(perm)
    
    randData = [data[p] for p in perm]
    randLabels = [labels[p] for p in perm]

    return randData, randLabels

def statistics(data, labels):
    
    pos = 0
    neg = 0
    for i in xrange(len(labels)):
        if labels[i] > 0:
            pos += 1
        else:
            neg += 1
    
    print('Positives: %f (%d/%d)' % ((float(pos) / len(labels)), pos, len(labels)))
    print('Negatives: %f (%d/%d)' % ((float(neg) / len(labels)), neg, len(labels)))

def fillCount(data, D):
    count = 0
    for d in data:
        count += len(d)
    print("Fill: %f (%d/%d)." % (float(count) / (len(data) * D), count, len(data)*D))

if __name__ == '__main__':
#    featureDim = int(sys.argv[1])
#    data, labels = load_data(sys.argv[2])
#    data, labels = normalize(data, labels, featureDim)
    
    data, labels = load_data(sys.argv[1])
    print 'Done loading data.'
    D = int(sys.argv[3])
    fillCount(data, D)
    data = normalize_features(data, D)
#    labels = normalize_labels(labels)
    data, labels = randomize(data, labels)
    print 'Done normalizing & randomizing data.'
    statistics(data, labels)            #Only for binary classification
    
    write_file(data, labels, sys.argv[2])
    sys.exit()