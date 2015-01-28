'''
Created on May 14, 2012

@author: Tom de Ruijter
'''

import math

def RMSE(labels, predictions): 

    #count of postive and negative labels
    assert len(labels) == len(predictions)
    rmse = 0.
    for i in xrange(len(labels)):
        try:
            rmse += (float(labels[i])-predictions[i])**2
        except OverflowError:
            return 1.e100
    return math.sqrt(rmse/len(labels))

def AUC(labels, predictions):
    #count of postive and negative labels
    db = []
    pos, neg = 0, 0
    
    # Count labels
    for i in xrange(len(labels)):
        if labels[i]>0:
            pos+=1
        else:    
            neg+=1
        db.append([predictions[i], labels[i]])    

    db.sort(lambda x,y: x[0] < y[0] and 1 or -1)
    
    #calculate ROC 
    xy_arr = []
    tp, fp = 0., 0.            #assure float division
    for i in xrange(len(db)):
        if db[i][1]>0:        #positive
            tp+=1
        else:
            fp+=1
            
        if neg == 0:
            s = 0
        else:
            s = fp/neg
        
        if pos == 0:
            t = 0
        else:
            t = tp/pos
        
        xy_arr.append([s,t])
    
    #area under curve
    auc = 0.
    prev_x = 0
    for x,y in xy_arr:
        if x != prev_x:
            auc += (x - prev_x) * y
            prev_x = x
    return auc
 

def accuracy(labels,predictions):
    correct = 0
    total = 0

    for i in xrange(len(labels)):
        if labels[i] == predictions[i]:
            correct +=1
        total += 1
    
    return correct / float(total)  