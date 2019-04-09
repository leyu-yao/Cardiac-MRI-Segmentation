# -*- coding: utf-8 -*-
"""
roc calculation in sk-learn is extremely slow. 
This module is implementation of auc, roc_auc_score, roc_curve. 

roc_auc_score, auc, roc_curve
"""
import numpy as np 
from sklearn.metrics import auc
from timeTick import timeTicker

@timeTicker
def roc_curve(y_true, y_score, points=20):
    '''
    input numpy.array in shape [n_samples]

    return value
    fpr, tpr, thresholds [>2]
    '''
    fpr = []
    tpr = []
    thresholds = np.percentile(y_score, [x for x in range(100,-5,-5)])

    for thr in thresholds:
        y_copy = y_score.copy()
        y_copy[y_copy<=thr] = 0
        y_copy[y_copy>thr] = 1

        TP = np.dot(y_true, y_copy)
        FP = np.dot(y_copy, 1 - y_true)
        FN = np.dot(y_true, 1 - y_copy)
        TN = np.dot(1 - y_copy, 1-y_true)
        
        #print(TP, FP)
        #print(FN, TN)
        #print('-'*10)
        
        tpr.append(TP / (TP + FN))
        fpr.append(FP / (FP + TN))


    return np.array(fpr), np.array(tpr), thresholds

if __name__ == "__main__":
    from sklearn import metrics
    
    y = np.random.randint(0,2,5000000)
    scores = y + 10*np.random.rand(5000000)

    @timeTicker
    def sk_roc_curve(y, scores):
        return metrics.roc_curve(y, scores)
    
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.plot(*sk_roc_curve(y, scores)[0:2])
    plt.subplot(1,2,2)
    plt.plot(*roc_curve(y, scores)[0:2])
    plt.show()
    
    from time import time
    t0 = time()
    auc(*sk_roc_curve(y, scores)[0:2])
    t1 = time()
    print("sk %.3f"%(t1-t0))
    
    auc(*roc_curve(y, scores)[0:2])
    t2 = time()
    print("my %.3f"%(t2-t1))
    
    #print(sk_roc_curve(y, scores))

    #print(roc_curve(y, scores))

