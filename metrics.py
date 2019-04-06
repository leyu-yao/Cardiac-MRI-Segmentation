# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:13:32 2019

@author: Dynasting

@reference:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
"""
import numpy as np
from sklearn.metrics import roc_auc_score, auc, roc_curve
from scipy import interp

class Metric_AUC():
    def __init__(self, curve=False, average='micro'):
        self.curve_required = curve
        self.average = average

    
    
    def __call__(self, y_score, y_test):
        '''
        @input
        pred tensor in (N,C,*)
        target tensor in (N,C,*)
        
        @return vaule
        auc in float
        
        To do:
        将fpr,tpr,roc_auc初始化保留，由各函数计算其中的项，根据参数选择是否计算
        目前之前返回
        
        考虑是否集成作图功能及优化参数选择方式，目前不作图
        '''
        # get n_classes and batch-size N
        batch_size = y_score.shape[0]
        n_classes = y_score.shape[1]
        
        # to numpy
        y_score = y_score.numpy()
        y_test = y_test.numpy()
        
        # to shape [n_samples, n_classes]
        def _reshape(inp, N, C):
            return inp.swapaxes(0,1).reshape(C, -1).swapaxes(0,1)
        y_score = _reshape(y_score, batch_size, n_classes)
        y_test = _reshape(y_test, batch_size, n_classes)
        
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # return value
        if self.average == "micro":
            return roc_auc["micro"]
        else:
            return roc_auc["macro"]

if __name__ == '__main__':
    
    import torch
    from sklearn.preprocessing import label_binarize
    y=torch.rand(4,2*3*3*3)
    y=label_binarize(y.argmax(dim=0).numpy(),classes=[0,1,2,3]).reshape(4,2,3,3,3)
    y=torch.from_numpy(y.swapaxes(0,1).astype(np.float32))
    
    
    noise=5*torch.rand(2,4,3,3,3)
    
    y_p=y + noise
    
    Auc=Metric_AUC()
    
    r = Auc(y_p, y)