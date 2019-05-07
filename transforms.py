import numpy as np 
import cv2
import random

class Normalization(object):
    '''
    This operation only afffects X, does not affect Y. 
    C,X,Y,Z
    '''
    def __call__(self, X, Y=None):
        c,d,h,w = X.shape
        X_norm =  (255*(X - X.min()) / (X.max() - X.min())).astype(np.uint8)
        
        img_2d = X_norm.reshape(d,h*w).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equ = clahe.apply(img_2d)
        img_out = equ.reshape(c,d,h,w).astype(np.float32)
        
        return img_out, Y


class RandomFlip(object):
    def __call__(self, X, Y):
        ran = random.randint(0,9)
        if ran == 0:
            return X.transpose(1,2), Y.transpose(1,2)
        elif ran == 1:
            return X.transpose(2,3), Y.transpose(2,3)
        elif ran == 2:
            return  X.transpose(1,3), Y.transpose(1,3)
        else:
            return X, Y


def normlize(X):
    '''
    takes np D H W  
    out np D H W
    '''
    out, _ = Normalization()(X[np.newaxis, :,:,:], None)
    return out[0]