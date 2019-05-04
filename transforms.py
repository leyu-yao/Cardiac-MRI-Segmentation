import numpy as np 
import cv2

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
