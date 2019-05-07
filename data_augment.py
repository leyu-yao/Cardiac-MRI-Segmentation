import numpy as np 
from scipy import ndimage
from timeTick import timeTicker

#@timeTicker
def rotate(x, y):
    res = []
    res.append((x,y))

    x1 = ndimage.rotate(x, 30.0, axes=(1,2), reshape=False, mode='nearest')
    y1 = ndimage.rotate(y, 30.0, axes=(1,2), reshape=False, mode='nearest')

    res.append((x1, y1))

    x2 = ndimage.rotate(x, -30.0, axes=(1,2), reshape=False, mode='nearest')
    y2 = ndimage.rotate(y, -30.0, axes=(1,2), reshape=False, mode='nearest')

    res.append((x2, y2))

    return res


#@timeTicker
def transpose(x, y):
    res = []
    res.append((x,y))

    x1 = np.transpose(x, axes=(0,1,3,2))
    y1 = np.transpose(y, axes=(0,1,3,2))
    res.append((x1, y1))

    x2 = np.transpose(x, axes=(0,2,1,3))
    y2 = np.transpose(y, axes=(0,2,1,3))
    res.append((x2, y2))

    x3 = np.transpose(x, axes=(0,3,1,2))
    y3 = np.transpose(y, axes=(0,3,1,2))
    res.append((x3, y3))

    x4 = np.transpose(x, axes=(0,2,3,1))
    y4 = np.transpose(y, axes=(0,2,3,1))
    res.append((x4, y4))


    x5 = np.transpose(x, axes=(0,3,2,1))
    y5 = np.transpose(y, axes=(0,3,2,1))
    res.append((x5, y5))

    return res

if __name__ == '__main__':
    x = np.random.rand(1,100,100,10)
    y = np.random.rand(8,100,100,10)
    transpose(x,y)
    rotate(x,y)
