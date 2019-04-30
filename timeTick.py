# -*- coding: utf-8 -*-

"""
This module implements a timer to count execution time. 
Usage: Add @timeTicker above a function is defined. 

Example:

@timeTicker
def foo():
    pass



"""
import time


def timeTicker(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        back = func(*args, **kwargs)
        t1 = time.time()
        t_exe = t1 - t0
        print("%s takes %.6f" % (func.__name__, t_exe))
        return back
    return wrapper
# --end of exeTime
 
@timeTicker
def foo():
    for i in range(10000000):
        pass
 
if __name__ == "__main__":
    foo()
