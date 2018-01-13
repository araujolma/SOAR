#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:55:59 2017

@author: levi
"""

##############################################################################
## Exemplo 1
#from multiprocessing import Process
#import os
#
#def info(title):
#    print('\n> '+title)
#    print('module name:', __name__)
#    print('parent process:', os.getppid())
#    print('process id:', os.getpid())
#
#def f(name):
#    info('function f')
#    print('Hello, Mr. '+name+'!')
#
#if __name__ == '__main__':
#    info('main line')
##    p = Process(target=f, args=('Bob','Moreira','Araújo'))
#    p = Process(target=f, args=('Bob',))
#    p.start()
#    p.join()

##############################################################################
## Exemplo 2
#import multiprocessing as mp
#
#def foo(q):
#    q.put('hello')
#
#if __name__ == '__main__':
#    ctx = mp.get_context('spawn')
#    q = ctx.Queue()
#    p = ctx.Process(target=foo, args=(q,))
#    p.start()
#    print(q.get())
#    p.join()

##############################################################################
## Exemplo 3
#from multiprocessing import Process, Queue
#
#def f(q):
#    q.put([42, None, 'hello'])
#
#if __name__ == '__main__':
#    q = Queue()
#    p = Process(target=f, args=(q,))
#    p.start()
#    print(q.get())    # prints "[42, None, 'hello']"
#    p.join()

##############################################################################
## Exemplo 4
#from multiprocessing import Pool, TimeoutError
#import time
#import os
#
#def f(x):
#    return x*x
#
#if __name__ == '__main__':
#    # start 4 worker processes
#    with Pool(processes=4) as pool:
#
#        print("\nParte 1:")
#        # print "[0, 1, 4,..., 81]"
#        print(pool.map(f, range(10)))
#
#        print("\nParte 2:")
#        # print same numbers in arbitrary order
#        for i in pool.imap_unordered(f, range(10)):
#            print(i)
#
#        print("\nParte 3:")
#        # evaluate "f(20)" asynchronously
#        res = pool.apply_async(f, (20,))      # runs in *only* one process
#        print(res.get(timeout=1))             # prints "400"
#
#        print("\nParte 4:")
#        # evaluate "os.getpid()" asynchronously
#        res = pool.apply_async(os.getpid, ()) # runs in *only* one process
#        print(res.get(timeout=1))             # prints the PID of that process
#
#        print("\nParte 5:")
#        # launching multiple evaluations asynchronously *may* use more processes
#        multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
#        print([res.get(timeout=1) for res in multiple_results])
#
#        print("\nParte 6:")
#        # make a single worker sleep for 10 secs
#        res = pool.apply_async(time.sleep, (10,))
#        try:
#            print(res.get(timeout=1))
#        except TimeoutError:
#            print("We lacked patience and got a multiprocessing.TimeoutError")
#
#        print("For the moment, the pool remains available for more work.")
#
#    # exiting the 'with'-block has stopped the pool
#    print("Now the pool is closed and no longer available.")


###############################################################################
### FUNCIONOU!
#import numpy, time
#from multiprocessing import Pool
#
#def integ(inpArg):
#    coefList = inpArg['coefList']
#    a = inpArg['a']
#    b = inpArg['b']
#    
#    S = 0.0
#
#    dt = (b-a)/1e7
#    t = numpy.arange(a,b,dt)
#    k = -1
#    for coef in coefList:
#        k += 1
#        thisS = 0.0
#        v = coef*(t**k)
#        thisS = v.sum()
#        thisS -= .5*(v[0]+v[-1])
#        thisS *= dt
#        S += thisS
#    return S
#
#if __name__ == "__main__":
#
#    d1 = {'coefList':None,'a':0.0, 'b':1.0}
#    N = 100
#    randCoef = numpy.random.rand(N,4)
#    d = []
#    for k in range(N):
#        thisD = d1.copy()
#        thisD['coefList'] = randCoef[k,:]
#        d.append(thisD.copy())
#    
#    print("\nPrimeira abordagem:")
#    t1 = time.time()
#    print(t1)
#    outp = [integ(thisDict) for thisDict in d]
#    t2 = time.time()
#    print(t2)
#
#    print("Tempo:",round(t2-t1,3))
#    
#    print("\nSegunda abordagem:")
#    t1 = time.time()
#    print(t1)
#    pool = Pool()
#    outp = pool.map(integ, d)
#    pool.close()
#    pool.join()
#    t2 = time.time()
#    print(t2)
#    print("Tempo:",round(t2-t1,3))

###############################################################################
### objectification: it worked!
#import numpy, time
#from multiprocessing import Pool
#
#class Tert():
#    def __init__(self,coefList,a,b):
#        self.coefList = coefList
#        self.a = a
#        self.b = b
#        
#    def integ(self,j):
#        thisCoefList = self.coefList[j,:]
#        a = self.a
#        b = self.b
#    
#        S = 0.0
#
#        dt = (b-a)/1e7
#        t = numpy.arange(a,b,dt)
#        k = -1
#        for coef in thisCoefList:
#            k += 1
#            thisS = 0.0
#            v = coef*(t**k)
#            thisS = v.sum()
#            thisS -= .5*(v[0]+v[-1])
#            thisS *= dt
#            S += thisS
#        return S
#
#if __name__ == "__main__":
#
#    N = 100
#    randCoef = numpy.random.rand(N,4)
#    obj = Tert(randCoef,0.0,1.0)
#    
#    print("\nPrimeira abordagem:")
#    t1 = time.time()
#    print(t1)
#    outp = [obj.integ(j) for j in range(N)]
#    t2 = time.time()
#    print(t2)
#
#    print("Tempo:",round(t2-t1,3))
#    
#    print("\nSegunda abordagem:")
#    t1 = time.time()
#    print(t1)
#    pool = Pool()
#    outp = pool.map(obj.integ, range(N))
#    pool.close()
#    pool.join()
#    t2 = time.time()
#    print(t2)
#    print("Tempo:",round(t2-t1,3))

##############################################################################
## objectification: it worked!
import numpy, time
from multiprocessing import Pool

class Tert():
    def __init__(self,coefList,a,b):
        self.coefList = coefList
        self.n = coefList.shape[0]
        self.a = a
        self.b = b
        
        self.S = numpy.zeros(self.n)
        
    def integ(self,j):
        thisCoefList = self.coefList[j,:]
        a = self.a
        b = self.b
    
        S = 0.0

        dt = (b-a)/1e7
        t = numpy.arange(a,b,dt)
        k = -1
        for coef in thisCoefList:
            k += 1
            thisS = 0.0
            v = coef*(t**k)
            thisS = v.sum()
            thisS -= .5*(v[0]+v[-1])
            thisS *= dt
            S += thisS
        
        self.S[j] = S
        print(self.S)
        return S

if __name__ == "__main__":

    #N = 100
    #randCoef = numpy.random.rand(N,4)
    N = 3
    randCoef = numpy.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    obj = Tert(randCoef,0.0,1.0)
    obj2 = Tert(randCoef,0.0,1.0)
    
    print("\nPrimeira abordagem:")
    t1 = time.time()
    print(t1)
    for j in range(N):
        obj.integ(j)
    #outp = [obj.integ(j) for j in range(N)]
    t2 = time.time()
    print(t2)

    print("Tempo:",round(t2-t1,3))
    
    print("\nSegunda abordagem:")
    t1 = time.time()
    print(t1)
    pool = Pool()
    #outp = pool.map(obj.integ, range(N))
    results = pool.map(obj2.integ,range(N))
    pool.close()
    pool.join()
    obj2.S = results
    t2 = time.time()
    print(t2)
    print("Tempo:",round(t2-t1,3))
    
    print("Erro =",max(obj.S-obj2.S))