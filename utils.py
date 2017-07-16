# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 18:53:01 2016

@author: levi
"""
import numpy

def interpV(t,tVec,xVec):
#	print("In interpV!")
#	print(numpy.shape(tVec))
#	print(numpy.shape(xVec))	
	#print("t =",t,"tVec =",tVec,"xVec =",xVec)
#	xVec = numpy.squeeze(xVec)
	
	Nsize = xVec.shape[1]

	ans = numpy.empty(Nsize)
	for k in range(Nsize):
		ans[k] = numpy.interp(t,tVec,xVec[:,k])
	return ans

def interpM(t,tVec,xVec):

	ni = xVec.shape[1]
	nj = xVec.shape[2]

	ans = numpy.empty((ni,nj))
	for i in range(ni):
		for j in range(nj):
			ans[i,j] = numpy.interp(t,tVec,xVec[:,i,j])
	return ans

def ddt(vec,N):

	dt = 1.0/(N-1)
	dvec = numpy.empty_like(vec)
	dvec[0] = (vec[1]-vec[0])/dt
	dvec[N-1] = (vec[N-1]-vec[N-2])/dt
	for k in range(1,N-1):
		dvec[k] = .5*(vec[k+1]-vec[k-1])/dt

	return dvec

def testAlgn(x,y):
    A = numpy.ones((3,3))
    A[:,1] = x
    A[:,2] = y
    return numpy.linalg.det(A)

if __name__ == "__main__":
    print("In utils.py!")
    print("Testing testAlgn:")
    
    x = numpy.array([1.0,2.0,3.0])
    y = 5.0*x
    print(testAlgn(x,y))
    x[0]=0.0
    print(testAlgn(x,y))
    
    