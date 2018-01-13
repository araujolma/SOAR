#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:23:57 2017

@author: levi
"""

import numpy, itsme
import matplotlib.pyplot as plt


t, x, u, alfa, beta, con, tphases, m0, mJ = itsme.sgra('default3st.its')
#plt.plot(t,x[:,3])
#plt.grid(True)
#plt.xlabel("t")
#plt.ylabel("Mass")
#plt.show()

alfaList = numpy.empty_like(t)
for i in range(len(t)):
    alfaList[i] = alfa.value(t[i])
    
plt.plot(t,alfaList)
plt.grid()
plt.show()

s = con['NStag']
arcBginIndx = numpy.empty(s)
arc = 0; arcBginIndx[arc] = 0
j = 0; nt = len(t)
for i in range(len(mJ)-1):
    print("i =",i)
    if mJ[i] > 0.0:
        print("Jettissoned mass found!")
        arc += 1
        print("arc =",arc)
        tTarg = tphases[i]
        print("Beginning search for tTarg =",tTarg)
        keepLook = True
        while (keepLook and (j < nt)):
            if abs(t[j]-tTarg) < 1e-10:
                keepLook = False
                arcBginIndx[arc] = j
                print("Found it! in j =",j)
            print("j =",j,"t[j] =",t[j])
            j += 1

