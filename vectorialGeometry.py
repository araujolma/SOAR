#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:44:34 2018

Module for 3D geometric vectors
Created to be fast and simple

@author: carlos
"""

from math import sqrt

    
class vector():
        
    def __init__(self, x: float, y: float, z: float):
        
        self.x = x
        self.y = y
        self.z = z
    
    def dot(self, vec2)-> float:
        
        return self.x*vec2.x + self.y*vec2.y + self.z*vec2.z

    def cross(self, vec2):
        
        return vector(self.y*vec2.z - self.z*vec2.y, 
                      self.z*vec2.x - self.x*vec2.z, 
                      self.x*vec2.y - self.y*vec2.x)
    
    def __mul__(self, s):
        
        return vector(self.x*s, self.y*s, self.z*s)
    
    def __rmul__(self, s):
        
        return vector(self.x*s, self.y*s, self.z*s)

    def __truediv__(self, s):
        
        return vector(self.x/s, self.y/s, self.z/s)

    def module(self)-> float:
        
        return sqrt(self.dot(self))

    def module2(self)-> float:
        
        return self.dot(self)
    
    def module4(self)-> float:
        
        return self.dot(self)**2

    def __add__(self, vec2):
        
        return vector(self.x + vec2.x, self.y + vec2.y, self.z + vec2.z)
    
    def __sub__(self, vec2):
        
        return vector(self.x - vec2.x, self.y - vec2.y, self.z - vec2.z)

    def __iadd__(self, vec2):
        
        self.x += vec2.x
        self.y += vec2.y
        self.z += vec2.z
        return self
    
    def __isub__(self, vec2):
        
        self.x -= vec2.x
        self.y -= vec2.y
        self.z -= vec2.z
        return self

    def opposite(self):
        # reflexive opposite
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        return None
       
    def __neg__(self):
        
        return vector(-self.x, -self.y, -self.z)
       
    def __pos__(self):
        
        return self
   
    def __repr__(self):
        
        return 'v(%f, %f, %f)' % (self.x, self.y, self.z)
    
    def reset(self, x: float, y: float, z: float)-> None:
        
        self.x = x
        self.y = y
        self.z = z
        
        return None

    def versor(self):
        
        return self/self.module()


class vector0(vector):
        
    def __init__(self):
        
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        

if __name__=='__main__':

    # Tests
    
    a = vector(0., 1., 2.)
    b = vector(4., 1., 2.)
    
    x = vector(1, 0, 0)
    y = vector(0, 1, 0)
    
    z = vector0()
    
    print('vector0 :', z)
    print('vector0 + a:', z+a)
    print('dot: ', a.dot(b))
    print('cross: ', x.cross(y))
    print('sum:', a+b)
    print('scalar mult.:', 2*a-b)
    print('scalar div. and neg.:',-b/2)
    a -= b
    print('-= :', a)
    print('positive :', +b)
    print('versor:', b.versor())