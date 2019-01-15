# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:30:43 2019

@author: zhangtie
"""
import numpy as np
from scipy.linalg import solve   #solve linear equations

class Rewei_Lp:
    def __init__(self,k,lam,p,tole,alpha,smoonth):
        self.num=[]
        self.dim=[]
        self.lam=lam
        self.p=p                          #Lp(0<p<1)
        self.tolerance=tole
        self.alpha=alpha                  # 0<alpha<1
        self.smoonth=smoonth              #smoonth parameter
        self.k=k                          #expected number of nonzero
    def train(self,A,b):
        self.num=np.size(A,0)
        self.dim=np.size(A,1)
        max_itera=20                      #max number of iteration
        x=np.zeros(self.dim)
        
        for i in np.arange(max_itera):
            #solve reweigthed equation
            w=(self.p*self.lam)/(np.power(self.smoonth**2+x**2,1-self.p/2))
            diag_w=np.diag(w)
            x=solve(np.dot(A.T,A)+diag_w,np.dot(A.T,b))
            dec_x=-np.sort(-x)
            self.smoonth=np.min([self.smoonth,self.alpha*dec_x[self.k]])
            print(self.smoonth)
        return x  
            
           
