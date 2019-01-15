# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:33:48 2019

@author: zhangtie
"""
import numpy as np

class Gene:
    def __init(self):
        return 0
    def gene_for_sparse(self,num,dim,k):
      A=np.random.randn(num,dim)
      x=np.zeros(dim)
      ind=np.random.permutation(dim)
      x[ind[0:k]]=np.random.randn(k)
      b=np.dot(A,x)
      return A,x,b
  
