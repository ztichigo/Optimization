# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:26:52 2019

@author: zhangtie
"""
import numpy as np
import Spa_vector
from gene_mat import Gene

num=64             #number of data
dim=128            #dimension of data
k=20               #number of nonzero
testnum=50         #number of test
lam=1e-6           #regularization parameter
p=0.5              #Lp(0<p<1)
tole=1e-5
alpha=0.9
smoonth=1          #smoonth parameter

Model_vec=Spa_vector.Rewei_Lp(k,lam,p,tole,alpha,smoonth)

Gene_data=Gene()                                #generate data for experiment
A,x0,b=Gene_data.gene_for_sparse(num,dim,k)     #data for sparse vector recovery 

x_res=Model_vec.train(A,b)
