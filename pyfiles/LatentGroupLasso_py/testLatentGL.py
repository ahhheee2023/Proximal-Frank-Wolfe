# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:50:26 2024

@author: zly
"""

from math import floor
import numpy as np
from numpy.linalg import norm 
from numpy.random import randn
from numpy.random import rand
import random
from scipy.sparse.linalg import svds
import latentGL

random.seed(10)

nf = 0.01    # noise factor
sr = 0.05   # sparsity ratio
index = 1
gs = 50    # now we set the size of all groups equal and the size of each group is 50
os = 5     # the size of the overlapping in grou i and i+1
K = 100    # K groups
n = (gs-os)*K + os
m = floor( (gs-os)*K/2 )*index

Grps = np.empty((K),dtype=object)   # cell: contain the index of the groups
for k in range(0,K,1):
    Grps[k] = np.arange((gs-os)*k, (gs-os)*k+gs,1)
Irp = np.random.permutation(K)
J = Irp[np.arange(floor(K*sr))]

xorig = np.zeros(n)
lglnm_xorig = 0
ss_orig = np.empty((len(J)),dtype=object)
for j in range(0,len(J),1):
    k = J[j]
    IdGk = Grps[k]
    sk_orig = randn(len(IdGk))
    ss_orig[j] = sk_orig
    xorig[IdGk] = xorig[IdGk] + sk_orig
    lglnm_xorig = lglnm_xorig + norm(sk_orig)
    
c1 = 0.95*lglnm_xorig
At = randn(n,m)
for j in range(0,n,1):
    At[j] = At[j]/norm(At[j])
A = np.transpose(At)

error = nf * ( rand(m)*rand(m) - rand(m)*rand(m) )  # Laplacian noise
b = A @ xorig + error

opts = {}
opts['verbose_freq']  = 100
opts['maxiter'] = 5000
u,L,vh = svds(A,1)
opts['noiseLev'] = L*c1 + norm(b) 

opts['xinit'] = np.zeros(n)
x,y,itr,history = latentGL.FW_lgl(A,b,c1,K, Grps, opts)


    
    