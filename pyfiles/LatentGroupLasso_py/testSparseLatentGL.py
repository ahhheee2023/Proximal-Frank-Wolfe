# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:19:58 2024

@author: zly
"""

from math import floor
import numpy as np
from numpy.linalg import norm 
from numpy.random import randn
from numpy.random import rand
import random
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import sparseLatentGL

random.seed(10)

nf = 0.01    # noise factor
sr = 0.05   # sparsity ratio
index = 1
gs = 50*index    # now we set the size of all groups equal and the size of each group is 50
os = 5*index     # the size of the overlapping in grou i and i+1
K = 100    # K groups
n = (gs-os)*K + os
m = floor( (gs-os)*K/2 )

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
A = At.T

error = nf *rand(m)   # Gaussian noise
b = A @ xorig + error

mu = 0.1 # the parameter of the sparsity regularizer; need to do cross-validation fot mu

opts = {}
opts['verbose_freq']  = 1000
opts['maxiter'] = 5000
u, s, v = svds(A, 1)   # eigs(A@A.T) gives a complex number in python...
L_A = s**2
opts['L_A'] = L_A


opts['xinit'] = np.zeros(n)
x,y,itr,history = sparseLatentGL.FW_sparselgl(A, b, mu, c1, K, Grps, opts)

plt.figure(dpi=800)
plt.plot(np.arange(0,n),xorig,'ro', np.arange(0,n),x,'b*')
plt.xlim(0,n)
plt.show()

fval_rec = history['fval_rec']
plt.figure(dpi=800)
plt.plot(np.arange(0,itr),fval_rec,'b-',linewidth=1.5)
plt.xlabel('Iterations')
plt.ylabel('fval')
plt.xlim(0,itr)
plt.show()

FWgap_rec = history['FWgap_rec']
plt.figure(dpi=800)
plt.plot(np.arange(0,itr),FWgap_rec,'b-',linewidth=1.5)
plt.xlabel('Iterations')
plt.ylabel('FWgap')
plt.xlim(0,itr)
plt.show()





    
    