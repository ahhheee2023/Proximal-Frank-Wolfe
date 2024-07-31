# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:35:25 2024

@author: zly
"""

import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import svds


def FW_lgl(A, b, c1, K, Grps,opts, gap_tol=1e-8):
    """
    This solves: min ||Ax-b||_1  subject to Kappa(x) <= c_1,
    where Kappa is the latent group norm
    
    Inputs:
        A: 2-D array with len(A[0])=n
        b: 1-D array with dimension m
        Grps: nparray, containg the index sets of the groups
        opts: a dictionary with keys noiseLev, xinit

    Returns: 
        x: 1-D array with dimension n
        iter: scalar
        history: a dictionary

    """
    n = len(A[0])
    fval_rec = []
    FWgap_rec = []
    history = np.empty((2),dtype=object)
    
    if 'noiseLev' in opts:
        c2 = opts['noiseLev']
    else:
        lambdaA = svds(A, k=1)
        c2 = lambdaA*c1 + norm(b)
    
    if 'xinit' in opts:
        xinit = opts['xinit']
    else:
        xinit = np.zeros(n)
        
    if 'verbose_freq' in opts:
        verbose_freq = opts['verbose_freq']
    else:
        verbose_freq = np.inf
        
    if 'maxiter' in opts:
        maxiter = opts['maxiter']
    else:
        maxiter = np.inf
     
    x = xinit
    itr = 1
    
    print("%6s  %8s  %8s \n" % ('iter', 'fvale', 'FW_gap'))
    while 1==1:
        beta = np.sqrt(itr)
        Axb = A@x - b
        fval = norm(Axb,1)
        fval_rec = np.append(fval_rec, fval)
        
        # update of y
        yplus = np.absolute(Axb) - 1/beta
        ytilde = np.sign(yplus) * np.maximum(yplus,0)
        # y = np.min([c2/norm(ytilde), 1]) * ytilde
        # print(norm(ytilde))
        nm_ytld = norm(ytilde)
        if nm_ytld < c2:
            y = ytilde
        else:
            y = (c2/nm_ytld)*ytilde
        
        # update of x using the FW linear oracle
        z = beta * (np.transpose(A) @ (Axb - y))
        norms = []
        for k in range(0,K):
            norms = np.append( norms, norm(z[Grps[k]]) )
            
        normmax = np.max(norms)
        gmax = np.argmax(norms)
        sk = (c1/normmax)*z[Grps[gmax]]
        dneg = sk + x[Grps[gmax]]
        alpha = 2/(itr+2)
        x[Grps[gmax]] = x[Grps[gmax]] -alpha* dneg
        
        FWgap = sk @ dneg
        FWgap_rec = np.append(FWgap_rec,FWgap)
        
        # # display
        if itr % verbose_freq == 0:
            print("%6d:  %8.2f   %8.2f \n"% (itr, fval, FWgap))
            
        # check termination
        if itr>=maxiter:
            print("Maximum iteration numbers used. \n")
            break
            
        if FWgap < gap_tol:
            print("The Frank-Wolfe gap is small enough. \n")
            break
        
        itr = itr +1
    
    history[0] = fval_rec
    history[1] = FWgap_rec
    
    return x, y, itr,history
        
            
        
            
            
        
    
        
    
    
            
    