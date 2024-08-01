# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:45:51 2024

@author: zly
"""

import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import svds


def FW_sparselgl(A, b, mu, c1, K, Grps,opts, gap_tol=1e-8):
    """
    This solves: min ||Ax-b||_2 + mu||x||_1  subject to Kappa(x) <= c_1,
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
    history = {}
    
    if 'L_A' in opts:
        L_A = opts['L_A']
    else:
        u, s, v = svds(A, 1)
        L_A = s**2
         
    
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
    y = x
    itr = 1
    
    print("%6s  %8s  %8s \n" % ('iter', 'fvale', 'FW_gap'))
    while 1==1:
        beta = np.sqrt(itr)
        Axb = A@x - b
        fval = norm(Axb) ** 2 / 2
        fval_rec = np.append(fval_rec, fval)
        
        # update of x by computing its soft thresholding and projection onto a ball ||x||<=c1
        v_tmp = L_A*y + beta * x - A.T@Axb
        xplus = np.absolute(v_tmp) - mu
        xtilde = ( np.sign(v_tmp) * np.maximum(xplus,0) ) / (beta+L_A)
        # x = np.min([c1/norm(xtilde), 1]) * xtilde
        # print(norm(xtilde))
        nm_xtld = norm(xtilde)
        if nm_xtld < c1:
            x = xtilde
        else:
            x = (c1/nm_xtld)*xtilde
        
        # update of y using the FW linear oracle
        z = beta * (y-x)
        norms = []
        for k in range(0,K):
            norms = np.append( norms, norm(z[Grps[k]]) )
            
        normmax = np.max(norms)
        gmax = np.argmax(norms)
        zGkmax = z[Grps[gmax]]
        sk = (c1/normmax)*zGkmax
        dneg = sk + y[Grps[gmax]]
        alpha = 2/(itr+2)
        y[Grps[gmax]] = y[Grps[gmax]] - alpha* dneg
        
        FWgap = zGkmax @ dneg
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
    
    history['fval_rec'] = fval_rec
    history['FWgap_rec'] = FWgap_rec
    
    return x, y, itr,history
        
            
        
            
            
        
    
        
    
    
            
    