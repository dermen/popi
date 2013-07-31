from __future__ import division
import numpy as np
ma = np.ma
import pylab as plt
import csv, warnings, copy, os, operator

#from scipy.optimize import curve_fit

def self_pca(intra,inter,num_inter=0,numVals=0,numHistBins=50,pc_comp=0,title=""):
    """ 
    Searches for a PCA basis within the intra- and inter- shot
    data set and projects the intras and inters onto the desired principal
    component (first principal component by default).
    Parameters
    ----------
    intra : 2darray of floats; axis 0 = shot index; axis 1 = delta phi index
    inter : 2darray of floats; axis 0 = shot index; axis 1 = delta phi index
    Optional
    --------
    numVals     : int, number of fft coefficients to include in the PCA matrix
    numHistBins : int, number of histogram bins in the plot
    pc_comp     : int, principal component to project onto (0 is the first princ component)
    """
    if numVals == 0:
        numVals = intra.shape[1]

    intra_fft = np.fft.fft( intra[:,0:], axis=1 )
    inter_fft = np.fft.fft( inter[:,0:], axis=1 )

    intra_fft = np.abs( intra_fft )
    inter_fft = np.abs( inter_fft )

    plt.figure(2)
    plt.plot(intra_fft.mean(axis=0) ) 
    plt.plot(inter_fft.mean(axis=0)) 
    
    intra_fft = intra_fft[:,0:numVals]
    inter_fft = inter_fft[:,0:numVals]

    all_ffts = np.vstack(( intra_fft,inter_fft ))

    all_PCA = PCA(all_ffts)
    fracs   = all_PCA.fracs # variance along each component
    Wt      = all_PCA.Wt # weights of each component

#   project the intras onto the 0th PC
    intra_PC = np.zeros(intra_fft.shape[0])
    for i in xrange(intra_fft.shape[0]):
        intra_PC[i] = np.sum(intra_fft[i,:] * Wt[pc_comp,:])
    inter_PC = np.zeros(inter_fft.shape[0])
    for i in xrange(inter_fft.shape[0]):
        inter_PC[i] = np.sum(inter_fft[i,:] * Wt[pc_comp,:])

#   plot the results
    plt.figure(1)
    plt.suptitle(title,fontsize=22)
    plt.subplot(121)
#   plt.hist(intra_PC,bins=numHistBins,histtype="stepfilled",color='m',label="intRA-shot")
#   plt.hist(inter_PC,bins=numHistBins,histtype="stepfilled",color='b',alpha="0.5",label="intER-shot")
    plt.hist(intra_PC,bins=numHistBins,histtype="step",normed=True,lw=2,label="intRA-shot")
    plt.hist(inter_PC,bins=numHistBins,histtype="step",normed=True,lw=2,label="intER-shot")

    plt.ylabel("number of counts",fontsize=22)
    plt.xlabel("projection onto principal component" + str(pc_comp+1),fontsize=22)
    plt.legend()

    plt.subplot(122)
    plt.xlabel("principal component vectors",fontsize=22)
    plt.ylabel("percentage variation",fontsize=22)
    plt.plot(range(len(fracs)),fracs,"bo",markersize=8)
    plt.plot(range(len(fracs)),fracs,"b--",linewidth=2)
    plt.xlim([-2,numVals - 0.5])

    plt.show()

    return intra_PC,inter_PC,Wt

## borrowed from matplotlib

def kl_calc(dat1,dat2,numHistBins=75,bins1 = [],bins2 = []):
    """
    Finds kl div between 2 datasets and returns a number.

    Parameters
    ----------
    dat1 : numpy 1darray, floats, first set of data points
    dat2 : numpy 1darray, floats, seconds set of data points

    Optional
    --------
    numHistBins : int, number of bins in each histogram
    bins1       : array floats, bin edges of first dataset histogram
    bins2       : array floats, bin edges of second dataset histogram
    Returns
    ------
    KL divergence : float  

    """

    if bins1 == []:
        low_bin  = np.min([dat1.min(), dat2.min() ])
        high_bin = np.max([dat1.max(), dat2.max()] )
        bins1 = np.linspace(low_bin,high_bin,numHistBins)
        bins2 = bins1

    h1,b1 = np.histogram(dat1,bins=bins1,normed=True)
    h2,b2 = np.histogram(dat2,bins=bins2,normed=True)
    h1 = h1/ np.sum(h1)
    h2 = h2/ np.sum(h2)

    h1_good  = np.where(h1 > 0)[0] # non zero indices
    h2_good  = np.where(h2 > 0)[0] # non zero indices
    h12_good = h1_good[np.in1d(h1_good,h2_good)] # finds common non zero indices

    kl = np.log(h1[h12_good]/h2[h12_good]) * h1[h12_good] 
    return np.sum(kl)
      
def kl_div(intra,inter,num_inter=0,numVals=0,numHistBins=75,pc_comps=[]):
    """
    Finds the Kullback-Leibler divergence between intRA-shot and intER-shot principle
    component distributions.

    Parameters
    ----------
    intra : 2darray of floats; axis 0 = shot index; axis 1 = fft coefficient
    inter : 2darray of floats; axis 0 = shot index; axis 1 = fft coefficient
    Optional
    --------
    numVals     : int, number of fft coefficients to include in the PCA matrix
    numHistBins : int, number of histogram bins in the plot
    pc_comps    : list of int, principal components to project onto (0 is the first princ component)
                  the return is a sum of the KL for the resulting distributions
    Returns
    -------
    float, the KL divergence
    """
    if numVals == 0:
        numVals = intra.shape[1]
    if pc_comps == []:
        pc_comps = [0]

    intra_fft = np.fft.fft( intra[:,0:], axis=1 )
    inter_fft = np.fft.fft( inter[:,0:], axis=1 )

    intra_fft = np.abs( intra_fft )
    inter_fft = np.abs( inter_fft )

    intra_fft = intra_fft[:,0:numVals]
    inter_fft = inter_fft[:,0:numVals]

    all_ffts = np.vstack(( intra_fft, inter_fft ))

    all_PCA = PCA(all_ffts)
    fracs   = all_PCA.fracs # variance along each component
    Wt      = all_PCA.Wt # weights o f each component

    kl = 0
    for pc_comp in pc_comps:
        intra_PC = np.zeros(intra_fft.shape[0])
        for i in xrange(intra_fft.shape[0]):
            intra_PC[i] = np.sum(intra_fft[i,:] * Wt[pc_comp,:])
        inter_PC = np.zeros(inter_fft.shape[0])
        for i in xrange(inter_fft.shape[0]):
            inter_PC[i] = np.sum(inter_fft[i,:] * Wt[pc_comp,:])
        kl += kl_calc(intra_PC,inter_PC,numHistBins=numHistBins)
    return kl

def bootleg(dat,numSamp = 0):
    """
    bootstrap sample the ndarray_float32
    Paramters
    ---------
    dat     :  numpy float32 array (axis 0 = shot index, axis 1 = correlation index  
    numSamp : int, number of samples in bootstrap
    Returns
    -------
    numpy 1d array float 32 (average bootstrap correlation)
    """
    if numSamp == 0:
        numSamp = dat.shape[0]
    shot_inds = np.random.randint(0,dat.shape[0],numSamp)
    return dat[shot_inds]#.mean(axis=0)

 
## from matplotlib mlab

class PCA:
    def __init__(self, a):
        """
        compute the SVD of a and store data for PCA.  Use project to
        project the data onto a reduced set of dimensions

        Inputs:

          *a*: a numobservations x numdims array

        Attrs:

          *a* a centered unit sigma version of input a

          *numrows*, *numcols*: the dimensions of a

          *mu* : a numdims array of means of a

          *sigma* : a numdims array of atandard deviation of a

          *fracs* : the proportion of variance of each of the principal components

          *Wt* : the weight vector for projecting a numdims point or array into PCA space

          *Y* : a projected into PCA space


        The factor loadings are in the Wt factor, ie the factor
        loadings for the 1st principal component are given by Wt[0]

        """
        n, m = a.shape
        if n<m:
            raise RuntimeError('we assume data in a is organized with numrows>numcols')

        self.numrows, self.numcols = n, m
        self.mu = a.mean(axis=0)
        self.sigma = a.std(axis=0)

        a = self.center(a)

        self.a = a

        U, s, Vh = np.linalg.svd(a, full_matrices=False)


        Y = np.dot(Vh, a.T).T

        vars = s**2/float(len(s))
        self.fracs = vars/vars.sum()


        self.Wt = Vh
        self.Y = Y


    def project(self, x, minfrac=0.):
        'project x onto the principle axes, dropping any axes where fraction of variance<minfrac'
        x = np.asarray(x)

        ndims = len(x.shape)

        if (x.shape[-1]!=self.numcols):
            raise ValueError('Expected an array with dims[-1]==%d'%self.numcols)


        Y = np.dot(self.Wt, self.center(x).T).T
        mask = self.fracs>=minfrac
        if ndims==2:
            Yreduced = Y[:,mask]
        else:
            Yreduced = Y[mask]
        return Yreduced



    def center(self, x):
        'center the data using the mean and sigma from training set a'
        return (x - self.mu)/self.sigma



    @staticmethod
    def _get_colinear():
        c0 = np.array([
            0.19294738,  0.6202667 ,  0.45962655,  0.07608613,  0.135818  ,
            0.83580842,  0.07218851,  0.48318321,  0.84472463,  0.18348462,
            0.81585306,  0.96923926,  0.12835919,  0.35075355,  0.15807861,
            0.837437  ,  0.10824303,  0.1723387 ,  0.43926494,  0.83705486])

        c1 = np.array([
            -1.17705601, -0.513883  , -0.26614584,  0.88067144,  1.00474954,
            -1.1616545 ,  0.0266109 ,  0.38227157,  1.80489433,  0.21472396,
            -1.41920399, -2.08158544, -0.10559009,  1.68999268,  0.34847107,
            -0.4685737 ,  1.23980423, -0.14638744, -0.35907697,  0.22442616])

        c2 = c0 + 2*c1
        c3 = -3*c0 + 4*c1
        a = np.array([c3, c0, c1, c2]).T
        return a
