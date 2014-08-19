import numpy as np
cimport numpy as np
import random
from scipy import fftpack

cdef extern from "corr.h":
  cdef cppclass Corr:
    Corr(int N_, float * ar1, float * ar2, float * ar3, float mask_val_) except +

cdef Corr * c

def correlate(A,B, mask_val = -1):
  """
  compute the correlation between 2 arrays
  Parameters:
  -----------
  A, 1D numpy array
  B, 1D numpy array
  mask_val, number value of each masked pixels ( defaults = -1 )
            they should all have the same value
  """
  if A.shape != B.shape:
    print "arrays must be of same size and shape" 
    return 0  # will add an traceback once I learn how :)
  N = A.shape[0]
  C = np.zeros_like(A)
  cdef np.ndarray[ndim=1,dtype=np.float32_t] v1
  cdef np.ndarray[ndim=1,dtype=np.float32_t] v2
  cdef np.ndarray[ndim=1,dtype=np.float32_t] v3
  
  v1 = np.ascontiguousarray(A.flatten(),dtype=np.float32)
  v2 = np.ascontiguousarray(B.flatten(),dtype=np.float32)
  v3 = np.ascontiguousarray(C.flatten(),dtype=np.float32)
  c  = new Corr(N,&v1[0], &v2[0], &v3[0], mask_val)
  del c
  return v3

def correlate_using_fft(self, x, y):
  """
  Compute the correlation between 2 arrays using the 
  convolution theorem. Works well for unmasked arrays.
  Passing masked arrays will results in numerical errors.

  Parameters
  ----------
  x : 1d numpy array of floats
      The intensities along ring 1
  y : 1d numpy array of floats
      The intensities along ring 2
    
  Returns
  -------
  iff : 1d numpy darray, float
      The correlation between x and y
  """

  #xm = np.ones_like(x)
  #ym = np.ones_like(y)
  #xm[np.where x == -1] = 0
  #ym[np.where y == -1] = 0

  #N_delta = np.zeros_like( x)
  #for delta in xrange( N_delta.shape[1]):
  #    N_delta[ :,delta] = np.sum( xm* np.roll( ym, delta,axis=1 ),axis=1 )

  #xmean = np.average( x,weights=xm,axis=1)[:,None]
  #ymean = np.average( y, weights=ym,axis=1)[:,None]

  # use d-FFT + convolution thm
  #ffx = fftpack.fft( (x-xmean)*xm,axis=1 )
  #ffy = fftpack.fft( (y-ymean)*ym, axis=1 )
  #iff = np.real( fftpack.ifft( np.conjugate(ffx) * ffy, axis=1 ) )

  return 0
  #return iff / N_delta


