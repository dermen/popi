import numpy as np
cimport numpy as np
import random
from scipy import fftpack

cdef extern from "corr.h":
  cdef cppclass Corr:
    Corr(int N_, float * ar1, float * ar2, float * ar3, float mask_val_ ,int mean_sub_) except +

cdef Corr * c

def correlate(A,B, mask_val = -1,mean_sub=1):
  """
  compute the correlation between 2 arrays
  Parameters:
  -----------
  A, 1D numpy array
  B, 1D numpy array
  mask_val, number value of each masked pixels ( defaults = -1 )
            they should all have the same value
  mean_sub, 1 to subtract the mean before correlating, 0 to not
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
  c  = new Corr(N,&v1[0], &v2[0], &v3[0], mask_val, mean_sub)
  del c
  return v3

def autocorrelate(A, mask_val = -1,mean_sub=1):
  """
  compute the correlation between 2 arrays
  Parameters:
  -----------
  A, 1D numpy array
  mask_val, number value of each masked pixels ( defaults = -1 )
            they should all have the same value
  mean_sub, 1 to subtract the mean before correlating, 0 to not
  """
  N = A.shape[0]
  C = np.zeros_like(A[:N/2])
  cdef np.ndarray[ndim=1,dtype=np.float32_t] v1
  cdef np.ndarray[ndim=1,dtype=np.float32_t] v3
  
  v1 = np.ascontiguousarray(A.flatten(),dtype=np.float32)
  v3 = np.ascontiguousarray(C.flatten(),dtype=np.float32)
  c  = new Corr(N,&v1[0], &v1[0], &v3[0], mask_val, mean_sub)
  del c
  return v3

def autocorrelate_using_fft( x):
  """
  Compute the correlation between 2 arrays using the 
  convolution theorem. Works well for unmasked arrays.
  Passing masked arrays will results in numerical errors.

  Parameters
  ----------
  x : 1d numpy array of floats
      The intensities along ring 1
    
  Returns
  -------
  iff : 1d numpy darray, float
      The correlation between x and y
  """
  np.fft.rfft( x, n=x.shape[1], axis=1 )
  return np.fft.irfft(x*np.conjugate(x), n=x.shape[1], axis=1 )
