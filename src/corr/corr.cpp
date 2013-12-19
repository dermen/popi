#include <math.h>
#include <iostream>

#include <corr.h>

using namespace std;


Corr::Corr(int N_, float * ar1, float * ar2, float * ar3, float mask_val_)

{
  N         =  N_;
  mask_val  = mask_val_;
  ar1_mean  =  mean_no_zero(ar1);
  ar2_mean  =  mean_no_zero(ar2);

  correlate(ar1,ar2,ar3);



}

void Corr::correlate(float * ar1, float * ar2,float * arC)
{
  int phi(0);
  while (phi < N)
  {
    arC[phi] = 0;
    int i(0);
    float counts(0);
    while (i < N)
    {
      int j = i + phi;
      if (j >= N)
        j -= N;
      if (ar1[i] != mask_val && ar2[j] != mask_val)
      {
        arC[phi] += (ar1[i]- ar1_mean) * (ar2[j]-ar2_mean);
        counts += 1;
      }
      i++;
    }
    if (counts > 0)
        arC[phi] = arC[phi] / counts;
    phi++;
  }

}

float Corr::mean_no_zero(float * ar)

{
  float ar_mean(0);
  float counts(0);
  int i(0);
  while(i < N)
  {
    if(ar[i] != mask_val)
    {
      ar_mean += ar[i];
      counts ++;
    }
    i ++;
  }
  if(counts > 0)
    return ar_mean / counts;
  else
    return 0;
}


Corr::~Corr()
{

}


