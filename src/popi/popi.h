#include <vector>

using namespace std;

class PolarPilatus
{
  vector<float> cosPhi,sinPhi,A;
  float a00,a10,a20,a30;
  float a01,a11,a21,a31;
  float a02,a12,a22,a32;
  float a03,a13,a23,a33;
  float interpI;

  float F( float ar[],int i, int j );
  float EvaluateIntensity( float coef[], float x , float y );
  float aveNo0(vector<float>& ar);
  
  void  bicubicCoefficients( float * I );
  void  getPixelsAtQ(vector<float>& IvsPhi, int q_index, float q, float a, float b);
  void  getRing (int Nphi_, float q_pix, vector<float>& IvsPhi);
  void  nearest_multiple(int& n, int m);
  void  binPhi(int numBins, int samplesPerBin, int qpix, vector<float>& IvsPhi, vector<float>& IvsPhi_binned);
  

public:
  PolarPilatus(int Xdim_,         int Ydim_,      float * binData, 
               float detdist_,    float pixsize_, float wavelen_,
		float x_center_, float y_center_);
  int    Xdim,Ydim;
  int    Nphi,Nq;
  float  pixsize,detdist,wavelen;
  float  qres;
  float  x_center,y_center;
  
  //float* polar_pixels;
  
  void  Center(float qMin, float qMax, float center_res, int Nphi_, float size, float dq);
  void  InterpolateToPolar(float qres_, int Nphi_, int Nq_, float maxq_pix, float maxq, float * polar_pixels);
 ~PolarPilatus();
};
