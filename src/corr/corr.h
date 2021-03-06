
class Corr
{
  int N;
  float mask_val;
  float ar1_mean;
  float ar2_mean;
  float mean_sans_mask(float * ar);
  void correlate(float * ar1, float * ar2, float * ar3);
  void autocorrelate(float * ar1, float * ar3);
public:
  Corr(int N_, float * ar1, float * ar2, float * ar3, float mask_val_, int mean_sub_);
  ~Corr();
};

