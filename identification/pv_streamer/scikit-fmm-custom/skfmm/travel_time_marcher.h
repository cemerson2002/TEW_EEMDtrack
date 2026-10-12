//travel_time_marcher.h
#include "distance_marcher.h"
#include <set>
#include <iostream>

class heap;

class travelTimeMarcher : public distanceMarcher
{
public:
  travelTimeMarcher(double *phi,      double *dx, double dH, long *flag,
                    double *distance, int ndim,   int *shape,
                    bool self_test,   int order,
                    double *speed,    double narrow,
                    int periodic) :
    distanceMarcher(phi, dx, dH, flag, distance, ndim, shape, self_test,
                    order, narrow, periodic),
    speed_(speed)
  {

    if (ndim == 3) // for our case
    for (int i=0; i < shape_[ndim - 2]*shape_[ndim - 1]; i++)
    {
      // we need to be carefull here: very small speed values can result
      // in an overflow
      if (speed_[i]<doubleEpsilon) flag_[i]=Mask;
    }
    else
    // for 2d
    for (int i=0; i<size_; i++)
    {
      // we need to be carefull here: very small speed values can result
      // in an overflow
      if (speed_[i]<doubleEpsilon) flag_[i]=Mask;
    }
  }

  virtual ~travelTimeMarcher() { }

protected:
  virtual void             initalizeFrozen();
  virtual double           updatePointOrderTwo(int i);
  virtual double           updatePointOrderTwo(int i, std::set<int> avoid_dim);
  virtual double           solveQuadratic(int i, const double &a,
                                          const double &b, double &c);
private:
  double *speed_;
};
