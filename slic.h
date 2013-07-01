#ifndef SLIC_H_
#define SLIC_H_

#include <iostream>
#include <vector>
#include <exception>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include "superpixel.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace cv;

namespace cv
{
  class Slic
  {

  public:
    Mat getOriginalImage();

    Mat getLabImage();

    Mat getGradientImage();

    vector<superpixel> getSuperpixels();

    int getK();

    int getS();

    int getM();

    int getWidth();

    int getHeight();


    // Return residual errors
    float iterate();

    Slic(Mat originalImage, int s, float m);
    virtual ~Slic();

    Mat getSIFTDescriptors();
    Mat getSurfDescriptors();
    Mat getOrbDescriptors();
    
    int superpixelRowCount;
    int superpixelColCount;

  private:
    int width;
    int height;
    float m;
    float S;
    Mat gradientImage;
    Mat originalImage;
    Mat labImage;
    vector<superpixel> superpixels;
    int k;

    void init(Mat originalImage, int s, float m);
    Mat convertToGradient(Mat originalImage);
    Vec3i rangeLabValues(Vec3b lab);
    void enforceConnectivity();
    vector<superpixel> getSuperpixelNeighborsAndSelf(int i);
    float slicDistance(vector<float> a, vector<float> b);
    vector<float> getLabxy(Mat lab, Point c);
  };
}
#endif
