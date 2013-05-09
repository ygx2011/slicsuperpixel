#ifndef SUPERPIXEL_H_
#define SUPERPIXEL_H_

#include <iostream>
#include <vector>
#include <exception>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace cv;

#define KERNEL_SIZE 5

struct superpixel
{
  int id;
  vector<Point> points;
  Point center;
  vector<float> center_point_value;
  vector<float> descriptors;
};

#endif
