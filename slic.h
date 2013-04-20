#ifndef SLIC_H_
#define SLIC_H_

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

#define KERNEL_SIZE 5

struct factor_pair
{
  int x;
  int y;
};

struct superpixel
{
  int id;
  vector<Point> points;
  Point center;
  double mean;
  float distance;
};

class Slic
{

public:
  int superpixel_width;
  int superpixel_length;

  Mat get_original_image();
  void set_original_image(Mat original_image);

  Mat get_lab_image();
  void set_lab_image(Mat lab_image);

  vector<superpixel> get_superpixels();

  Mat get_gradient_image();

  int get_k();
  void set_k(int k);

  void init(Mat original_image, int k, float m);
  float slic_distance(Point p1, Point p2);

  Vec3i range_lab_values(Vec3b lab);

  void iterate_superpixels();
  void enforce_connectivity();

  Slic();
  virtual ~Slic();

  int superpixel_row_count;
  int superpixel_col_count;
  vector<superpixel> get_superpixel_neighbors_and_self(int i);
  int width;
  int length;
  float m;
  float S;

private:
  Mat original_image;
  Mat lab_image;
  Mat gradient_image;
  int k;

  vector<superpixel> superpixels;
  vector<factor_pair> factor_pairs(int index);
  Mat convert_to_gradient(Mat original_image);
};

#endif
