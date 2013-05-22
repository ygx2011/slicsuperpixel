#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "timer.cpp"

using namespace cv;
using namespace std;

#include "slic.h"

Slic *s;

Mat contours;

Mat labels;

Mat descriptors;

ofstream data_file;

int label;

void syntax();

void mouse_event(int e, int x, int y, int flags, void *param);

Scalar white(255, 255, 255);

Mat edge_result(Slic *s);

int main(int argc, char **argv)
{
  if(argc != 7)
  {
    syntax();
    exit(-1);
  }

  Mat original_image = imread(argv[1]);
  labels = Mat_<int>(original_image.rows, original_image.cols);

  int k = atoi(argv[2]);
  float m = atof(argv[3]);
  float threshold = atof(argv[4]);
  string filename = argv[5];
  label = atoi(argv[6]);
  
  // Data file for writing
  data_file.open(filename.c_str());
  if(!data_file.is_open())
  {
    cout << "Error in opening file " << filename << endl;
    exit(-1);
  }

  // Timer for benchmarking
  Timer tm;

  cout << "Initializing..." << endl;
  tm.start();
  s = new Slic(original_image, k , m);
  tm.stop();

  cout << "Initialization time: " << tm.duration();
  cout << endl;

  cout << "Resolution: " << s->getWidth() << " x " << s->getHeight() << endl;
  cout << "Num superpixels: " << s->getSuperpixels().size() << endl;
  cout << "S: " << s->getS() << endl;
  cout << "m: " << s->getM() << endl;
  cout << endl;

  // iterate
  float e = 1000000;
  while(e > threshold)
  {
    tm.start();
    e = s->iterate();
    tm.stop();
    cout << "Error: " << e << ", Duration: " << tm.duration() << endl;
    // printf("Residual error: %f, Duration: %d", e, tm.duration());
  }

  contours = edge_result(s);
  labels = Mat_<int>(contours.rows, contours.cols);
  vector<superpixel> superpixels = s->getSuperpixels();
  for(int i = 0; i < superpixels.size(); i++)
  {
    for(int j = 0; j < superpixels.at(i).points.size(); j++)
    {
      labels.at<int>(superpixels.at(i).points.at(j).y, superpixels.at(i).points.at(j).x) = i;
    }
  }

  // Descriptors
  descriptors = s->getSIFTDescriptors();

  // Show contours
  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", contours );
  setMouseCallback("Contours", mouse_event, 0);
  imwrite("contours.jpg", contours);

  cout << "Press any key to continue..." << endl;
  waitKey(0);

  data_file.close();

  return 0;
}

Mat edge_result(Slic *s)
{
  Mat edges;
  Mat image;
  s->getOriginalImage().copyTo(edges);
  
  s->getOriginalImage().copyTo(image);

  vector<superpixel> superpixels = s->getSuperpixels();
  RNG rng( 0xFFFFFFFF );
  for(int i = 0; i < superpixels.size(); i++)
  {
    Scalar s(rng.uniform(0, 255),rng.uniform(0, 255),rng.uniform(0, 255));
    for(int j = 0; j < superpixels.at(i).points.size(); j++)
    {
      line(image,
        superpixels.at(i).points.at(j),
        superpixels.at(i).points.at(j),
        s,
        1);
    }
  }

  Mat src;
  image.copyTo(src);
  Mat src_gray;
  cvtColor( src, src_gray, CV_BGR2GRAY );

  Mat canny_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// Detect edges using canny
  Canny( src_gray, canny_output, 5, 5*2, 3 );
  /// Find contours
  findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// Draw contours
  for( int i = 0; i< contours.size(); i++ )
  {
    drawContours(edges, contours, i, white, 1, 1, hierarchy, 0, Point() );
  }

  return edges;
}

void mouse_event(int e, int x, int y, int flags, void *param)
{
  if(e == CV_EVENT_LBUTTONDOWN)
  {
    int l = labels.at<int>(y, x);
    vector<Point> points = s->getSuperpixels().at(l).points;
    for(int i = 0; i < points.size(); i++)
    {
      contours.at<Vec3b>(points.at(i).y, points.at(i).x)[0] = 255;
    }

    imshow("Contours", contours);

    // Print out descriptors
    for(int i = 0; i < descriptors.rows; i++)
    {
      data_file << descriptors.at<float>(l, i) << ",";
    }

    data_file << label << "\n";
  }
}

void syntax()
{
  cout << "Syntax: slic [image_file] [super_pixel_size] [m] [threshold] [data_file] [label]" << endl;
}
