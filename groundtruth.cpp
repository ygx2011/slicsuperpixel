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
  if(argc != 6)
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
  
  // Data file for writing
  data_file.open(filename.c_str());
  if(!data_file.is_open())
  {
    cout << "Error in opening file " << filename << endl;
    exit(-1);
  }

  cout << "Initializing..." << endl;
  s = new Slic(original_image, k , m);

  cout << "Resolution: " << s->getWidth() << " x " << s->getHeight() << endl;
  cout << "Num superpixels: " << s->getSuperpixels().size() << endl;
  cout << "S: " << s->getS() << endl;
  cout << "m: " << s->getM() << endl;
  cout << endl;

  // iterate
  float e = 1000000;
  while(e > threshold)
  {
    e = s->iterate();
    cout << "Error: " << e << endl;
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
  imwrite("contours.jpg", contours);

  cout << "Press any key to continue..." << endl;
  int key = waitKey(-1);

  for(int i = 0; i < superpixels.size(); i++)
  {
    vector<Point> points = superpixels.at(i).points;

    cout << "Label for SP " << i << ": ";
    cin >> label;

    if(label == 0)
    {
      for(int j = 0; j < points.size(); j++)
      {
        contours.at<Vec3b>(points.at(i).y, points.at(i).x)[2] = 255;
      }
    }
    else if(label == 1)
    {
      for(int j = 0; j < points.size(); j++)
      {
        contours.at<Vec3b>(points.at(i).y, points.at(i).x)[0] = 255;
      }
    }

    imshow("Contours", contours);
  }

  cout << "Press ESC to exit..." << endl;

  while(key != 27)
  {
    key = waitKey(-1);
  }

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
    cout << "Ground truth:";
    cin >> label;

    data_file << label << "\n";
    
    int l = labels.at<int>(y, x);
    
    // Print out descriptors
    for(int i = 0; i < descriptors.cols; i++)
    {
      data_file << descriptors.at<float>(l, i) << ",";
    }
    
    vector<Point> points = s->getSuperpixels().at(l).points;
    for(int i = 0; i < points.size(); i++)
    {
      if(label == 1)
      {
        contours.at<Vec3b>(points.at(i).y, points.at(i).x)[0] = 255;
      }
      else
      {
        contours.at<Vec3b>(points.at(i).y, points.at(i).x)[2] = 255;
      }
    }

    imshow("Contours", contours);
  }
}

void syntax()
{
  cout << "Syntax: groundtruth [image_file] [super_pixel_size] [m] [threshold] [data_file]" << endl;
}
