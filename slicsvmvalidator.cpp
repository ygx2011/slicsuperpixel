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
#include <opencv2/ml/ml.hpp>

#include "slic.h"
#include "timer.cpp"

using namespace cv;
using namespace std;

void syntax();

Mat edge_result(Slic *s);

Scalar white = Scalar(255, 255, 255);

int main(int argc, char **argv)
{
  if(argc != 7)
  {
    syntax();
    exit(-1);
  }

  Mat original_image = imread(argv[1]);
  int k = atoi(argv[2]);
  float m = atof(argv[3]);
  float threshold = atof(argv[4]);
  string svm_model_file = argv[5];
  string output_file_ref = argv[6];
  ofstream output_file;
  // Data file for writing
  output_file.open(output_file_ref.c_str());
  if(!output_file.is_open())
  {
    cout << "Error in opening file " << output_file_ref << endl;
    exit(-1);
  }

  // Timer for benchmarking
  Timer tm;

  cout << "Initializing..." << endl;
  tm.start();
  Slic *s = new Slic(original_image, k , m);
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
  }

  Mat contours = edge_result(s);
  vector<superpixel> superpixels = s->getSuperpixels();

  Mat descriptors = s->getSIFTDescriptors();

  CvSVM svm;
  svm.load(svm_model_file.c_str());

  for(int i = 0; i < superpixels.size(); i++)
  {
    Mat sample = Mat_<float>(1, descriptors.cols);
    for(int col = 0; col < descriptors.cols; col++)
    {
      sample.at<float>(0, col) = descriptors.at<float>(i, col);
    }

    float prediction = svm.predict(sample);
    output_file << i << "," << prediction << "\n";
    vector<Point> points = superpixels.at(i).points;
    for(int j = 0; j < points.size(); j++)
    {
      Point p = points.at(j);
      if(prediction == 0)
      {
        contours.at<Vec3b>(p.y, p.x)[2] = 255;
      }
      else if(prediction == 1)
      {
        contours.at<Vec3b>(p.y, p.x)[0] = 255;
      }
    }
  }

  printf("Closing output file...\n");
  output_file.close();

  // Show contours
  namedWindow( "Results", CV_WINDOW_AUTOSIZE );
  imshow( "Results", contours );

  waitKey(0);

  imwrite("results.jpg", contours);

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

void syntax()
{
  cout << "Syntax: svmslicvalidator [image_file] [s] [m] [threshold] [model_file] [output_file]" << endl;
}