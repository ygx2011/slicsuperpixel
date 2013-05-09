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

void syntax();

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
  int k = atoi(argv[2]);
  float m = atof(argv[3]);
  float threshold = atof(argv[4]);
  string filename = argv[5];

  Slic *s = new Slic(original_image, k , m);

  // iterate
  float e = 1000000;
  while(e > threshold)
  {
    e = s->iterate();
    cout << "Error: " << e << endl;
  }

  cout << "Resolution: " << s->getWidth() << " x " << s->getHeight() << endl;
  cout << "Num superpixels: " << s->getSuperpixels().size() << endl;
  cout << "S: " << s->getS() << endl;
  cout << "m: " << s->getM() << endl;
  cout << endl;

  Mat edges = edge_result(s);

  // Show contours
  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", edges );
  imwrite("contours.jpg", edges);

  cout << "Press any key to continue..." << endl;
  waitKey(0);

  destroyWindow("Contours");

  // Loop through each superpixel and determine label. Write to file
  cout << "Extracting SIFT features to file " << filename << endl;
  ofstream data_file;
  data_file.open(filename.c_str());
  if(!data_file.is_open())
  {
    cout << "Something went wrong in opening file " << filename.c_str() << endl;
    exit(0);
  }

  Mat descriptors = s->getSIFTDescriptors();
  for(int i = 0; i < s->getSuperpixels().size(); i++)
  {
    Mat sp_display = Mat::zeros(s->getHeight(), s->getWidth(), CV_8UC3);
    
    vector<Point> sp_points = s->getSuperpixels().at(i).points;
    for(int j = 0; j < sp_points.size(); j++)
    {
      Point p = sp_points.at(j);
      sp_display.at<Vec3b>(p.y, p.x) = s->getOriginalImage().at<Vec3b>(p.y, p.x);
    }

    cout << "Input label for SP " << i << ": " << endl;
    namedWindow("Superpixel", CV_WINDOW_AUTOSIZE);
    imshow("Superpixel", sp_display);

    for(int col_counter = 0; col_counter < descriptors.cols; col_counter++)
    {
      data_file << descriptors.at<float>(i, col_counter) << ",";
    }

    int label = waitKey(0);
    switch(label)
    {
      case 48:
        cout << "Label: 0" << endl;
        data_file << "0" << endl;
        break;
      case 49:
        cout << "Label: 1" << endl;
        data_file << "1" << endl;
        break;
      case 50:
        cout << "Label: 2" << endl;
        data_file << "2" << endl;
        break;
      case 51:
        cout << "Label: 3" << endl;
        data_file << "3" << endl;
        break;
      case 52:
        cout << "Label: 4" << endl;
        data_file << "4" << endl;
        break;
      case 53:
        cout << "Label: 5" << endl;
        data_file << "5" << endl;
        break;
      case 54:
        cout << "Label: 6" << endl;
        data_file << "6" << endl;
        break;
      case 55:
        cout << "Label: 7" << endl;
        data_file << "7" << endl;
        break;
      case 56:
        cout << "Label: 8" << endl;
        data_file << "8" << endl;
        break;
      case 57:
        cout << "Label: 9" << endl;
        data_file << "9" << endl;
        break;
      case 27:
        cout << "Escape pressed..." << endl;
        cout << "Done..." << endl;
        data_file.close();
        exit(0);
      default:
        cout << "Invalid Input. Possible: 1, 2, 3, 4, 5, 6, 7, 8, 9, 0" << endl;
        data_file.close();
        exit(0);
    }
  }

  cout << "Data saved to file " << filename << endl;
  data_file.close();
  cout << "Done." << endl;

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
  cout << "Syntax: slic [image_file] [super_pixel_size] [m] [threshold] [data_file]" << endl;
}
