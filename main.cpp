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

Mat edges;

Mat labels;

Mat descriptors;

ofstream data_file;

int label;

int feature_set;

int mode;

int key;

int current_id = 0;

vector<superpixel> superpixels;

void syntax();

Scalar white(255, 255, 255);

Mat edge_result(Slic *s);

void mark_current(int current_id);

void mark_labelled(int id);

int get_descriptor_type();

int get_mode();

void write_row_descriptors(int id, int label);

void write_row_groundtruth(int id, int label);

void generate_labels();

void generate_groundtruth();

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

  // Option: label generation or groundtruth
  mode = get_mode();

  // iterate
  printf("Producing superpixels...\n");
  float e = 1000000;
  while(e > threshold)
  {
    e = s->iterate();
    cout << "Error: " << e << endl;
  }

  if(mode == 1)
  {
    feature_set = get_descriptor_type();

    // Descriptors
    printf("Using feature set %d\n", feature_set);
    descriptors = s->getDescriptors(feature_set);
  }

  contours = edge_result(s);
  contours.copyTo(edges);
  labels = Mat_<int>(contours.rows, contours.cols);
  superpixels = s->getSuperpixels();
  for(int i = 0; i < superpixels.size(); i++)
  {
    for(int j = 0; j < superpixels.at(i).points.size(); j++)
    {
      labels.at<int>(superpixels.at(i).points.at(j).y, superpixels.at(i).points.at(j).x) = i;
    }
  }

  // Show contours
  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", contours );
  imwrite("contours.jpg", contours);

  cout << "Press any key to continue..." << endl;
  mark_current(current_id);

  if(mode == 1)
  {
    printf("Generating labels...\n");
    generate_labels();
  }
  else if(mode == 2)
  {
    printf("Generating groundtruth...\n");
    generate_groundtruth();
  }

  printf("Closing file...\n");
  data_file.close();

  printf("Program terminating...\n");
  return 0;
}

void generate_labels()
{
  while(key = waitKey(100), key != 27)
  {
    switch(key)
    {
      case 65362:
        if(current_id - s->superpixelColCount >= 0 && current_id - s->superpixelColCount < superpixels.size())
        {
          current_id = current_id - s->superpixelColCount;
          mark_current(current_id);
        }
        break;
      case 65361:
        if(current_id - 1 >= 0 && current_id - 1 < superpixels.size())
        {
          current_id = current_id - 1;
          mark_current(current_id);
        }
        break;
      case 65364:
        if(current_id + s->superpixelColCount >= 0 && current_id + s->superpixelColCount < superpixels.size())
        {
          current_id = current_id + s->superpixelColCount;
          mark_current(current_id);
        }
        break; 
      case 65363:
        if(current_id + 1 >= 0 && current_id + 1 < superpixels.size())
        {
          current_id = current_id + 1;
          mark_current(current_id);
        }
        break;
      case 48:
        label = 0;
        mark_labelled(current_id);
        write_row_descriptors(current_id, label);
        break;
      case 49:
        label = 1;
        mark_labelled(current_id);
        write_row_descriptors(current_id, label);
        break;
      case 50:
        label = 2;
        mark_labelled(current_id);
        write_row_descriptors(current_id, label);
        break;
      case 51:
        label = 3;
        mark_labelled(current_id);
        write_row_descriptors(current_id, label);
        break;
      case 52:
        label = 4;
        mark_labelled(current_id);
        write_row_descriptors(current_id, label);
        break;
      case 53:
        label = 5;
        mark_labelled(current_id);
        write_row_descriptors(current_id, label);
        break;
      case 54:
        label = 6;
        mark_labelled(current_id);
        write_row_descriptors(current_id, label);
        break;
      case 55:
        label = 7;
        mark_labelled(current_id);
        write_row_descriptors(current_id, label);
        break;
      case 56:
        label = 8;
        mark_labelled(current_id);
        write_row_descriptors(current_id, label);
        break;
      case 57:
        label = 9;
        mark_labelled(current_id);
        write_row_descriptors(current_id, label);
        break;
      default:
        //printf("Unregistered key: %d\n", key);
        break;
    }
  }
}

void generate_groundtruth()
{
  while(key = waitKey(100), key != 27)
  {
    switch(key)
    {
      case 65362:
        if(current_id - s->superpixelColCount >= 0 && current_id - s->superpixelColCount < superpixels.size())
        {
          current_id = current_id - s->superpixelColCount;
          mark_current(current_id);
        }
        break;
      case 65361:
        if(current_id - 1 >= 0 && current_id - 1 < superpixels.size())
        {
          current_id = current_id - 1;
          mark_current(current_id);
        }
        break;
      case 65364:
        if(current_id + s->superpixelColCount >= 0 && current_id + s->superpixelColCount < superpixels.size())
        {
          current_id = current_id + s->superpixelColCount;
          mark_current(current_id);
        }
        break; 
      case 65363:
        if(current_id + 1 >= 0 && current_id + 1 < superpixels.size())
        {
          current_id = current_id + 1;
          mark_current(current_id);
        }
        break;
      case 48:
        label = 0;
        mark_labelled(current_id);
        write_row_groundtruth(current_id, label);
        break;
      case 49:
        label = 1;
        mark_labelled(current_id);
        write_row_groundtruth(current_id, label);
        break;
      case 50:
        label = 2;
        mark_labelled(current_id);
        write_row_groundtruth(current_id, label);
        break;
      case 51:
        label = 3;
        mark_labelled(current_id);
        write_row_groundtruth(current_id, label);
        break;
      case 52:
        label = 4;
        mark_labelled(current_id);
        write_row_groundtruth(current_id, label);
        break;
      case 53:
        label = 5;
        mark_labelled(current_id);
        write_row_groundtruth(current_id, label);
        break;
      case 54:
        label = 6;
        mark_labelled(current_id);
        write_row_groundtruth(current_id, label);
        break;
      case 55:
        label = 7;
        mark_labelled(current_id);
        write_row_groundtruth(current_id, label);
        break;
      case 56:
        label = 8;
        mark_labelled(current_id);
        write_row_groundtruth(current_id, label);
        break;
      case 57:
        label = 9;
        mark_labelled(current_id);
        write_row_groundtruth(current_id, label);
        break;
      default:
        //printf("Unregistered key: %d\n", key);
        break;
    }
  }
}

void mark_current(int current_id)
{
  if(current_id >= 0)
  {
    edges.copyTo(contours);
    vector<Point> points = s->getSuperpixels().at(current_id).points;
    for(int i = 0; i < points.size(); i++)
    {
      contours.at<Vec3b>(points.at(i).y, points.at(i).x)[2] = 255;
    }

    imshow("Contours", contours);
  }
}

void mark_labelled(int id)
{
  if(id >= 0)
  {
    vector<Point> points = s->getSuperpixels().at(id).points;
    for(int i = 0; i < points.size(); i++)
    {
      edges.at<Vec3b>(points.at(i).y, points.at(i).x)[0] = 255;
    }
  }
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

int get_descriptor_type()
{
  int desc_type;

  printf("Available descriptors:\n");
  printf("-- MID SIFT (1)\n");
  printf("-- MID SURF (2)\n");
  printf("-- MID ORB (3)\n");

  cout << "Enter number: ";

  cin >> desc_type;

  if(desc_type != DESC_MID_SIFT &&
      desc_type != DESC_MID_SURF &&
      desc_type != DESC_MID_ORB)
  {
    printf("Invalid descriptor set.\n");
    return get_descriptor_type();
  }
  else
  {
    return desc_type;
  }
}

int get_mode()
{
  int mode;

  printf("Available modes:\n");
  printf("-- Label Generation (1)\n");
  printf("-- Groundtruth (2)\n");

  cout << "Enter number: ";

  cin >> mode;

  if(mode != 1 && mode != 2)
  {
    printf("Invalid mode.\n");
    return get_mode();
  }
  else
  {
    return mode;
  }
}

void write_row_descriptors(int id, int label)
{
  // Print out descriptors
  cout << "[DEBUG] Features: ";
  for(int i = 0; i < descriptors.cols; i++)
  {
    cout << descriptors.at<float>(id, i) << ",";
    data_file << descriptors.at<float>(id, i) << ",";
  }

  cout << label << endl;
  data_file << label << "\n";
}

void write_row_groundtruth(int id, int label)
{
  cout << "[DEBUG] Ground truth: ";
  cout << id << "," << label << "\n";
  data_file << id << "," << label << "\n";
}

void syntax()
{
  cout << "Syntax: slic [image_file] [super_pixel_size] [m] [threshold] [data_file]" << endl;
}