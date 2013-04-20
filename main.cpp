#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

#include "slic.h"

void syntax();

int main(int argc, char **argv)
{
  if(argc != 3) 
  {
    syntax();
    exit(-1);
  }

  Mat original_image = imread(argv[1]);
  int k = atoi(argv[2]);

  Slic *s = new Slic();
  s->init(original_image, k);

  // iterate
  s->iterate();

  // Show original
  namedWindow("original");
  imshow("original", s->get_original_image());

  // Show gradient
  namedWindow("gradient");
  imshow("gradient", s->get_gradient_image());

  // Show centers
  Mat image = s->get_original_image();
  vector<superpixel> superpixels = s->get_superpixels();
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

    line(
      image,
      superpixels.at(i).center,
      superpixels.at(i).center,
      Scalar(255, 0, 0),
      5
    );
  }

  for(int i = 0; i < s->get_superpixels().size(); i++)
  {
    printf("Neighbors for %d:\n", i);
    vector<superpixel> neighbors = s->get_superpixel_neighbors_and_self(i);
    for(int j = 0; j < neighbors.size(); j++)
    {
      printf("%d\n", neighbors.at(j).id);
    }
  }


  // Show centers
  namedWindow("centers");
  imshow("centers", image);

  waitKey(0);

  return 0;
}

void syntax()
{
  cout << "Syntax: slic [image_file] [super_pixel_size]" << endl;
}

