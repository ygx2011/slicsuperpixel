#include "slic.h"

void Slic::init(Mat originalImage, int s, float m)
{
  // TODO: Assert valid s
  CV_Assert(originalImage.cols % s == 0);
  CV_Assert(originalImage.rows % s == 0);

  this->width = originalImage.cols;
  this->height = originalImage.rows;
  this->originalImage = originalImage;

  this->S = (float)s;
  this->m = m;

  this->superpixelRowCount = this->height / s;
  this->superpixelColCount = this->width / s;

  // Setup the image in lab color space
  cvtColor(originalImage, this->labImage, CV_BGR2Lab);
  
  // Setup the gradient image based on labImage
  this->gradientImage = this->convertToGradient(this->labImage);

  int superpixel_counter = 0;
  for(int y_counter = 0; y_counter < height; y_counter += s)
  {
    for(int x_counter = 0; x_counter < width; x_counter += s)
    {
      superpixel sp;
      
      int lowest_gradient = 100000;
      
      for(int x = x_counter; x < x_counter + (s); x++)
      {
        for(int y = y_counter; y < y_counter + (s); y++)
        {
          Point p(x, y);
          sp.points.push_back(p);
          if((int)this->gradientImage.at<uchar>(p.y, p.x) < lowest_gradient)
          {
            sp.center = Point(p.x, p.y);
            lowest_gradient = (int)this->gradientImage.at<uchar>(p.y, p.x);
          }
        }
      }

      // Get center point value
      sp.center_point_value = getLabxy(labImage, sp.center);

      sp.id = superpixel_counter; 

      superpixels.push_back(sp);
      superpixel_counter++;
    }
  }

  this->superpixels = superpixels;
}

float Slic::iterate()
{
  vector<superpixel> new_superpixels(this->superpixels.size());
  int num_superpixels = new_superpixels.size();

  // Update IDs.
  for(int sp_counter = 0; sp_counter < num_superpixels; sp_counter++)
  {
    new_superpixels.at(sp_counter).id = this->superpixels.at(sp_counter).id;
    new_superpixels.at(sp_counter).center = this->superpixels.at(sp_counter).center;
  }

  for(int sp_counter = 0; sp_counter < num_superpixels; sp_counter++)
  {
    superpixel current_sp = this->superpixels.at(sp_counter);

    vector<superpixel> current_sp_neighbors = getSuperpixelNeighborsAndSelf(current_sp.id);
    int num_neighbors = current_sp_neighbors.size();

    vector<Point> test_points = current_sp.points;
    int num_test_points = test_points.size();

    for(int tp_counter = 0; tp_counter < num_test_points; tp_counter++)
    {
      Point current_point = test_points.at(tp_counter);
      vector<float> current_point_value = getLabxy(this->labImage, current_point);

      superpixel lowest_sp = current_sp;
      float lowest_distance = slicDistance(current_point_value, lowest_sp.center_point_value);
      int lowest_id = lowest_sp.id;

      for(int j = 0; j < num_neighbors; j++)
      {
        float d = slicDistance(current_point_value, current_sp_neighbors.at(j).center_point_value);

        if(d < lowest_distance)
        {
          lowest_distance = d;
          lowest_sp = current_sp_neighbors.at(j);
          lowest_id = current_sp_neighbors.at(j).id;
        }
      }

      new_superpixels.at(lowest_id).points.push_back(current_point);
    }
  }

  // Recompute centers and error
  float residual_error = 0;

  for(int i = 0; i < num_superpixels; i++)
  {
    float mean_l = 0;
    float mean_a = 0;
    float mean_b = 0;
    float mean_x = 0;
    float mean_y = 0;

    superpixel s = new_superpixels.at(i);
    int previous_x = s.center.x;
    int previous_y = s.center.y;
    vector<Point> s_points = s.points;
    int s_size = s.points.size();

    for(int j = 0; j < s_size; j++)
    {
      mean_l += this->labImage.at<Vec3b>(s_points.at(j).y, s_points.at(j).x)[0];
      mean_a += this->labImage.at<Vec3b>(s_points.at(j).y, s_points.at(j).x)[1];
      mean_b += this->labImage.at<Vec3b>(s_points.at(j).y, s_points.at(j).x)[2];
      mean_x += s_points.at(j).x;
      mean_y += s_points.at(j).y;
    }

    mean_l = (mean_l / s_size);
    mean_a = (mean_a / s_size);
    mean_b = (mean_b / s_size);
    mean_x = (mean_x / s_size);
    mean_y = (mean_y / s_size);

    vector<float> center_point_value;
    center_point_value.push_back(mean_l);
    center_point_value.push_back(mean_a);
    center_point_value.push_back(mean_b);
    center_point_value.push_back(mean_x);
    center_point_value.push_back(mean_y);

    if(mean_x >= 0 || mean_y >= 0)
    {
      Point new_center((int)mean_x, (int)mean_y);
      new_superpixels.at(i).center = new_center;
      new_superpixels.at(i).center_point_value = center_point_value;
    }

    residual_error += std::abs(previous_x - new_superpixels.at(i).center.x) + std::abs(previous_y - new_superpixels.at(i).center.y); 
  }

  this->superpixels = new_superpixels;

  return residual_error / new_superpixels.size();
}

// TODO: Implement this
void Slic::enforceConnectivity()
{

}

float Slic::slicDistance(vector<float> a, vector<float> b)
{
  float d = 0;

  float S = this->S;
  float m = this->m;

  float d_lab = sqrt(pow((a.at(0) - b.at(0)), 2) + pow((a.at(1) - b.at(1)),2) + pow((a.at(2) - b.at(2)),2));
  float d_xy = sqrt(pow((a.at(3) - b.at(3)),2) + pow((a.at(4) - b.at(4)), 2));

  d = d_lab + ((m / S) * d_xy);

  return d;
}


vector<superpixel> Slic::getSuperpixelNeighborsAndSelf(int index)
{
  vector<superpixel> ret;
  superpixel s = this->superpixels.at(index);
  ret.push_back(s);

  // Get top
  if((index - this->superpixelColCount) >= 0)
  {
    s = this->superpixels.at(index - this->superpixelColCount);
    ret.push_back(s);
  }

  int v_center = index - ((int)(index / this->superpixelColCount) * 2);
  int v_right = index - ((int)((index + 1) / this->superpixelColCount) * 2);
  int v_left = index - ((int)((index - 1) / this->superpixelColCount) * 2);

  // Get left
  if(v_center >= v_left && index != 0)
  {
    s = this->superpixels.at(index - 1);
    ret.push_back(s);

    // Get top left
    int left_id = index - 1;
    if((left_id - this->superpixelColCount) >= 0) {
      s = this->superpixels.at(left_id - this->superpixelColCount);
      ret.push_back(s);
    }

    // Get bottom left
    if((left_id + this->superpixelColCount) <= (this->superpixels.size() - 1))
    {
      s = this->superpixels.at(left_id + this->superpixelColCount);
      ret.push_back(s);
    }
  }

  // Get right
  if(v_center <= v_right && index != this->superpixels.size() - 1)
  {
    s = this->superpixels.at(index + 1);
    ret.push_back(s);

    // Get top right
    int right_id = index + 1;
    if((right_id - this->superpixelColCount) >= 0) {
      s = this->superpixels.at(right_id - this->superpixelColCount);
      ret.push_back(s);
    }

    // Get bottom right
    if((right_id + this->superpixelColCount) <= (this->superpixels.size() - 1))
    {
      s = this->superpixels.at(right_id + this->superpixelColCount);
      ret.push_back(s);
    }
  }

  // Get bottom
  if((index + this->superpixelColCount) <= (this->superpixels.size() - 1))
  {
    s = this->superpixels.at(index + this->superpixelColCount);
    ret.push_back(s);
  }

  //printf("v_center: %d, v_right: %d, v_left: %d\n", v_center, v_right, v_left);

  return ret;
}

Mat Slic::convertToGradient(Mat originalImage)
{
  if(originalImage.channels() > 0)
  {
    cvtColor(originalImage, originalImage, CV_BGR2GRAY);
  }
  Mat result_image;

  // Equalize histogram 
  equalizeHist(originalImage, result_image);

  // Gaussian blur
  // GaussianBlur(src, dst, Size(x, y), ox, oy)
  GaussianBlur(result_image, result_image, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0);

  // Compute gradients using Sobel operator
  Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
  
  // Gradient X
  Sobel(result_image, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
  convertScaleAbs(grad_x, abs_grad_x);

  // Gradient Y
  Sobel(result_image, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
  convertScaleAbs(grad_y, abs_grad_y);

  // Total Gradient (approximation)
  addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, result_image);

  return result_image;
}

Slic::Slic(Mat original_image, int s, float m)
{
  this->init(original_image, s, m);
}

Slic::~Slic()
{
}

Mat Slic::getOriginalImage()
{
  return this->originalImage;
}


vector<superpixel> Slic::getSuperpixels()
{
  return this->superpixels;
}

Mat Slic::getLabImage()
{
  return this->labImage;
}


Mat Slic::getGradientImage()
{
  return this->gradientImage;
}

int Slic::getK()
{
  return this->k;
}

int Slic::getHeight()
{
  return this->height;
}

int Slic::getWidth()
{
  return this->width;
}

int Slic::getM()
{
  return this->m;
}

int Slic::getS()
{
  return this->S;
}


// Recommended by rafajafar (opencv IRC)
Vec3i Slic::rangeLabValues(Vec3b lab)
{
  Vec3i n_lab;
  n_lab[0] = (lab[0] * 100) / 255;
  n_lab[1] = lab[1] - 128;
  n_lab[2] = lab[2] - 128;

  return n_lab;
}


vector<float> Slic::getLabxy(Mat lab, Point c)
{
  vector<float> ret;
  float l = lab.at<Vec3b>(c.y, c.x)[0];
  float a = lab.at<Vec3b>(c.y, c.x)[1];
  float b = lab.at<Vec3b>(c.y, c.x)[2];
  float x = (float)c.x;
  float y = (float)c.y;
  ret.push_back(l);
  ret.push_back(a);
  ret.push_back(b);
  ret.push_back(x);
  ret.push_back(y);

  return ret;
}

Mat Slic::getSIFTDescriptors()
{
  Mat descriptors;
  vector<KeyPoint> keypoints;

  vector<superpixel> superpixels = this->superpixels;

  for(int i = 0; i < superpixels.size(); i++)
  {
    superpixel sp = superpixels.at(i);
    KeyPoint kp(sp.center, 8);
    keypoints.push_back(kp);
  }

  SiftDescriptorExtractor extractor;

  extractor.compute(this->originalImage, keypoints, descriptors);

  return descriptors;
}

Mat Slic::getSurfDescriptors()
{
  Mat descriptors;
  vector<KeyPoint> keypoints;

  vector<superpixel> superpixels = this->superpixels;

  for(int i = 0; i < superpixels.size(); i++)
  {
    superpixel sp = superpixels.at(i);
    KeyPoint kp(sp.center, 8);
    keypoints.push_back(kp);
  }

  SurfDescriptorExtractor extractor;

  extractor.compute(this->originalImage, keypoints, descriptors);

  return descriptors;
}

Mat Slic::getOrbDescriptors()
{
  Mat descriptors;
  vector<KeyPoint> keypoints;

  vector<superpixel> superpixels = this->superpixels;

  for(int i = 0; i < superpixels.size(); i++)
  {
    superpixel sp = superpixels.at(i);
    KeyPoint kp(sp.center, 8);
    keypoints.push_back(kp);
  }

  OrbDescriptorExtractor extractor;

  extractor.compute(this->originalImage, keypoints, descriptors);

  return descriptors;
}
