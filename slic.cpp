#include "slic.h"

/**
 * TODO: Need proper handling if init fail
 */
void Slic::init(Mat original_image, int k)
{
  int width = original_image.cols;
  int length = original_image.rows;

  // Check if we can generate k number of regularly shaped superpixels.
  // If not, we crop the extra to fit the desired k.
  vector<factor_pair> fps = factor_pairs(k);
  int fx = fps.back().x;
  int fy = fps.back().y;

  int width_size = width / fx;
  int length_size = length / fy;

  this->superpixel_width = width_size;
  this->superpixel_length = length_size;

  int ir_width_size = 0;
  if(width % width_size != 0)
  {
    ir_width_size = width - (width_size * (fx - 1));
  }

  int ir_length_size = 0;
  if(length % length_size != 0)
  {
    ir_length_size = length - (length_size * (fy - 1));
  }

  if(ir_width_size > 0 || ir_length_size > 0)
  {
    Rect roi(0, 0, width - (ir_width_size), length - (ir_length_size));
    Mat cropped_image = original_image(roi);
    this->set_original_image(cropped_image);
  }

  // Setup the image in lab color space
  Mat lab_image;
  cvtColor(this->get_original_image(), lab_image, CV_BGR2Lab);
  this->set_lab_image(lab_image);

  // Setup the gradient image based on lab_image
  Mat gradient_image = this->convert_to_gradient(lab_image);
  this->gradient_image = gradient_image;

  // Make sure we have the updated width and length
  width = lab_image.cols;
  length = lab_image.rows;
  this->width = width;
  this->length = length;

  // Initialize superpixels into equal regions
  // Get superpixel_row_count and superpixel_col_count
  vector<superpixel> superpixels;
  superpixel_col_count = 0;
  superpixel_row_count = 0;
  int superpixel_counter = 0;
  for(int x_counter = 0; x_counter < width; x_counter +=  width_size)
  {
    for(int y_counter = 0; y_counter < length; y_counter += length_size)
    {
      superpixel sp;
      for(int x = x_counter; x < x_counter + (width_size); x++)
      {
        for(int y = y_counter; y < y_counter + (length_size); y++)
        {
          Point p(x, y);
          sp.points.push_back(p);
        }
      }
      sp.id = superpixel_counter; 
      superpixels.push_back(sp);
      superpixel_counter++;
    }
    superpixel_col_count++;
  }
  superpixel_row_count = superpixels.size() / superpixel_col_count;
  this->superpixel_row_count = superpixel_row_count;
  this->superpixel_col_count = superpixel_col_count;

  this->superpixels = superpixels;

  // Initialize center for each superpixel based on lowest gradient value
  for(int i = 0; i < superpixels.size(); i++)
  {
    superpixel sp = superpixels.at(i);
    vector<Point> sp_points = sp.points;
    vector<int> g_points;
    for(int j = 0; j < sp.points.size(); j++)
    {
      int g = (int)this->gradient_image.at<uchar>(sp_points.at(j).y, sp_points.at(j).x);
      g_points.push_back(g);
    }

    sort(g_points.begin(), g_points.end());
    for(int j = 0; j < sp.points.size(); j++)
    {
      int g = (int)this->gradient_image.at<uchar>(sp_points.at(j).y, sp_points.at(j).x);
      if(g == g_points.at(0))
      {
        this->superpixels.at(i).center = sp_points.at(j);
      }
    }
  }
}

void Slic::iterate()
{
  // Create new superpixel buffer with same size as current
  vector<superpixel> new_superpixels(this->superpixels.size());

  // Iterate through each superpixel
  for(int i = 0; i < this->superpixels.size(); i++)
  {
    vector<superpixel> sps = get_superpixel_neighbors_and_self(i);
    superpixel current_superpixel = this->superpixels.at(i);

    // loop through each pixel of the current superpixel
    for(int j = 0; j < current_superpixel.points.size(); j++)
    {
      Point current_point = current_superpixel.points.at(j);
      superpixel lowest_superpixel = sps.at(0);
      for(int c = 0; c < sps.size(); c++)
      {
        float d = slic_distance(current_point, sps.at(c).center);
        cout << "Distance: P(" 
              << current_point.x << ", " << current_point.y
              << ") and P("
              << sps.at(c).center.x << ", " << sps.at(c).center.y << "): "
              << d << endl;
      }
    }
  }
  

  // Replace the current superpixel set
  //this->superpixels = new_superpixels;
}

float Slic::slic_distance(Point p1, Point p2)
{
  Mat lab = this->lab_image;
  float ret = 0;
  float S = this->superpixel_width;
  float m = 10;

  float p1_l = lab.at<Vec3b>(p1.y, p1.x)[0];
  float p1_a = lab.at<Vec3b>(p1.y, p1.x)[1];
  float p1_b = lab.at<Vec3b>(p1.y, p1.x)[2];

  float p2_l = lab.at<Vec3b>(p2.y, p2.x)[0];
  float p2_a = lab.at<Vec3b>(p2.y, p2.x)[1];
  float p2_b = lab.at<Vec3b>(p2.y, p2.x)[2];

  float d_lab = sqrt(pow(p1_l - p2_l, 2) + pow(p1_a - p2_a, 2) + pow(p1_b - p2_b, 2));
  float d_xy = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));

  ret = d_lab + ((m / S) * d_xy);

  return ret;
}

vector<superpixel> Slic::get_superpixel_neighbors_and_self(int index)
{
  vector<superpixel> ret;
  superpixel s = this->superpixels.at(index);
  ret.push_back(s);

  // Get top
  if((index - this->superpixel_col_count) >= 0)
  {
    s = this->superpixels.at(index - this->superpixel_col_count);
    ret.push_back(s);
  }

  
  int v_center = index - ((int)(index / this->superpixel_col_count) * 2);
  int v_right = index - ((int)((index + 1) / this->superpixel_col_count) * 2);
  int v_left = index - ((int)((index - 1) / this->superpixel_col_count) * 2);

  // printf("v_center: %d, v_right: %d, v_left: %d\n", v_center, v_right, v_left);

  // Get left
  if(v_center >= v_left && index != 0)
  {
    s = this->superpixels.at(index - 1);
    ret.push_back(s);
  }

  // Get right
  if(v_center <= v_right && index != this->superpixels.size() - 1)
  {
    s = this->superpixels.at(index + 1);
    ret.push_back(s);
  }

  // Get bottom
  if((index + this->superpixel_col_count) <= (this->superpixels.size() - 1))
  {
    s = this->superpixels.at(index + this->superpixel_col_count);
    ret.push_back(s);
  }

  return ret;
}

Mat Slic::convert_to_gradient(Mat original_image)
{
  if(original_image.channels() > 0)
  {
    cvtColor(original_image, original_image, CV_BGR2GRAY);
  }
  Mat result_image;

  // Equalize histogram 
  equalizeHist(original_image, result_image);

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

Slic::Slic()
{
}

Slic::~Slic()
{
}

Mat Slic::get_original_image()
{
  return this->original_image;
}

void Slic::set_original_image(Mat original_image)
{
  this->original_image = original_image;
}

vector<superpixel> Slic::get_superpixels()
{
  return this->superpixels;
}

Mat Slic::get_lab_image()
{
  return this->lab_image;
}

void Slic::set_lab_image(Mat lab_image)
{
  this->lab_image = lab_image;
}

Mat Slic::get_gradient_image()
{
  return this->gradient_image;
}

int Slic::get_k()
{
  return this->k;
}

void Slic::set_k(int k)
{
  this->k = k;
}

vector<factor_pair> Slic::factor_pairs(int k)
{
  vector<factor_pair> ret;
  for(int i = 1; i <= ((int)sqrt(k)); i++)
  {
    if(k % i == 0)
    {
      factor_pair p;
      p.x = i;
      p.y = k / i;
      ret.push_back(p);
    }
  }

  return ret;
}
