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

using namespace cv;
using namespace std;

struct label_count
{
  float label;
  int count;
};

// Reference: http://docs.opencv.org/modules/ml/doc/support_vector_machines.html
// SVM Types
enum SVM_TYPE
{
  C_SVC = 1,
  NU_SVC = 2,
  ONE_CLASS = 3,
  EPS_SVR = 4,
  NU_SVR = 5
};

// Kernel Types
enum KERNEL_TYPE
{
  LINEAR = 1,
  POLY = 2,
  RBF = 3,
  SIGMOID = 4
};

ifstream data_file;

void syntax();
vector<float> get_labels(string filename);
vector<float> get_possible_labels(vector<float> labels);
int get_dimensions(string filename);
Mat get_labels_matrix(vector<float> labels);
Mat get_features_matrix(string filename, int rows, int dimensions);
vector<label_count> get_label_count(vector<float> labels, vector<float> possible_labels);

int main(int argc, char **argv)
{
  if(argc != 3)
  {
    syntax();
    exit(-1);
  }

  string data_filename = argv[1];
  string model_filename = argv[2];

  int num_lines = 0;
  vector<float> labels;
  vector<float> possible_labels;

  cout << "Extracting labels..." << endl;
  labels = get_labels(data_filename);

  int dimensions = get_dimensions(data_filename);
  cout << "Number of dimensions: " << dimensions << endl;

  // Get sample size
  cout << "Number of samples: " << labels.size() << endl;

  // Unique labels
  cout << "Getting unique labels..." << endl;
  possible_labels = get_possible_labels(labels);

  // Get stats
  vector<label_count> label_counts = get_label_count(labels, possible_labels);

  cout << "Possible labels:" << endl;
  for(int i = 0; i < possible_labels.size(); i++) 
  {
    cout << "==> " << possible_labels.at(i) << " (" << label_counts.at(i).count << ")" << endl;
  }  

  cout << "Generating labels..." << endl;
  Mat labels_matrix = get_labels_matrix(labels);

  cout << "Extracting features..." << endl;
  Mat features_matrix = get_features_matrix(data_filename, labels.size(), dimensions);

  CvSVMParams params;
  int svm_type;
  int kernel_type;
  double degree;
  double gamma;
  double coef;
  double c_value;

  cout << "SVM TYPE: ";
  cin >> svm_type;
  switch(svm_type)
  {
    case C_SVC:
      params.svm_type = CvSVM::C_SVC;

      cout << "C VALUE: ";
      cin >> c_value;
      params.C = c_value;

      break;
    case NU_SVC:
      params.svm_type = CvSVM::NU_SVC;
      break;
    case ONE_CLASS:
      params.svm_type = CvSVM::ONE_CLASS;
      break;
    case EPS_SVR:
      params.svm_type = CvSVM::EPS_SVR;

      cout << "C VALUE: ";
      cin >> c_value;
      params.C = c_value;

      break;
    case NU_SVR:
      params.svm_type = CvSVM::NU_SVR;

      cout << "C VALUE: ";
      cin >> c_value;
      params.C = (double)c_value;

      break;
    default:
      params.svm_type = CvSVM::C_SVC;
      break;
  }

  cout << "KERNEL TYPE: ";
  cin >> kernel_type;
  switch(kernel_type)
  {
    case LINEAR:
      params.kernel_type = CvSVM::LINEAR;
      break;
    case POLY:
      params.kernel_type = CvSVM::POLY;

      cout << "DEGREE: ";
      cin >> degree;
      params.degree = degree;

      cout << "GAMMA: ";
      cin >> gamma;
      params.gamma = gamma;

      cout << "COEF: ";
      cin >> coef;
      params.coef0 = coef;

      break;
    case RBF:
      params.kernel_type = CvSVM::RBF;

      cout << "GAMMA: ";
      cin >> gamma;
      params.gamma = gamma;

      break;
    case SIGMOID:
      params.kernel_type = CvSVM::SIGMOID;

      cout << "GAMMA: ";
      cin >> gamma;
      params.gamma = gamma;

      cout << "COEF: ";
      cin >> coef;
      params.coef0 = coef;

      break;
    default:
      params.kernel_type = CvSVM::LINEAR;
      break;
  }

  cout << "ITERATIONS: ";
  int iterations;
  cin >> iterations;
  params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, iterations, FLT_EPSILON);

  cout << "Training..." << endl;
  CvSVM SVM;
  SVM.train_auto(features_matrix, labels_matrix, Mat(), Mat(), params);

  cout << "Done training!" << endl;
  SVM.save(model_filename.c_str());

  return 0;
}

void syntax()
{
  cout << "Syntax: svmtrain [data_file] [model_file]" << endl;
}

int get_dimensions(string filename)
{
  ifstream data_file;
  int dimensions = 0;

  data_file.open(filename.c_str());
  if(!data_file.is_open())
  {
    cout << "Cannot open file " << filename << endl;
    exit(-1);
  }

  while(!data_file.eof())
  {
    string data_line;
    getline(data_file, data_line);

    string string_value;
    stringstream stream(data_line);

    dimensions = 0;
    while(getline(stream, string_value, ',')) 
    {
      dimensions++;
    }
  }
  
  data_file.close();

  return dimensions - 1;
}

vector<float> get_labels(string filename)
{
  ifstream data_file;
  vector<float> labels;

  data_file.open(filename.c_str());
  if(!data_file.is_open())
  {
    cout << "Cannot open file " << filename << endl;
    exit(-1);
  }

  while(!data_file.eof())
  {
    string data_line;
    getline(data_file, data_line);

    string string_value;
    stringstream stream(data_line);

    // Parse the line and store it as a string in string_value
    float val = 0;
    while(getline(stream, string_value, ',')) 
    {
      // Convert the value to a float
      val = ::atof(string_value.c_str());
    }

    labels.push_back(val);
  }

  data_file.close();

  return labels;
}

vector<float> get_possible_labels(vector<float> labels)
{
  vector<float> possible_labels;
  for(int i = 0; i < labels.size(); i++) 
  {
    bool unique = true;
    if(possible_labels.size() == 0) 
    {
      possible_labels.push_back(labels.at(i));
      unique = false;
    }
    else 
    {
      for(int j = 0; j < possible_labels.size(); j++) 
      {
        if(possible_labels.at(j) == labels.at(i)) 
        {
          unique = false;
        }
      }
    }

    if(unique) 
    {
      possible_labels.push_back(labels.at(i));
    }
  }

  return possible_labels;
}

Mat get_labels_matrix(vector<float> labels)
{
  Mat label_mat = Mat_<float>(Size(1, labels.size()));
  for(int i = 0; i < labels.size(); i++)
  {
    label_mat.at<float>(i, 0) = labels.at(i);
  }

  return label_mat;
}

Mat get_features_matrix(string data_filename, int rows, int dimensions)
{
  Mat features_matrix(rows, dimensions, DataType<float>::type);
  data_file.open(data_filename.c_str());
  if(!data_file.is_open())
  {
    cout << "Cannot open file " << data_filename << endl;
    exit(-1);
  }

  int index = 0;
  while(!data_file.eof())
  {
    string data_line;
    getline(data_file, data_line);

    string string_value;
    stringstream stream(data_line);

    int counter = 0;
    while(getline(stream, string_value, ',')) 
    {
      float val = ::atof(string_value.c_str());
      if(counter < dimensions) 
      {
        features_matrix.at<float>(index, counter) = val;
      }
      counter++;
    }
    index++;
  }

  data_file.close();

  return features_matrix;
}

vector<label_count> get_label_count(vector<float> labels, vector<float> possible_labels)
{
  vector<label_count> label_counts;

  for(int i = 0; i < possible_labels.size(); i++)
  {
    label_count lc;
    lc.label = possible_labels.at(i);
    lc.count = 0;
    for(int j = 0; j < labels.size(); j++)
    {
      if(labels.at(j) == possible_labels.at(i))
      {
        lc.count++;
      }
    }
    label_counts.push_back(lc);
  }

  return label_counts;
}