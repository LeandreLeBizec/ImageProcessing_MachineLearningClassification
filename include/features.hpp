#ifndef PROJET_OPENCV_CMAKE_FEATURES_HPP
#define PROJET_OPENCV_CMAKE_FEATURES_HPP

#include <dirent.h>
#include <regex>
#include <fstream>
#include <queue>

#include <iostream>
using namespace std;
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
using namespace cv;



class Features {
private:
    vector<vector<double>> mat;
    vector<string> className;
public:
    Features();
    vector<vector<double>> getFeaturesMatrice();
    vector<string> getClassNameVector();
    Mat loadImage(string path);
    Mat image_BGRtoBIN(const Mat& im);
    Rect findBoundingBox(const Mat& im);
    Mat displayBoundingBox(const Mat& im);
    bool isEmpty(const Mat& im);
    Mat croppedImage(const Mat& im);
    tuple<double,double> findGravityCenter(const Mat& im);
    Mat displayGravityCenter(const Mat& im);
    Mat removeNoise(const Mat &img);
    double findPixelDensity(const Mat& im);
    vector<Mat> zoning(const Mat& im, int row, int col);
    vector<KeyPoint> findSift(const Mat& im, int size);
    HOGDescriptor findHogVector(const Mat& im);
    void normalize();
    void generateFeaturesMatrice(const char *path, int row, int col);
    void exportArff(string arffName, int row, int col);
};


#endif //PROJET_OPENCV_CMAKE_FEATURES_HPP
