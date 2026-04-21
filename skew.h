#pragma once
#include <opencv2/core.hpp>

cv::Mat skewImageNearest(const cv::Mat& src, double shear_x, double shear_y);
cv::Mat skewImageBilinear(const cv::Mat& src, double shear_x, double shear_y);
cv::Mat skewImageBicubic(const cv::Mat& src, double shear_x, double shear_y);

double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2);