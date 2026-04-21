#pragma once
#include <opencv2/core.hpp>

cv::Mat rotateScaleImageNearest(const cv::Mat& src, double angle_deg, double scale);
cv::Mat rotateScaleImageBilinear(const cv::Mat& src, double angle_deg, double scale);
cv::Mat rotateScaleImageBicubic(const cv::Mat& src, double angle_deg, double scale);
