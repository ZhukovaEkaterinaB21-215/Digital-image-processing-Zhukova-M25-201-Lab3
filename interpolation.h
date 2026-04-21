#pragma once
#include <opencv2/core.hpp>

cv::Mat rotateImageNearest(const cv::Mat& src, double angle_deg);

cv::Mat rotateImageBilinear(const cv::Mat& src, double angle_deg);

cv::Mat rotateImageBicubic(const cv::Mat& src, double angle_deg);

double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2);

double evaluateRotationQuality(const cv::Mat& src, const cv::Mat& rotate_img);