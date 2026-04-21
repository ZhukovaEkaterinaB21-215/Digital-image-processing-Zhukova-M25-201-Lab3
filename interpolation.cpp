#include "interpolation.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>


#ifndef CV_PI
#define CV_PI 3.14159265358979323846
#endif


inline uint8_t clampPixel(double v) {
    return static_cast<uint8_t>(std::max(0.0, std::min(255.0, v)));
}

inline uint8_t getPixelWhiteBG(const cv::Mat& img, int x, int y) {
    if (x < 0 || x >= img.cols || y < 0 || y >= img.rows) return 255;
    return img.at<uint8_t>(y, x);
}

inline double cubicKernel(double x, double a = -0.5) {
    double ax = std::abs(x);
    if (ax < 1.0) return (a + 2) * ax * ax * ax - (a + 3) * ax * ax + 1;
    if (ax < 2.0) return a * ax * ax * ax - 5 * a * ax * ax + 8 * a * ax - 4 * a;
    return 0.0;
}

cv::Size calcRotatedSize(const cv::Size& src_sz, double angle_rad) {
    double cx = src_sz.width / 2.0;
    double cy = src_sz.height / 2.0;
    double cos_a = std::cos(angle_rad);
    double sin_a = std::sin(angle_rad);

    double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
    std::vector<cv::Point2f> corners = {
        {0, 0}, {(float)src_sz.width, 0},
        {0, (float)src_sz.height}, {(float)src_sz.width, (float)src_sz.height}
    };
    for (const auto& p : corners) {
        double dx = p.x - cx;
        double dy = p.y - cy;
        double nx = dx * cos_a - dy * sin_a;
        double ny = dx * sin_a + dy * cos_a;
        min_x = std::min(min_x, nx); max_x = std::max(max_x, nx);
        min_y = std::min(min_y, ny); max_y = std::max(max_y, ny);
    }
    return cv::Size(static_cast<int>(std::ceil(max_x - min_x)),
        static_cast<int>(std::ceil(max_y - min_y)));
}

inline void computeSourceCoords(double dst_x, double dst_y,
    double src_cx, double src_cy,
    double dst_cx, double dst_cy,
    double cos_a, double sin_a,
    double& out_x, double& out_y) {
    double dx = dst_x - dst_cx;
    double dy = dst_y - dst_cy;
    out_x = src_cx + dx * cos_a + dy * sin_a;
    out_y = src_cy - dx * sin_a + dy * cos_a;
}


cv::Mat rotateImageNearest(const cv::Mat& src, double angle_deg) {

    if (src.empty()) {
        throw std::invalid_argument("Ошибка: исходное изображение пустое или не было загружено.");
    }
    if (src.type() != CV_8UC1) {
        throw std::invalid_argument("Ошибка: поддерживается только полутоновой формат 8 bpp (1 канал).");
    }
    double angle_rad = angle_deg * CV_PI / 180.0;
    cv::Size dst_sz = calcRotatedSize(src.size(), angle_rad);
    cv::Mat dst(dst_sz.height, dst_sz.width, CV_8UC1, cv::Scalar(255));

    double src_cx = src.cols / 2.0, src_cy = src.rows / 2.0;
    double dst_cx = dst.cols / 2.0, dst_cy = dst.rows / 2.0;
    double cos_a = std::cos(angle_rad), sin_a = std::sin(angle_rad);

    for (int y = 0; y < dst.rows; ++y) {
        for (int x = 0; x < dst.cols; ++x) {
            double sx, sy;
            computeSourceCoords(x, y, src_cx, src_cy, dst_cx, dst_cy, cos_a, sin_a, sx, sy);
            dst.at<uint8_t>(y, x) = getPixelWhiteBG(src, static_cast<int>(std::round(sx)), static_cast<int>(std::round(sy)));
        }
    }
    return dst;
}

cv::Mat rotateImageBilinear(const cv::Mat& src, double angle_deg) {
    if (src.empty()) {
        throw std::invalid_argument("Ошибка: исходное изображение пустое или не было загружено.");
    }
    if (src.type() != CV_8UC1) {
        throw std::invalid_argument("Ошибка: поддерживается только полутоновой формат 8 bpp (1 канал).");
    }
    double angle_rad = angle_deg * CV_PI / 180.0;
    cv::Size dst_sz = calcRotatedSize(src.size(), angle_rad);
    cv::Mat dst(dst_sz.height, dst_sz.width, CV_8UC1, cv::Scalar(255));

    double src_cx = src.cols / 2.0, src_cy = src.rows / 2.0;
    double dst_cx = dst.cols / 2.0, dst_cy = dst.rows / 2.0;
    double cos_a = std::cos(angle_rad), sin_a = std::sin(angle_rad);

    for (int y = 0; y < dst.rows; ++y) {
        for (int x = 0; x < dst.cols; ++x) {
            double sx, sy;
            computeSourceCoords(x, y, src_cx, src_cy, dst_cx, dst_cy, cos_a, sin_a, sx, sy);

            int x0 = static_cast<int>(std::floor(sx));
            int y0 = static_cast<int>(std::floor(sy));
            double fx = sx - x0, fy = sy - y0;

            double val = (1 - fx) * (1 - fy) * getPixelWhiteBG(src, x0, y0) +
                fx * (1 - fy) * getPixelWhiteBG(src, x0 + 1, y0) +
                (1 - fx) * fy * getPixelWhiteBG(src, x0, y0 + 1) +
                fx * fy * getPixelWhiteBG(src, x0 + 1, y0 + 1);
            dst.at<uint8_t>(y, x) = clampPixel(val);
        }
    }
    return dst;
}

cv::Mat rotateImageBicubic(const cv::Mat& src, double angle_deg) {
    if (src.empty()) {
        throw std::invalid_argument("Ошибка: исходное изображение пустое или не было загружено.");
    }
    if (src.type() != CV_8UC1) {
        throw std::invalid_argument("Ошибка: поддерживается только полутоновой формат 8 bpp (1 канал).");
    }
    double angle_rad = angle_deg * CV_PI / 180.0;
    cv::Size dst_sz = calcRotatedSize(src.size(), angle_rad);
    cv::Mat dst(dst_sz.height, dst_sz.width, CV_8UC1, cv::Scalar(255));

    double src_cx = src.cols / 2.0, src_cy = src.rows / 2.0;
    double dst_cx = dst.cols / 2.0, dst_cy = dst.rows / 2.0;
    double cos_a = std::cos(angle_rad), sin_a = std::sin(angle_rad);

    for (int y = 0; y < dst.rows; ++y) {
        for (int x = 0; x < dst.cols; ++x) {
            double sx, sy;
            computeSourceCoords(x, y, src_cx, src_cy, dst_cx, dst_cy, cos_a, sin_a, sx, sy);

            int x0 = static_cast<int>(std::floor(sx));
            int y0 = static_cast<int>(std::floor(sy));
            double fx = sx - x0, fy = sy - y0;

            double val = 0.0;
            for (int m = -1; m <= 2; ++m) {
                for (int n = -1; n <= 2; ++n) {
                    double wx = cubicKernel(fx - m);
                    double wy = cubicKernel(fy - n);
                    val += getPixelWhiteBG(src, x0 + m, y0 + n) * wx * wy;
                }
            }
            dst.at<uint8_t>(y, x) = clampPixel(val);
        }
    }
    return dst;
}


double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.empty() || img2.empty() ||
        img1.rows != img2.rows || img1.cols != img2.cols ||
        img1.type() != CV_8UC1 || img2.type() != CV_8UC1) {
        throw std::runtime_error("PSNR: изображения должны быть одинакового размера и формата 8 bpp grayscale.");
    }

    double sumSquaredDiff = 0.0;
    int totalPixels = img1.rows * img1.cols;

    for (int y = 0; y < img1.rows; ++y) {
        const uint8_t* ptr1 = img1.ptr<uint8_t>(y);
        const uint8_t* ptr2 = img2.ptr<uint8_t>(y);

        for (int x = 0; x < img1.cols; ++x) {
            double diff = static_cast<double>(ptr1[x]) - ptr2[x];
            sumSquaredDiff += diff * diff;
        }
    }

    double mse = sumSquaredDiff / totalPixels;
    if (mse < 1e-10) return 1e10; // Защита от деления на ноль при полном совпадении
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}

double evaluateRotationQuality(const cv::Mat& src, const cv::Mat& rotate_img) {
    if (src.empty()) {
        throw std::invalid_argument("Ошибка: исходное изображение пустое или не было загружено.");
    }
    if (src.type() != CV_8UC1) {
        throw std::invalid_argument("Ошибка: поддерживается только полутоновой формат 8 bpp (1 канал).");
    }

    cv::Mat aligned(src.size(), CV_8UC1, cv::Scalar(255));

    int offsetX = (rotate_img.cols - src.cols) / 2;
    int offsetY = (rotate_img.rows - src.rows) / 2;


    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            
            int rx = x + offsetX;
            int ry = y + offsetY;

            if (rx >= 0 && rx < rotate_img.cols && ry >= 0 && ry < rotate_img.rows) {
                aligned.at<uint8_t>(y, x) = rotate_img.at<uint8_t>(ry, rx);
            }
        }
    }

    return calculatePSNR(src, aligned);
}
