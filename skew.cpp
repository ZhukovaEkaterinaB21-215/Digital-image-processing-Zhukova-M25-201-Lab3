#include "interpolation.h"
#include "skew.h"
#include <cmath>
#include <algorithm>
#include <vector>

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

struct TransformBounds {
    int w, h;
    double offset_x, offset_y;
};

TransformBounds calcBounds(const cv::Size& sz,
    std::vector<cv::Point2d>& corners,
    std::function<void(double, double, double&, double&)> forward) {
    double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
    for (auto& p : corners) {
        double nx, ny;
        forward(p.x, p.y, nx, ny);
        p = cv::Point2d(nx, ny);
        min_x = std::min(min_x, nx); max_x = std::max(max_x, nx);
        min_y = std::min(min_y, ny); max_y = std::max(max_y, ny);
    }
    return {
        static_cast<int>(std::ceil(max_x - min_x)),
        static_cast<int>(std::ceil(max_y - min_y)),
        min_x, min_y
    };
}

void forwardSkew(double x, double y, double kx, double ky, double& nx, double& ny) {
    nx = x + kx * y;
    ny = ky * x + y;
}


cv::Mat skewImageNearest(const cv::Mat& src, double shear_x, double shear_y) {
    if (src.empty()) {
        throw std::invalid_argument("Ошибка: исходное изображение пустое или не было загружено.");
    }
    if (src.type() != CV_8UC1) {
        throw std::invalid_argument("Ошибка: поддерживается только полутоновой формат 8 bpp (1 канал).");
    }
    double det = 1.0 - shear_x * shear_y;
    if (std::abs(det) <= 1e-6) {
        throw std::invalid_argument("Скос: вырожденная матрица (|1 - kx*ky| слишком мало). Измените коэффициенты.");
    }

    std::vector<cv::Point2d> corners = { {0,0}, {(double)src.cols,0}, {0,(double)src.rows}, {(double)src.cols,(double)src.rows} };
    auto bounds = calcBounds(src.size(), corners, [&](double x, double y, double& nx, double& ny) { forwardSkew(x, y, shear_x, shear_y, nx, ny); });

    cv::Mat dst(bounds.h, bounds.w, CV_8UC1, cv::Scalar(255));
    for (int y = 0; y < dst.rows; ++y) {
        for (int x = 0; x < dst.cols; ++x) {
            double fx = x + bounds.offset_x, fy = y + bounds.offset_y;
            // Обратное преобразование скоса
            double sx = (fx - shear_x * fy) / det;
            double sy = (fy - shear_y * fx) / det;
            dst.at<uint8_t>(y, x) = getPixelWhiteBG(src, static_cast<int>(std::round(sx)), static_cast<int>(std::round(sy)));
        }
    }
    return dst;
}

cv::Mat skewImageBilinear(const cv::Mat& src, double shear_x, double shear_y) {
    if (src.empty()) {
        throw std::invalid_argument("Ошибка: исходное изображение пустое или не было загружено.");
    }
    if (src.type() != CV_8UC1) {
        throw std::invalid_argument("Ошибка: поддерживается только полутоновой формат 8 bpp (1 канал).");
    }
    double det = 1.0 - shear_x * shear_y;
    if (std::abs(det) <= 1e-6) {
        throw std::invalid_argument("Скос: вырожденная матрица (|1 - kx*ky| слишком мало). Измените коэффициенты.");
    }

    std::vector<cv::Point2d> corners = { {0,0}, {(double)src.cols,0}, {0,(double)src.rows}, {(double)src.cols,(double)src.rows} };
    auto bounds = calcBounds(src.size(), corners, [&](double x, double y, double& nx, double& ny) { forwardSkew(x, y, shear_x, shear_y, nx, ny); });

    cv::Mat dst(bounds.h, bounds.w, CV_8UC1, cv::Scalar(255));
    for (int y = 0; y < dst.rows; ++y) {
        for (int x = 0; x < dst.cols; ++x) {
            double fx = x + bounds.offset_x, fy = y + bounds.offset_y;
            double sx = (fx - shear_x * fy) / det;
            double sy = (fy - shear_y * fx) / det;

            int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
            double fx_d = sx - x0, fy_d = sy - y0;
            double val = (1 - fx_d) * (1 - fy_d) * getPixelWhiteBG(src, x0, y0) + fx_d * (1 - fy_d) * getPixelWhiteBG(src, x0 + 1, y0) +
                (1 - fx_d) * fy_d * getPixelWhiteBG(src, x0, y0 + 1) + fx_d * fy_d * getPixelWhiteBG(src, x0 + 1, y0 + 1);
            dst.at<uint8_t>(y, x) = clampPixel(val);
        }
    }
    return dst;
}

cv::Mat skewImageBicubic(const cv::Mat& src, double shear_x, double shear_y) {
    if (src.empty()) {
        throw std::invalid_argument("Ошибка: исходное изображение пустое или не было загружено.");
    }
    if (src.type() != CV_8UC1) {
        throw std::invalid_argument("Ошибка: поддерживается только полутоновой формат 8 bpp (1 канал).");
    }
    double det = 1.0 - shear_x * shear_y;
    if (std::abs(det) <= 1e-6) {
        throw std::invalid_argument("Скос: вырожденная матрица (|1 - kx*ky| слишком мало). Измените коэффициенты.");
    }

    std::vector<cv::Point2d> corners = { {0,0}, {(double)src.cols,0}, {0,(double)src.rows}, {(double)src.cols,(double)src.rows} };
    auto bounds = calcBounds(src.size(), corners, [&](double x, double y, double& nx, double& ny) { forwardSkew(x, y, shear_x, shear_y, nx, ny); });

    cv::Mat dst(bounds.h, bounds.w, CV_8UC1, cv::Scalar(255));
    for (int y = 0; y < dst.rows; ++y) {
        for (int x = 0; x < dst.cols; ++x) {
            double fx = x + bounds.offset_x, fy = y + bounds.offset_y;
            double sx = (fx - shear_x * fy) / det;
            double sy = (fy - shear_y * fx) / det;

            int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
            double fx_d = sx - x0, fy_d = sy - y0;
            double val = 0.0;
            for (int m = -1; m <= 2; ++m)
                for (int n = -1; n <= 2; ++n)
                    val += getPixelWhiteBG(src, x0 + m, y0 + n) * cubicKernel(fx_d - m) * cubicKernel(fy_d - n);
            dst.at<uint8_t>(y, x) = clampPixel(val);
        }
    }
    return dst;
}


