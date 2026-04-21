#include "interpolation.h"
#include "skew.h"
#include "rotatescale.h"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    #ifdef _WIN32
        system("chcp 65001 >nul");
    #endif

    cv::Mat src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Ошибка: не удалось загрузить изображение.\n";
        return 1;
    }

    std::cout << "Исходное изображение: " << src.cols << "x" << src.rows << " (8 bpp)\n";
    
    std::vector<double> test_angles = {-360.0, -180.0, -90.0, -45.0, 0.0, 13.1, 30.0, 45.0, 69.9, 90.0, 180.0, 360.0};
    
    for (double angle : test_angles) {
        std::cout << "--- Угол поворота: " << angle << "° ---\n";
        std::cout << std::left << std::setw(18) << "Метод"
            << std::setw(24) << "Время(мс)"
            << std::setw(24) << "Размер"
            << std::setw(24) << "PSNR"
            << "Файл\n";

        auto t1 = std::chrono::high_resolution_clock::now();
        cv::Mat r_nn = rotateImageNearest(src, angle);
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms_nn = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::string f_nn = "rot_" + std::to_string(static_cast<int>(angle)) + "_nearest.png";
        cv::Mat rotate_img_nn = rotateImageNearest(r_nn, -angle);
        double psnr_nn = evaluateRotationQuality(src, rotate_img_nn);
        cv::imwrite(f_nn, r_nn);
        std::cout << std::setw(18) << "Nearest" << std::fixed << std::setprecision(2) << std::setw(14) << ms_nn
            << std::setw(14) << (std::to_string(r_nn.cols) + "x" + std::to_string(r_nn.rows)) << std::setw(14) << psnr_nn << f_nn << "\n";
        
        auto t3 = std::chrono::high_resolution_clock::now();
        cv::Mat r_bl = rotateImageBilinear(src, angle);
        auto t4 = std::chrono::high_resolution_clock::now();
        double ms_bl = std::chrono::duration<double, std::milli>(t4 - t3).count();
        cv::Mat rotate_img_bl = rotateImageBilinear(r_bl, -angle);
        double psnr_bl = evaluateRotationQuality(src, rotate_img_bl);
        std::string f_bl = "rot_" + std::to_string(static_cast<int>(angle)) + "_bilinear.png";
        cv::imwrite(f_bl, r_bl);
        std::cout << std::setw(18) << "Bilinear" << std::fixed << std::setprecision(2) << std::setw(14) << ms_bl
            << std::setw(14) << (std::to_string(r_bl.cols) + "x" + std::to_string(r_bl.rows)) << std::setw(14) << psnr_bl << f_bl << "\n";

        auto t5 = std::chrono::high_resolution_clock::now();
        cv::Mat r_bc = rotateImageBicubic(src, angle);
        auto t6 = std::chrono::high_resolution_clock::now();
        double ms_bc = std::chrono::duration<double, std::milli>(t6 - t5).count();
        cv::Mat rotate_img_bc = rotateImageBicubic(r_bc, -angle);
        double psnr_bc = evaluateRotationQuality(src, rotate_img_bc);
        std::string f_bc = "rot_" + std::to_string(static_cast<int>(angle)) + "_bicubic.png";
        cv::imwrite(f_bc, r_bc);
        std::cout << std::setw(18) << "Bicubic" << std::fixed << std::setprecision(2) << std::setw(14) << ms_bc
            << std::setw(14) << (std::to_string(r_bc.cols) + "x" + std::to_string(r_bc.rows)) << std::setw(14) << psnr_bc << f_bc << "\n\n";
            
    }
    
    
    std::vector<std::pair<double, double>> test_params = { {0.5, 0.0}, {0.0, 0.5}, {0.25, 0.25}, {0.9, 0.01}, {0.3, 0.3}, {0, 0} };
    for (auto [kx, ky] : test_params) {
        std::cout << "--- Скос: kx=" << kx << ", ky=" << ky << " ---\n";
        std::cout << std::left << std::setw(18) << "Метод"
            << std::setw(24) << "Время (мс)"
            << std::setw(24) << "Размер"
            << "Файл\n";

        auto t1 = std::chrono::high_resolution_clock::now();
        cv::Mat r_nn = skewImageNearest(src, kx, ky);
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms_nn = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::string f_nn = "skew_" + std::to_string((int)(kx * 100)) + "_" + std::to_string((int)(ky * 100)) + "_nearest.png";
        cv::imwrite(f_nn, r_nn);
        std::cout << std::setw(18) << "Nearest" << std::fixed << std::setprecision(2) << std::setw(14) << ms_nn
            << std::setw(16) << (std::to_string(r_nn.cols) + "x" + std::to_string(r_nn.rows)) << f_nn << "\n";

        auto t3 = std::chrono::high_resolution_clock::now();
        cv::Mat r_bl = skewImageBilinear(src, kx, ky);
        auto t4 = std::chrono::high_resolution_clock::now();
        double ms_bl = std::chrono::duration<double, std::milli>(t4 - t3).count();
        std::string f_bl = "skew_" + std::to_string((int)(kx * 100)) + "_" + std::to_string((int)(ky * 100)) + "_bilinear.png";
        cv::imwrite(f_bl, r_bl);
        std::cout << std::setw(18) << "Bilinear" << std::fixed << std::setprecision(2) << std::setw(14) << ms_bl
            << std::setw(16) << (std::to_string(r_bl.cols) + "x" + std::to_string(r_bl.rows)) << f_bl << "\n";

        auto t5 = std::chrono::high_resolution_clock::now();
        cv::Mat r_bc = skewImageBicubic(src, kx, ky);
        auto t6 = std::chrono::high_resolution_clock::now();
        double ms_bc = std::chrono::duration<double, std::milli>(t6 - t5).count();
        std::string f_bc = "skew_" + std::to_string((int)(kx * 100)) + "_" + std::to_string((int)(ky * 100)) + "_bicubic.png";
        cv::imwrite(f_bc, r_bc);
        std::cout << std::setw(18) << "Bicubic" << std::fixed << std::setprecision(2) << std::setw(14) << ms_bc
            << std::setw(16) << (std::to_string(r_bc.cols) + "x" + std::to_string(r_bc.rows)) << f_bc << "\n\n";

    }
    

    std::vector<std::pair<double, double>> test_params_ang = { {0.0, 2}, {0.0, 0.3}, {45.0, 3}, {45.0, 0.2}, {90.0, 1.5}, {90.0, 0.5} };
    int count = 0;
    for (auto [ang, sc] : test_params_ang) {
        std::cout << "--- Угол: " << ang << "°, Масштаб: x" << sc << " ---\n";
        std::cout << std::left << std::setw(18) << "Метод"
            << std::setw(24) << "Время (мс)"
            << std::setw(24) << "Размер"
            << "Файл\n";

        auto t1 = std::chrono::high_resolution_clock::now();
        cv::Mat r_nn = rotateScaleImageNearest(src, ang, sc);
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms_nn = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::string f_nn = "rotscale_" + std::to_string((int)ang) + "_" + std::to_string((int)(sc * 10)) + "_nearest.png";
        cv::imwrite(f_nn, r_nn);
        std::cout << std::setw(18) << "Nearest" << std::fixed << std::setprecision(2) << std::setw(14) << ms_nn
            << std::setw(16) << (std::to_string(r_nn.cols) + "x" + std::to_string(r_nn.rows)) << f_nn << "\n";

        auto t3 = std::chrono::high_resolution_clock::now();
        cv::Mat r_bl = rotateScaleImageBilinear(src, ang, sc);
        auto t4 = std::chrono::high_resolution_clock::now();
        double ms_bl = std::chrono::duration<double, std::milli>(t4 - t3).count();
        std::string f_bl = "rotscale_" + std::to_string((int)ang) + "_" + std::to_string((int)(sc * 10)) + "_bilinear.png";
        cv::imwrite(f_bl, r_bl);
        std::cout << std::setw(18) << "Bilinear" << std::fixed << std::setprecision(2) << std::setw(14) << ms_bl
            << std::setw(16) << (std::to_string(r_bl.cols) + "x" + std::to_string(r_bl.rows)) << f_bl << "\n";

        auto t5 = std::chrono::high_resolution_clock::now();
        cv::Mat r_bc = rotateScaleImageBicubic(src, ang, sc);
        auto t6 = std::chrono::high_resolution_clock::now();
        double ms_bc = std::chrono::duration<double, std::milli>(t6 - t5).count();
        std::string f_bc = "rotscale_" + std::to_string((int)ang) + "_" + std::to_string((int)(sc * 10)) + "_bicubic.png";
        cv::imwrite(f_bc, r_bc);
        std::cout << std::setw(18) << "Bicubic" << std::fixed << std::setprecision(2) << std::setw(14) << ms_bc
            << std::setw(16) << (std::to_string(r_bc.cols) + "x" + std::to_string(r_bc.rows)) << f_bc << "\n\n";

        count = count + 1;
    }
    
    
    return 0;
}