#include <iostream>
#include "elas/elas.h"
#include "elas/image.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

cv::Mat convertDisparity(cv::Mat const &disparity) {
    double minVal, maxVal;
    cv::minMaxIdx(disparity, &minVal, &maxVal);
    auto mask = disparity > 0;

    auto range = maxVal - minVal;
    cv::Mat disparity2, disparityDisplay;
    disparity.copyTo(disparity2, mask);
    disparity2.convertTo(disparityDisplay, CV_8U, 255.0 / maxVal);
    return disparityDisplay;
}

int main(int argc, char **argv) {
    auto I1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    auto I2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    cv::imshow("Image 1", I1);
    cv::imshow("Image 2", I2);
    // get image width and height
    int32_t width = I1.size().width;
    int32_t height = I1.size().height;

    // allocate memory for disparity images
    const int32_t dims[3] = {width, height, width};  // bytes per line = width

    cv::Mat D1 = cv::Mat(cv::Size(width, height), CV_32FC1);
    cv::Mat D2 = cv::Mat(cv::Size(width, height), CV_32FC1);

    float *D1_data = reinterpret_cast<float *>(D1.data);
    float *D2_data = reinterpret_cast<float *>(D2.data);

    // process
    Elas::parameters param;
    param.postprocess_only_left = false;
    Elas elas(param);
    elas.process(I1.data, I2.data, D1_data, D2_data, dims);

    auto dis = convertDisparity(D1);

    cv::imshow("M.ALI: Dis", dis);
    cv::waitKey();
}