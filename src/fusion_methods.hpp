#ifndef FUSION_METHODS
#define FUSION_METHODS

#include <iostream>
#include <opencv2/opencv.hpp>

cv::Mat fusion_arithmetic_mean(const std::vector<cv::Mat> &scale_images) {
    cv::Mat add;
    double max;

    for (int i = 0; i < scale_images.size(); ++i) {
        cv::Mat img = scale_images[i].clone();
        double min_val, max_val;
        cv::minMaxLoc(img, &min_val, &max_val);
        max = std::max(max, max_val);

        if (i == 0) add = cv::Mat::zeros(img.size(), CV_32F);
        add += img;
    }

    cv::Mat result = add / 2.0f;
    cv::normalize(result, result, 0.0f, (float) max, cv::NORM_MINMAX);

    return result;
}

cv::Mat fusion_across_scale_addition(const std::vector<cv::Mat> &scale_images) {
    cv::Size im_size = scale_images[0].size();
    cv::Mat add = scale_images[0];
    for (int i = 0; i < scale_images.size(); ++i) {
        cv::Mat resized;
        if (scale_images[i].size() != im_size) {
            cv::resize(scale_images[i], resized, im_size, 0, 0, cv::INTER_CUBIC);
            add += resized;
        }
    }

    return add;
}
#endif