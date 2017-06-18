#include "GaussPyr.h"

GaussPyr::GaussPyr(cv::Mat &img, int number_of_layers, float sigma, bool convert2Float32) {
    cv::Mat dst = img.clone(); //deep copy to not modify original image
    for (int i = 0; i < number_of_layers; ++i) {
        cv::GaussianBlur(dst, dst, cv::Size(0, 0), sigma, 0, cv::BORDER_REPLICATE);
        _layers.push_back(dst.clone()); // remember to deep copy!
        //std::cout << "layer " << i << " size " << _layers.back().size() << std::endl;
        cv::resize(dst, dst, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
        if (convert2Float32) dst.convertTo(dst, CV_32F);
        //std::cout << "Pyramid has " << _layers.size() << " layers." << std::endl;
    }
}
