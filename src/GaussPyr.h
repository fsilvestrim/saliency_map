#ifndef GAUSS_PYR
#define GAUSS_PYR

#include <iostream>
#include <opencv2/opencv.hpp>

class GaussPyr {
public:
    GaussPyr(cv::Mat &img, int number_of_layers, float sigma, bool convert2Float32 = false);

    int getLayersSize() const {
        return _layers.size();
    }

    cv::Mat getLayer(int layer) const {
        return _layers[layer];
    }

    std::vector<cv::Mat> getLayers() const {
        return _layers;
    }

private:
    std::vector<cv::Mat> _layers;
};

#endif
