#ifndef LAPLACIAN_PYR
#define LAPLACIAN_PYR

#include <iostream>
#include <opencv2/opencv.hpp>
#include "GaussPyr.h"

class LaplacianPyr {
public:
    LaplacianPyr(const GaussPyr &p, float sigma);

    int getLayersSize() const {
        return _layers.size();
    }

    cv::Mat getLayer(int layer) const {
        return _layers[layer];
    }

private:
    std::vector<cv::Mat> _layers;
};

#endif
