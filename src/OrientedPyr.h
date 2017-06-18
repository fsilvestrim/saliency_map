#ifndef ORIEN_PYR
#define ORIEN_PYR

#include <iostream>
#include <opencv2/opencv.hpp>
#include "LaplacianPyr.h"

class OrientedPyr {
public:
    OrientedPyr(const LaplacianPyr &p,
                int num_orientations,
                int filter_size = 25,
                double sigma = 10.0,
                double lambd = 10.0,
                double gamma = 1.0,
                double psi = 0.0);

    std::vector<std::vector<cv::Mat> > orientationMaps;
};

#endif