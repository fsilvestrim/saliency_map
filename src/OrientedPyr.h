#ifndef ORIEN_PYR
#define ORIEN_PYR

#include <iostream>
#include <opencv2/opencv.hpp>
#include "LaplacianPyr.h"

class OrientedPyr {
public:
    OrientedPyr(const LaplacianPyr &p, int num_orientations);

    std::vector<std::vector<cv::Mat> > orientationMaps;
};

#endif