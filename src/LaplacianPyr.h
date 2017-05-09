#ifndef LAPLACIAN_PYR
#define LAPLACIAN_PYR
#include <iostream>
#include <opencv2/opencv.hpp>
#include "GaussPyr.h"

class LaplacianPyr
{
public:
	LaplacianPyr(const GaussPyr& p, float sigma);
private:
        std::vector <cv::Mat> layers;
};
#endif
