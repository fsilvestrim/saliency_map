#include "OrientedPyr.h"

OrientedPyr::OrientedPyr(const LaplacianPyr &p,
                         int num_orientations,
                         int filter_size,
                         double sigma,
                         double lambd,
                         double gamma,
                         double psi) {

    for (int o = 0; o < num_orientations; ++o)
    {
        std::vector<cv::Mat> garborLayer;
        double theta = (o * CV_PI) / num_orientations;
        cv::Mat gabor_kernel = cv::getGaborKernel( cv::Size(filter_size,filter_size), sigma, theta, lambd, gamma, psi, CV_32F );

        for ( int i = 0; i < p.getLayersSize(); ++i)
        {
            cv::Mat dest;
            cv::filter2D(p.getLayer(i).clone(), dest, -1, gabor_kernel);
            garborLayer.push_back(dest);
        }

        orientationMaps.push_back(garborLayer);
    }
}
