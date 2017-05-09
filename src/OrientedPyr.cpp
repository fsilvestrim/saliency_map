#include "OrientedPyr.h"

OrientedPyr::OrientedPyr(const LaplacianPyr& p, float num_orientations)
{
	for (int m = 0; m < p.Layers.size(); ++m)
	{
		std::vector<cv::Mat> garborLayer;

		for (int n = 0; n < num_orientations; ++n)
        	{
			cv::Mat kernel = cv::getGaborKernel(cv::Size(20, 20), 10, 8.0/i, 10, 1, 0, CV_32F);
        		cv::Mat dest;
		        cv::filter2D(p.Layers[m].clone(), dest, CV_32F, kernel);
			
			garborLayer.push_back(dest);
		}
		
		orientationMaps.push_back(garborLayer);
	}
}
