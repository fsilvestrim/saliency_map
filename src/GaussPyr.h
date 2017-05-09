#ifndef GAUSS_PYR
#define GAUSS_PYR
#include <iostream>
#include <opencv2/opencv.hpp>

class GaussPyr
{
public:
	GaussPyr(cv::Mat& img, int number_of_layers, float sigma);
	cv::Mat get(int layer);

	const std::vector<cv::Mat>& Layers()
	{
		return layers;
	}

private:
	std::vector <cv::Mat> layers;
};
#endif
