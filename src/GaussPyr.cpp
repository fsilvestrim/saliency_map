#include "GaussPyr.h"

GaussPyr::GaussPyr(cv::Mat& img, int number_of_layers)
{
	std::cout << "Test" << std::endl;
	cv::Mat layerRet;
	for (int i = 0; i < number_of_layers; i++)
	{
		cv::GaussianBlur(img, layerRet, cv::Size(5,5), 0);
	} 
}
