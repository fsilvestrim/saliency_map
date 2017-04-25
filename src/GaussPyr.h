#include <iostream>
#include <opencv2/opencv.hpp>

class GaussPyr
{
public:
	GaussPyr(cv::Mat& img, int number_of_layers);
private:
	std::vector <cv::Mat> layers;
};
