#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "GaussPyr.h"
#include <vector>

cv::Mat across_scale_addition(const std::vector<cv::Mat>& scale_images)
{
	cv::Size s;
	for (std::vector<cv::Mat>::iterator it = scale_images.begin() ; it != scale_images.end(); ++it)
	{
		s = *it->size() > s ? *it->size() : s;
	}

    cv::Mat feat = cv::Mat(s, CV_32F);

	for (std::vector<cv::Mat>::iterator it = scale_images.begin() ; it != scale_images.end(); ++it)
    {
		cv::Mat mat = *it->clone();
        cv::resize(mat, mat, s, 0, 0, cv::INTER_CUBIC);
        feat += mat;
    }

	return feat;
}

void showPyramid(GaussPyr pc, GaussPyr ps, int numLayers)
{
	std::vector<cv::Mat> mCsC;
	std::vector<cv::Mat> mSCC;
 
    for (int l = 0; l < numLayers; ++l)
    {
        cv::Mat csC = pc.get(l) - ps.get(l);
        cv::threshold(csC, csC, 0, 1, cv::THRESH_TOZERO);
		mCsC.push_back(csC);

        cv::imshow("Gaussian pyramid CS Contrast", csC ); 
        
		cv::Mat scC = ps.get(l) - pc.get(l);
        cv::threshold(scC , scC, 0, 1, cv::THRESH_TOZERO);
		mSCC.push_back(scC); 

        cv::imshow("Gaussian pyramid SC Contrast", scC ); 

		cv::waitKey(0);
    }
	
	cv::imshow("Feature Map Center", across_scale_addition(mCsC));
	cv::imshow("Feature Map Surround", across_scale_addition(mSCC));
	cv::waitKey(0);
}

int main()
{
	// load img
	cv::Mat image;
    image = cv::imread("../img/benchmark/i1.jpg" , CV_LOAD_IMAGE_COLOR);

	if(!image.data)
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

	cv::Mat image_lab;
	cv::cvtColor(image, image_lab, CV_RGB2Lab);

	image_lab.convertTo(image_lab, CV_32F);
	image_lab /= 255.0f;

	cv::Mat lab_channels[3];
	cv::split(image_lab, lab_channels); 

	//
	int numLayers = 4;
	int stdC = 5;
	int stdS = 20;
	
	// channel L
	GaussPyr pcL = GaussPyr(lab_channels[0], numLayers, stdC);
	GaussPyr psL = GaussPyr(lab_channels[0], numLayers, stdS);
	showPyramid(pcL, psL, numLayers);

	// channel A
	GaussPyr pcA = GaussPyr(lab_channels[1], numLayers, stdC);
	GaussPyr psA = GaussPyr(lab_channels[1], numLayers, stdS);
	showPyramid(pcA, psA, numLayers);

	// channel B
	GaussPyr pcB = GaussPyr(lab_channels[2], numLayers, stdC);
	GaussPyr psB = GaussPyr(lab_channels[2], numLayers, stdS);
	showPyramid(pcB, psB, numLayers);

	cv::waitKey(0);

	return 0;
}
