#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "GaussPyr.h"
 
int main()
{
	std::cout << "Test" << std::endl;
	
	cv::Mat M = (cv::Mat_<float>(3,3, CV_32F) << 	1.0, 1.0, 1.0, 
													1.0, 2.0, 1.0, 
													1.0, 1.0, 1.0); 
	std::cout << "M: " << M << std::endl;

	cv::Mat G = (cv::Mat_<float>(3,3, CV_32F) << 	1, 2, 1,
													2, 4, 2,
													1, 2, 1) * (1.0/16.0);
	std::cout << "G: " << G << std::endl;

	// conv
	cv::Mat R;
	cv::GaussianBlur(M, R, cv::Size(3, 3), 1, 1, cv::BORDER_CONSTANT);
	//cv::filter2D(M, R, -1, G, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	std::cout << "R: " << R << std::endl;

	// step 1
	cv::Mat g1 = (cv::Mat_<float>(1,3, CV_32F) <<    1, 2, 1) * (1.0/4.0);
	std::cout << "g1: "<< g1 << std::endl;
		
	cv::filter2D(M, R, -1, g1, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);	
	std::cout << "R: " << R << std::endl;

	cv::Mat g2 = g1.t();
	std::cout << "g2: " << g2 << std::endl;
	
	cv::filter2D(R, R, -1, g2, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    std::cout << "R: " << R << std::endl;

	// load img
	cv::Mat image;
    image = cv::imread("../img/lena.png" , CV_LOAD_IMAGE_COLOR);

	if(!image.data)
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

	/* gaussian */ /*	
	cv::namedWindow( "Lena", cv::WINDOW_AUTOSIZE );
    
	for (int k = 3; k < 30; k+=2)
	{
	    cv::GaussianBlur(image, R, cv::Size(k, k), 0, 0, cv::BORDER_CONSTANT);
		cv::imshow( "Lena", R );
		cv::waitKey(500);
	}
	*/

	/* DoG */ /*
	cv::Mat image_gray;
	cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
	image_gray.convertTo(image_gray, CV_32F);

	cv::GaussianBlur(image_gray, g1, cv::Size(1,1), 0);
	cv::GaussianBlur(image_gray, g2, cv::Size(3,3), 0);
	cv::Mat dog = g1 - g2;
	cv::namedWindow( "Lena", cv::WINDOW_AUTOSIZE );
	cv::imshow( "Lena", dog );
	// */

	/* img pyramides */ // /*
	cv::Mat image_lab;
	cv::cvtColor(image, image_lab, CV_RGB2Lab);

	cv::Mat lab_channels[3];
	cv::split(image_lab, lab_channels); 
	
	GaussPyr p = GaussPyr(image_lab,7);
	
	cv::namedWindow( "Lena", cv::WINDOW_AUTOSIZE );
    cv::imshow( "Lena", image_lab );
	// */
	
	cv::waitKey(0);

	return 0;
}
