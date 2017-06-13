#include "OrientedPyr.h"
#include "GaussPyr.h"
#include "LaplacianPyr.h"
#include "argparse.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <vector>
#include <string>

int main(int argc, const char** argv)
{
	ArgumentParser parser;
  	parser.addArgument("-i", "--input");
  	parser.parse(argc, argv);

  	std::string input = parser.retrieve<std::string>("input");
	
	std::cout << "Loading image : " << input << std::endl;

	// load img
    	cv::Mat image;
    	image = cv::imread(input , CV_LOAD_IMAGE_COLOR);

    	if(!image.data)
    	{
        	std::cout << "Could not open or find the image" << std::endl;
	        return -1;
    	}
	
	// gaussian
	//GaussPyr gp = GaussPyr(image, 4, 5);
        
	// laplacian
	//LaplacianPyr lp = LaplacianPyr(gp, 5);

	// oriented
	//OrientedPyr op = OrientedPyr(lp, 8); 
	
	return 0;

}
