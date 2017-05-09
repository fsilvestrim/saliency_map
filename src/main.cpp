#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "OrientedPyr.h"
#include "GaussPyr.h"
#include "LaplacianPyr.h"
#include <vector>

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
	
	// gaussian
	GaussPyr gp = GaussPyr(image, 4, 5);
        
	// laplacian
	LaplacianPyr lp = LaplacianPyr(gp, 5);

	// oriented
	OrientedPyr op = OrientedPyr(lp, 8); 
		
	
	return 0;

}
