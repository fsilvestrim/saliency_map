#include "LaplacianPyr.h"

LaplacianPyr::LaplacianPyr(const GaussPyr& p, float sigma)
{
	for (int i = 0; i < p.Layers.size(); ++i)
        {
		cv::Mat cloned = p.Layers[i].clone();
		cv::GaussianBlur(cloned, cloned, cv::Size(0, 0), sigma, 0, cv::BORDER_REPLICATE);
		layers.push_back(p.Layers[i] - cloned);		
	}
}
