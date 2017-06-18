#include "LaplacianPyr.h"

LaplacianPyr::LaplacianPyr(const GaussPyr &p, float sigma) {
    for (int i = 0; i < p.getLayersSize(); ++i) {
        cv::Mat cloned = p.getLayer(i).clone();
        cv::GaussianBlur(cloned, cloned, cv::Size(), sigma, 0, cv::BORDER_REPLICATE);
        _layers.push_back(p.getLayer(i) - cloned);
    }
}
