#include "OrientedPyr.h"
#include "argparse.hpp"

#include "boost/filesystem.hpp"
#include "boost/regex.hpp"

cv::Mat across_scale_addition(const std::vector<cv::Mat> &scale_images) {
    cv::Size im_size = scale_images[0].size();
    cv::Mat result = scale_images[0];
    for (int i = 1; i < scale_images.size(); ++i) {
        cv::Mat resized;
        if (scale_images[i].size() != im_size) {
            cv::resize(scale_images[i], resized, im_size, 0, 0, cv::INTER_CUBIC);
        } else {
            resized = scale_images[0];
        }

        result += resized;
    }
    return result;
}

void showPyramid(GaussPyr pc, GaussPyr ps, int numLayers) {
    std::vector<cv::Mat> mCsC;
    std::vector<cv::Mat> mSCC;

    for (int l = 0; l < numLayers; ++l) {
        cv::Mat csC = pc.getLayer(l) - ps.getLayer(l);
        cv::threshold(csC, csC, 0, 1, cv::THRESH_TOZERO);
        mCsC.push_back(csC);

        cv::imshow("Gaussian pyramid CS Contrast", csC);

        cv::Mat scC = ps.getLayer(l) - pc.getLayer(l);
        cv::threshold(scC, scC, 0, 1, cv::THRESH_TOZERO);
        mSCC.push_back(scC);

        cv::imshow("Gaussian pyramid SC Contrast", scC);

        cv::waitKey(0);
    }

    cv::imshow("Feature Map Center", across_scale_addition(mCsC));
    cv::imshow("Feature Map Surround", across_scale_addition(mSCC));
    cv::waitKey(0);
}

void saveImage(cv::Mat image, std::string outPath) {
    cv::Mat outImage;
    image.convertTo(outImage, CV_8UC1, 255);
    cv::imwrite(outPath, outImage);
}

void process_image(std::string image_path)
{
    std::cout << "Processing '" << image_path << "' ..." << std::endl;

    // load img
    cv::Mat image;
    image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);

    if (!image.data) {
        std::cout << "Skipping: Could not open or find the image" << std::endl;
        return;
    }

    // convert to lab
    cv::Mat image_lab;
    cv::cvtColor(image, image_lab, CV_RGB2Lab);
    image_lab.convertTo(image_lab, CV_32F);
    image_lab /= 255.0f;

    //get channels
    cv::Mat lab_channels[3];
    cv::split(image_lab, lab_channels);
}

int main(int argc, const char **argv) {
    ArgumentParser parser;
    parser.addArgument("-i", "--input", 1);
    parser.parse(argc, argv);

    std::string input = parser.retrieve<std::string>("input");

    boost::filesystem::path input_path(input);

    if ( !boost::filesystem::exists( input_path ) )
    {
        std::cout << "Path '" << input << "' not found." << std::endl;
        return -1;
    }

    if(boost::filesystem::is_regular_file(input_path))
    {
        std::cout << "You inputed just one image." << std::endl;
        process_image(input_path.string());
    }
    else if (boost::filesystem::is_directory(input_path))
    {
        std::cout << "You inputed a directory. walking through its content." << std::endl;

        boost::regex pattern("\\S+.(jpg|png)$"); // list all files starting with a

        for (boost::filesystem::recursive_directory_iterator iter(input_path), end; iter != end; ++iter)
        {
            std::string name = iter->path().filename().string();
            if (regex_match(name, pattern))
            {
                process_image(iter->path().string());
            }
            else
            {
                std::cout << "Skipping '" << iter->path().string() << "' because it doesn't match the file type" << std::endl;
            }
        }
    }

    /*
    // hyper parameters
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

    // laplacian
    //LaplacianPyr lp = LaplacianPyr(gp, 5);
    //saveImage(gp.getLayer(0), "test.png");


    // oriented
    //OrientedPyr op = OrientedPyr(lp, 8);
    */

    return 0;

}
