#include "OrientedPyr.h"
#include "argparse.hpp"

#include "boost/filesystem.hpp"
#include "boost/regex.hpp"

struct hyperparams {
    int numLayers = 4;
    int centerSigma = 5;
    int surroundSigma = 20;
} settings;

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

#ifdef VISUAL
    cv::imshow("Across Scale Addition", result);
#endif

    return result;
}

void compute_pyramid(cv::Mat mat, std::vector<cv::Mat>& mCsC, std::vector<cv::Mat>& mSCC) {
    GaussPyr pc = GaussPyr(mat, settings.numLayers, settings.centerSigma);
    GaussPyr ps = GaussPyr(mat, settings.numLayers, settings.surroundSigma);

    for (int l = 0; l < settings.numLayers; ++l) {
        cv::Mat csC = pc.getLayer(l) - ps.getLayer(l);
        cv::threshold(csC, csC, 0, 1, cv::THRESH_TOZERO);
        mCsC.push_back(csC);

#ifdef VISUAL
        cv::imshow("Gaussian pyramid CS Contrast", csC);
#endif

        cv::Mat scC = ps.getLayer(l) - pc.getLayer(l);
        cv::threshold(scC, scC, 0, 1, cv::THRESH_TOZERO);
        mSCC.push_back(scC);

#ifdef VISUAL
        cv::imshow("Gaussian pyramid SC Contrast", scC);
        cv::waitKey(0);
#endif
    }
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

    //center-surround pyramids
    std::vector<cv::Mat> mCsC[3];
    std::vector<cv::Mat> mSCC[3];
    compute_pyramid(lab_channels[0], mCsC[0], mSCC[0]);
    compute_pyramid(lab_channels[1], mCsC[1], mSCC[1]);
    compute_pyramid(lab_channels[2], mCsC[2], mSCC[2]);

    //contrast pyramids
    cv::Mat across_scale_additions[3];
    cv::Mat featMapCenterL = across_scale_addition(mCsC[0]);
    cv::Mat featMapSurroundL = across_scale_addition(mSCC[0]);

    //feature maps

    //conspicuity maps

    //saliency
}

int main(int argc, const char **argv) {
    ArgumentParser parser;
    parser.addArgument("-i", "--input", 1);
    parser.addArgument("-o", "--output", 1);
    parser.parse(argc, argv);

    std::string input = parser.retrieve<std::string>("input");
    std::string output = parser.retrieve<std::string>("output");

    boost::filesystem::path input_path(input);

    if ( !boost::filesystem::exists( input_path ) )
    {
        std::cout << "Input path '" << input << "' not found." << std::endl;
        return -1;
    }

    boost::filesystem::path output_path(output);
    if(!boost::filesystem::is_directory(output_path))
    {
        std::cout << "Ouput path '" << output << "' must be a directory (and it's not)." << std::endl;
        return -1;
    }

    if (boost::filesystem::exists(output_path))
        boost::filesystem::remove_all(output_path);
    boost::filesystem::create_directory(output_path);

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
    int centerSigma = 5;
    int surroundSigma = 20;

    // channel L
    GaussPyr pcL = GaussPyr(lab_channels[0], numLayers, centerSigma);
    GaussPyr psL = GaussPyr(lab_channels[0], numLayers, surroundSigma);
    showPyramid(pcL, psL, numLayers);

    // channel A
    GaussPyr pcA = GaussPyr(lab_channels[1], numLayers, centerSigma);
    GaussPyr psA = GaussPyr(lab_channels[1], numLayers, surroundSigma);
    showPyramid(pcA, psA, numLayers);

    // channel B
    GaussPyr pcB = GaussPyr(lab_channels[2], numLayers, centerSigma);
    GaussPyr psB = GaussPyr(lab_channels[2], numLayers, surroundSigma);
    showPyramid(pcB, psB, numLayers);

    // laplacian
    //LaplacianPyr lp = LaplacianPyr(gp, 5);
    //saveImage(gp.getLayer(0), "test.png");


    // oriented
    //OrientedPyr op = OrientedPyr(lp, 8);
    */

    return 0;

}
