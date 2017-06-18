#include "OrientedPyr.h"
#include "argparse.hpp"

#include "boost/filesystem.hpp"
#include "boost/regex.hpp"

struct hyperparams {
    int num_layers = 4;
    int center_sigma = 5;
    int surround_sigma = 20;
    boost::filesystem::path out_path;
} settings;

cv::Mat fusion_arithmetic_mean(const std::vector<cv::Mat> &scale_images) {
    cv::Mat add;
    double max;

    for (int i = 0; i < scale_images.size(); ++i) {
        cv::Mat img = scale_images[i].clone();
        double min_val, max_val;
        cv::minMaxLoc(img, &min_val, &max_val);
        max = std::max(max, max_val);

        if (i == 0) add = cv::Mat::zeros(img.size(), CV_32F);
        add += img;
    }

    cv::Mat result = add / 2.0f;
    cv::normalize(result, result, 0.0f, (float) max, cv::NORM_MINMAX);

#ifdef VISUAL
    cv::imshow("Fusion: Arithmetic Mean", result);
#endif

    return result;
}

cv::Mat fusion_across_scale_addition(const std::vector<cv::Mat> &scale_images) {
    cv::Size im_size = scale_images[0].size();
    cv::Mat add = scale_images[0];
    for (int i = 0; i < scale_images.size(); ++i) {
        cv::Mat resized;
        if (scale_images[i].size() != im_size) {
            cv::resize(scale_images[i], resized, im_size, 0, 0, cv::INTER_CUBIC);
            add += resized;
        }
    }

#ifdef VISUAL
    cv::imshow("Fusion: Across Scale Addition", result);
#endif

    return add;
}

void compute_pyramid(cv::Mat mat, std::vector<cv::Mat> &on_off, std::vector<cv::Mat> &off_on) {
    GaussPyr pc = GaussPyr(mat, settings.num_layers, settings.center_sigma, true);
    GaussPyr ps = GaussPyr(mat, settings.num_layers, settings.surround_sigma, true);

    //DoG contrast is obtained by subtracting center from
    //surround and vice versa
    for (int l = 0; l < settings.num_layers; ++l) {
        cv::Mat csC = pc.getLayer(l) - ps.getLayer(l);
        cv::threshold(csC, csC, 0, 1, cv::THRESH_TOZERO);
        on_off.push_back(csC);

#ifdef VISUAL
        cv::imshow("Gaussian pyramid CS Contrast", csC);
#endif

        cv::Mat scC = ps.getLayer(l) - pc.getLayer(l);
        cv::threshold(scC, scC, 0, 1, cv::THRESH_TOZERO);
        off_on.push_back(scC);

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

void process_image(boost::filesystem::path image_path) {
    std::cout << "Processing '" << image_path << "' ..." << std::endl;

    // load img
    cv::Mat image;
    image = cv::imread(image_path.string(), CV_LOAD_IMAGE_COLOR);

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

    std::vector<cv::Mat> conspicuity_maps;

    // LAB channels iteration
    for (int i = 0; i < 3; ++i) {

        //center-surround - pyramids
        std::vector<cv::Mat> on_off;
        std::vector<cv::Mat> off_on;
        compute_pyramid(lab_channels[i], on_off, off_on);

        //center-surround - feature maps
        cv::Mat on_off_feature;
        on_off_feature = fusion_across_scale_addition(on_off);

        cv::Mat off_on_feature;
        off_on_feature = fusion_across_scale_addition(off_on);

        //contrast - pyramids
        GaussPyr on_off_contrast = GaussPyr(on_off_feature, settings.num_layers, settings.center_sigma, true);
        GaussPyr off_on_contrast = GaussPyr(off_on_feature, settings.num_layers, settings.surround_sigma, true);

        //contrast - feature maps
        cv::Mat on_off_contrast_feature;
        on_off_contrast_feature = fusion_across_scale_addition(on_off_contrast.getLayers());

        cv::Mat off_on_contrast_feature;
        off_on_contrast_feature = fusion_across_scale_addition(off_on_contrast.getLayers());

        //conspicuity maps
        std::vector<cv::Mat> contrast_features = {on_off_contrast_feature, off_on_contrast_feature};
        conspicuity_maps.push_back(fusion_arithmetic_mean(contrast_features));
    }

    //saliency
    cv::Mat saliency = fusion_arithmetic_mean(conspicuity_maps);

    //save
    std::string output_path = (settings.out_path / image_path.stem()).string() + "_saliency.png";
    saveImage(saliency, output_path);
}

int main(int argc, const char **argv) {

    int num_layers = 4;
    int center_sigma = 5;
    int surround_sigma = 20;

    //params
    ArgumentParser parser;
    parser.addArgument("-i", "--input", 1, false);
    parser.addArgument("-o", "--output", 1, false);
    parser.addArgument("-l", "--layers", 1);
    parser.addArgument("-c", "--center", 1);
    parser.addArgument("-s", "--surround", 1);
    parser.parse(argc, argv);

    std::string input = parser.retrieve<std::string>("input");
    std::string output = parser.retrieve<std::string>("output");

    if(parser.exists("layers")) settings.num_layers = std::stoi(parser.retrieve<std::string>("layers"),nullptr,0);
    if(parser.exists("center")) settings.center_sigma = std::stoi(parser.retrieve<std::string>("center"),nullptr,0);
    if(parser.exists("surround")) settings.surround_sigma = std::stoi(parser.retrieve<std::string>("surround"),nullptr,0);

    //input
    boost::filesystem::path input_path(input);

    if (!boost::filesystem::exists(input_path)) {
        std::cout << "Input path '" << input << "' not found." << std::endl;
        return -1;
    }

    //output
    boost::filesystem::path output_path(output);
    if (!boost::filesystem::is_directory(output_path)) {
        std::cout << "Ouput path '" << output << "' must be a directory (and it's not)." << std::endl;
        return -1;
    }

    if (boost::filesystem::exists(output_path))
        boost::filesystem::remove_all(output_path);
    boost::filesystem::create_directory(output_path);

    settings.out_path = output_path;

    //process
    if (boost::filesystem::is_regular_file(input_path)) {
        std::cout << "You inputed just one image." << std::endl;
        process_image(input_path);
    } else if (boost::filesystem::is_directory(input_path)) {
        std::cout << "You inputed a directory. walking through its content." << std::endl;

        boost::regex pattern("\\S+.(jpg|png)$"); // list all files starting with a

        for (boost::filesystem::recursive_directory_iterator iter(input_path), end; iter != end; ++iter) {
            std::string name = iter->path().filename().string();
            if (regex_match(name, pattern)) {
                process_image(iter->path());
            } else {
                std::cout << "Skipping '" << iter->path().string() << "' because it doesn't match the file type"
                          << std::endl;
            }
        }
    }

    std::cout << "DONE! Check results at '" << settings.out_path.string() << std::endl;
    return 0;
}
