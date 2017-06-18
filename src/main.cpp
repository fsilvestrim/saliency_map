#include "OrientedPyr.h"
#include "argparse.hpp"
#include "fusion_methods.hpp"

#include "boost/filesystem.hpp"
#include "boost/regex.hpp"

#define VISUAL

struct hyperparams {
    int num_layers = 4;
    int center_sigma = 5;
    int surround_sigma = 20;
    boost::filesystem::path out_path;
} settings;

void compute_pyramid(cv::Mat mat, std::vector<cv::Mat> &on_off, std::vector<cv::Mat> &off_on) {
    GaussPyr pc = GaussPyr(mat, settings.num_layers, settings.center_sigma, true);
    GaussPyr ps = GaussPyr(mat, settings.num_layers, settings.surround_sigma, true);

    //DoG contrast is obtained by subtracting center from
    //surround and vice versa
    for (int l = 0; l < settings.num_layers; ++l) {
        cv::Mat csC = pc.getLayer(l) - ps.getLayer(l);
        cv::threshold(csC, csC, 0, 1, cv::THRESH_TOZERO);
        on_off.push_back(csC);

        cv::Mat scC = ps.getLayer(l) - pc.getLayer(l);
        cv::threshold(scC, scC, 0, 1, cv::THRESH_TOZERO);
        off_on.push_back(scC);
    }
}

void show_image(std::string header, cv::Mat img)
{
    cv::namedWindow(header, cv::WINDOW_AUTOSIZE);
    cv::imshow(header, img);
    cv::waitKey(0);
    cv::destroyWindow(header);
}

void show_two_images(std::string header, cv::Mat left, cv::Mat right)
{
    cv::Mat merged(left.size().height, left.size().width+right.size().width, CV_32F);
    cv::Mat in_left(merged, cv::Rect(0, 0, left.size().width, left.size().height));
    left.copyTo(in_left);
    cv::Mat in_right(merged, cv::Rect(left.size().width, 0, right.size().width, right.size().height));
    right.copyTo(in_right);
    show_image(header, merged);
}

void save_image(cv::Mat image, std::string outPath) {
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
    std::string lab_verbsose[3] = {"L (black/white)", "A (green/red)", "B (blue/yellow)"};

    // LAB channels iteration
    for (int i = 0; i < 3; ++i) {

        //center-surround - pyramids
        std::vector<cv::Mat> on_off;
        std::vector<cv::Mat> off_on;
        compute_pyramid(lab_channels[i], on_off, off_on);

#ifdef VISUAL
        show_two_images("Center/Surround Upper Layer for " + lab_verbsose[i], on_off[0], off_on[0]);
#endif

        //center-surround - feature maps
        cv::Mat on_off_feature;
        on_off_feature = fusion_across_scale_addition(on_off);

        cv::Mat off_on_feature;
        off_on_feature = fusion_across_scale_addition(off_on);

#ifdef VISUAL
        show_two_images("Center/Surround Feature (fused) for " + lab_verbsose[i], on_off_feature, off_on_feature);
#endif

        //contrast - pyramids
        GaussPyr on_off_contrast = GaussPyr(on_off_feature, settings.num_layers, settings.center_sigma, true);
        GaussPyr off_on_contrast = GaussPyr(off_on_feature, settings.num_layers, settings.surround_sigma, true);

        //contrast - feature maps
        cv::Mat on_off_contrast_feature;
        on_off_contrast_feature = fusion_across_scale_addition(on_off_contrast.getLayers());

        cv::Mat off_on_contrast_feature;
        off_on_contrast_feature = fusion_across_scale_addition(off_on_contrast.getLayers());

#ifdef VISUAL
        show_two_images("Center/Surround Contrast Feature (fused) for " + lab_verbsose[i], on_off_contrast_feature, off_on_contrast_feature);
#endif

        //conspicuity maps
        std::vector<cv::Mat> contrast_features = {on_off_contrast_feature, off_on_contrast_feature};
        conspicuity_maps.push_back(fusion_arithmetic_mean(contrast_features));

#ifdef VISUAL
        show_image("Conspicuity Feature for " + lab_verbsose[i], conspicuity_maps[i]);
#endif
    }

    //saliency
    cv::Mat saliency = fusion_arithmetic_mean(conspicuity_maps);

#ifdef VISUAL
    show_image("Final", saliency);
#endif

    //save
    std::string output_path = (settings.out_path / image_path.stem()).string() + "_saliency.png";
    save_image(saliency, output_path);
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
