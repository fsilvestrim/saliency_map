#include "OrientedPyr.h"
#include "argparse.hpp"
#include "fusion_methods.hpp"

#include "boost/filesystem.hpp"
#include "boost/regex.hpp"

//#define VISUAL
//#define SHOW_FINAL
//#define SHOW_ORIENTATION

struct hyperparams {
    int num_layers = 4;
    int center_sigma = 2;
    int surround_sigma = 5;
    int orientation_sigma = 1;
    int orientation_amount = 4;
    int o_gardor_size = 10;
    double o_gardor_sigma = 5.0;
    double o_gardor_lambd = 5.0;
    double o_gardor_gamma = 1.0;
    double o_gardor_psi = 0.0;
    boost::filesystem::path out_path;
} settings;

void on_off_contrasts(GaussPyr center_pyr, GaussPyr surround_pyr,
                      std::vector<cv::Mat> &on_off, std::vector<cv::Mat> &off_on) {

    //DoG contrast is obtained by subtracting center from
    //surround and vice versa
    for (int l = 0; l < settings.num_layers; ++l) {
        cv::Mat center_surround = center_pyr.getLayer(l) - surround_pyr.getLayer(l);
        cv::threshold(center_surround, center_surround, 0, 1, cv::THRESH_TOZERO);
        on_off.push_back(center_surround);

        cv::Mat surround_center = surround_pyr.getLayer(l) - center_pyr.getLayer(l);
        cv::threshold(surround_center, surround_center, 0, 1, cv::THRESH_TOZERO);
        off_on.push_back(surround_center);
    }
}

void show_image(std::string header, cv::Mat img, bool scale=false)
{
    cv::namedWindow(header, cv::WINDOW_AUTOSIZE);
    if (scale) cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
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
    image_lab.convertTo(image_lab, CV_32FC3);
    image_lab /= 255.0f;

    //get channels
    cv::Mat channels[3];
    cv::split(image_lab, channels);

    std::vector<cv::Mat> conspicuity_maps;
    std::string lab_verbsose[3] = {"L (black/white)", "A (green/red)", "B (blue/yellow)"};

    // LAB channels iteration
    for (int i = 0; i < 3; ++i) {

        //center-surround - pyramids
        GaussPyr center_pyr = GaussPyr(channels[i], settings.num_layers, settings.center_sigma);
        GaussPyr surround_pyr = GaussPyr(channels[i], settings.num_layers, settings.surround_sigma);

#ifdef VISUAL
        show_two_images("Center/Surround Pyramid Layer for " + lab_verbsose[i], center_pyr.getLayer(0), surround_pyr.getLayer(0));
#endif

        //center-surround contrasts
        std::vector<cv::Mat> on_off, off_on;
        on_off_contrasts(center_pyr, surround_pyr, on_off, off_on);

#ifdef VISUAL
        show_two_images("Center/Surround Contrast 4 Upper Layer for " + lab_verbsose[i], on_off[0], off_on[0]);
#endif

        //center-surround - feature maps
        cv::Mat on_off_feature, off_on_feature;
        on_off_feature = fusion_across_scale_addition(on_off);
        off_on_feature = fusion_across_scale_addition(off_on);

#ifdef VISUAL
        show_two_images("Center/Surround Feature (fused) for " + lab_verbsose[i], on_off_feature, off_on_feature);
#endif

        //contrast - pyramids
        GaussPyr on_off_contrast = GaussPyr(on_off_feature, settings.num_layers, settings.center_sigma);
        GaussPyr off_on_contrast = GaussPyr(off_on_feature, settings.num_layers, settings.surround_sigma);

        //contrast - feature maps
        cv::Mat on_off_contrast_feature, off_on_contrast_feature;
        on_off_contrast_feature = fusion_across_scale_addition(on_off_contrast.getLayers());
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

    //orientation
    GaussPyr center_pyr = GaussPyr(channels[0], settings.num_layers, settings.center_sigma);
    LaplacianPyr lap_pyr = LaplacianPyr(center_pyr, settings.orientation_sigma);
    OrientedPyr or_pyr = OrientedPyr(lap_pyr,
                                     settings.orientation_amount,
                                     settings.o_gardor_size,
                                     settings.o_gardor_sigma,
                                     settings.o_gardor_lambd,
                                     settings.o_gardor_gamma,
                                     settings.o_gardor_psi);

    //orientations - features
    std::vector<cv::Mat> orientation_features;
    for (int j = 0; j < settings.orientation_amount; ++j) {
        orientation_features.push_back(fusion_across_scale_addition(or_pyr.orientationMaps[j]));
    }

    //conspicuity maps for orientation
    cv::Mat orientation_feature;
    orientation_feature = fusion_max(orientation_features);
    conspicuity_maps.push_back(orientation_feature);

#if defined(VISUAL) || defined(SHOW_ORIENTATION)
    show_image("Conspicuity Feature for Orientation", orientation_feature, true);
#endif

    //saliency
    cv::Mat saliency = fusion_arithmetic_mean(conspicuity_maps);

#if defined(VISUAL) || defined(SHOW_FINAL)
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
    parser.addArgument("-a", "--oamount", 1);
    parser.addArgument("-t", "--osigma", 1);
    parser.addArgument("--gardor_size", 1);
    parser.addArgument("--gardor_sigma", 1);
    parser.addArgument("--gardor_lambd", 1);
    parser.addArgument("--gardor_gamma", 1);
    parser.addArgument("--gardor_psi", 1);
    parser.parse(argc, argv);

    std::string input = parser.retrieve<std::string>("input");
    std::string output = parser.retrieve<std::string>("output");

    if(parser.exists("--layers")) settings.num_layers = std::stoi(parser.retrieve<std::string>("layers"),nullptr,0);
    if(parser.exists("--center")) settings.center_sigma = std::stoi(parser.retrieve<std::string>("center"),nullptr,0);
    if(parser.exists("--surround")) settings.surround_sigma = std::stoi(parser.retrieve<std::string>("surround"),nullptr,0);
    if(parser.exists("--oamount")) settings.orientation_amount = std::stoi(parser.retrieve<std::string>("oamount"),nullptr,0);
    if(parser.exists("--osigma")) settings.orientation_sigma = std::stoi(parser.retrieve<std::string>("osigma"),nullptr,0);
    if(parser.exists("--gardor_size")) settings.o_gardor_size = std::stoi(parser.retrieve<std::string>("gardor_size"),nullptr,0);
    if(parser.exists("--gardor_sigma")) settings.o_gardor_sigma = std::stoi(parser.retrieve<std::string>("gardor_sigma"),nullptr,0);
    if(parser.exists("--gardor_lambd")) settings.o_gardor_lambd = std::stoi(parser.retrieve<std::string>("gardor_lambd"),nullptr,0);
    if(parser.exists("--gardor_gamma")) settings.o_gardor_gamma = std::stoi(parser.retrieve<std::string>("gardor_gamma"),nullptr,0);
    if(parser.exists("--gardor_psi")) settings.o_gardor_psi = std::stoi(parser.retrieve<std::string>("gardor_psi"),nullptr,0);

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
