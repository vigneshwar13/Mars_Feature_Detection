#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // Load the image
    cv::Mat image = cv::imread("mars_image.jpg");
    
    if (image.empty()) {
        std::cout << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Detect features
    std::vector<cv::KeyPoint> keypoints;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    detector->detect(gray, keypoints);

    // Draw detected keypoints
    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);

    // Save the result
    cv::imwrite("mars_image_features.jpg", output);

    std::cout << "Feature detection completed!" << std::endl;
    return 0;
}