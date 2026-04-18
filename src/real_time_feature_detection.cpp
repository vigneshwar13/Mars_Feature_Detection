// real_time_feature_detection.cpp

#include <opencv2/opencv.hpp>
#include <iostream>

// Function for real-time feature detection
void detectFeatures() {
    cv::VideoCapture cap(0); // Open the default camera
    if(!cap.isOpened()) {
        std::cerr << "Error: Could not open camera!" << std::endl;
        return;
    }

    cv::Mat frame, gray, keypoints;
    std::vector<cv::KeyPoint> keypointsVector;
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    while(true) {
        cap >> frame; // Capture frame-by-frame
        if(frame.empty()) break;

        // Convert to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Detect keypoints
        orb->detect(gray, keypointsVector);
        cv::drawKeypoints(frame, keypointsVector, keypoints);

        // Show the frame with keypoints
        cv::imshow("Feature Detection", keypoints);

        // Exit if the user presses 'q'
        if(cv::waitKey(30) >= 0) break;
    }

    cap.release();
    cv::destroyAllWindows();
}

int main() {
    detectFeatures();
    return 0;
}