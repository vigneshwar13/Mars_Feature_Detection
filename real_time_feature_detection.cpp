#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// ---------------- CONSTANTS ----------------
#define ICE_GRY_PIX_MEAN_ERROR    20
#define ICE_GRY_PIX_LOW_THR       200
#define ICE_BGR_CLR_DIFF          20
#define ICE_HSV_SAT_HIGH_THR      80

#define IRON_OXD_RED_THR_LOW      150
#define IRON_OXD_HSV_SAT_LOW      100
#define IRON_OXD_HSV_SAT_HGH      200
#define IRON_OXD_HSV_HUE_LOW      5
#define IRON_OXD_HSV_HUE_HGH      25

#define CNTR_AREA_THR             150
#define BLUR_MAT_SIZE             15
#define CLAHE_MAT_SIZE            8
#define MORPH_MAT_SIZE            3

int main() {

    VideoCapture cap(0); // Open camera
    if (!cap.isOpened()) {
        cout << "Error opening camera!" << endl;
        return -1;
    }

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // ---------- Preprocessing ----------
        Mat gray, hsv;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        Ptr<CLAHE> clahe = createCLAHE(2.0, Size(CLAHE_MAT_SIZE, CLAHE_MAT_SIZE));
        Mat enhancedGray;
        clahe->apply(gray, enhancedGray);

        Mat localMean;
        blur(enhancedGray, localMean, Size(BLUR_MAT_SIZE, BLUR_MAT_SIZE));

        // ---------- Masks ----------
        Mat iceMask = Mat::zeros(frame.size(), CV_8UC1);
        Mat ironMask = Mat::zeros(frame.size(), CV_8UC1);

        // ---------- Pixel Classification ----------
        for (int y = 0; y < frame.rows; y++) {
            for (int x = 0; x < frame.cols; x++) {

                Vec3b bgr = frame.at<Vec3b>(y, x);
                Vec3b hsvPix = hsv.at<Vec3b>(y, x);

                int b = bgr[0];
                int g = bgr[1];
                int r = bgr[2];

                int gPix = enhancedGray.at<uchar>(y, x);
                int lm = localMean.at<uchar>(y, x);

                int hue = hsvPix[0];
                int sat = hsvPix[1];

                // ICE Detection
                if (gPix > lm + ICE_GRY_PIX_MEAN_ERROR &&
                    gPix > ICE_GRY_PIX_LOW_THR &&
                    abs(r - g) < ICE_BGR_CLR_DIFF &&
                    abs(g - b) < ICE_BGR_CLR_DIFF &&
                    sat < ICE_HSV_SAT_HIGH_THR) {

                    iceMask.at<uchar>(y, x) = 255;
                }

                // IRON OXIDE Detection
                if (r > g && g > b &&
                    r > IRON_OXD_RED_THR_LOW &&
                    sat > IRON_OXD_HSV_SAT_LOW &&
                    sat < IRON_OXD_HSV_SAT_HGH &&
                    hue > IRON_OXD_HSV_HUE_LOW &&
                    hue < IRON_OXD_HSV_HUE_HGH) {

                    ironMask.at<uchar>(y, x) = 255;
                }
            }
        }

        // ---------- Morphology ----------
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(MORPH_MAT_SIZE, MORPH_MAT_SIZE));
        morphologyEx(iceMask, iceMask, MORPH_OPEN, kernel);
        morphologyEx(ironMask, ironMask, MORPH_OPEN, kernel);

        // ---------- Edge Detection ----------
        Mat gradX, gradY, gradMag;
        Scharr(enhancedGray, gradX, CV_32F, 1, 0);
        Scharr(enhancedGray, gradY, CV_32F, 0, 1);
        magnitude(gradX, gradY, gradMag);

        normalize(gradMag, gradMag, 0, 255, NORM_MINMAX);
        gradMag.convertTo(gradMag, CV_8U);

        Mat strataEdges;
        threshold(gradMag, strataEdges, 30, 255, THRESH_BINARY);

        morphologyEx(strataEdges, strataEdges, MORPH_CLOSE,
                     getStructuringElement(MORPH_RECT, Size(25, 7)));

        // ---------- Visualization ----------
        Mat output = frame.clone();
        vector<vector<Point>> contours;

        // ICE
        findContours(iceMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (auto &cnt : contours) {
            if (contourArea(cnt) < CNTR_AREA_THR) continue;
            Rect box = boundingRect(cnt);
            rectangle(output, box, Scalar(255, 0, 0), 2);
            putText(output, "Ice", Point(box.x, box.y - 5),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
        }

        // IRON
        contours.clear();
        findContours(ironMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (auto &cnt : contours) {
            if (contourArea(cnt) < CNTR_AREA_THR) continue;
            Rect box = boundingRect(cnt);
            rectangle(output, box, Scalar(0, 0, 255), 2);
            putText(output, "Iron Oxide", Point(box.x, box.y - 5),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
        }

        // STRATA
        contours.clear();
        findContours(strataEdges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (auto &cnt : contours) {
            if (contourArea(cnt) < 800) continue;
            Rect box = boundingRect(cnt);
            rectangle(output, box, Scalar(0, 255, 0), 2);
            putText(output, "Strata", Point(box.x, box.y - 5),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        }

        // ---------- Display ----------
        imshow("Live Detection", output);
        imshow("Ice Mask", iceMask);
        imshow("Iron Mask", ironMask);

        if (waitKey(1) == 27) break; // ESC to exit
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
