#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

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
#define BGR_RED_INDEX             2
#define BGR_GREEN_INDEX           1
#define BGR_BLUE_INDEX            0
#define HSV_HUE_INDEX             0
#define HSV_SAT_INDEX             1


int main(int argCnt, char * args[]){
    // ---------- Load Image ----------
    Mat img = imread(args[1]);
    if (img.empty()) {
        cout << "Error: Image not loaded!" << endl;
        return -1;
    }

    // ---------- Convert Color Spaces ----------
    Mat gray, hsv;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    cvtColor(img, hsv, COLOR_BGR2HSV);

    // ---------- Contrast Enhancement ----------
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(CLAHE_MAT_SIZE,CLAHE_MAT_SIZE));
    Mat enhancedGray;
    clahe->apply(gray, enhancedGray);

    // ---------- Local Mean ----------
    Mat localMean;
    blur(enhancedGray, localMean, Size(BLUR_MAT_SIZE,BLUR_MAT_SIZE));

    // ---------- Masks ----------
    Mat iceMask = Mat::zeros(img.size(), CV_8UC1);
    Mat ironOxideMask = Mat::zeros(img.size(), CV_8UC1);
    //Mat ironShadowMask = Mat::zeros(img.size(), CV_8UC1);

    // ---------- Pixel Classification ----------
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {

            Vec3b bgr = img.at<Vec3b>(y, x);
            Vec3b hsvPix = hsv.at<Vec3b>(y, x);

            int blueVal = bgr[BGR_BLUE_INDEX];
            int greenVal = bgr[BGR_GREEN_INDEX];
            int redVal = bgr[BGR_RED_INDEX];

            int gPix = enhancedGray.at<uchar>(y, x);
            int lm = localMean.at<uchar>(y, x);

            int hueVal = hsvPix[HSV_HUE_INDEX];
            int satVal = hsvPix[HSV_SAT_INDEX];

            // ---- ICE ----
            if ( gPix > lm + ICE_GRY_PIX_MEAN_ERROR && gPix > ICE_GRY_PIX_LOW_THR && abs(redVal - greenVal) < ICE_BGR_CLR_DIFF && abs(greenVal - blueVal) < ICE_BGR_CLR_DIFF && satVal < ICE_HSV_SAT_HIGH_THR ){
                iceMask.at<uchar>(y, x) = 255;
            }

            // ---- IRON OXIDE ----
            if ( redVal > greenVal && greenVal > blueVal && redVal > IRON_OXD_RED_THR_LOW && satVal > IRON_OXD_HSV_SAT_LOW && satVal < IRON_OXD_HSV_SAT_HGH && hueVal > IRON_OXD_HSV_HUE_LOW && hueVal < IRON_OXD_HSV_HUE_HGH ){
                ironOxideMask.at<uchar>(y, x) = 255;
            }

           /* // ---- IRON SHADOW ----   // Iron Unsaturated  - Commented 
            else if ( r > g && g > b &&
                      r > 200 &&
                      S < 100 &&
                      H > 15 && H < 40 )
            {
                ironShadowMask.at<uchar>(y, x) = 255;
            }*/
        }
    }

    // ---------- Morphological Cleanup ----------
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(MORPH_MAT_SIZE,MORPH_MAT_SIZE));
    morphologyEx(iceMask, iceMask, MORPH_OPEN, kernel);
    morphologyEx(ironOxideMask, ironOxideMask, MORPH_OPEN, kernel);
   // morphologyEx(ironShadowMask, ironShadowMask, MORPH_OPEN, kernel);

    // ==================================================
    //          BOUNDING BOX VISUALIZATION
    // ==================================================
    

   
    
     // =====================================================
    // STRATIGRAPHIC EDGE DETECTION (RELAXED)
    // =====================================================
    Mat gradX, gradY, gradMag;
    Scharr(enhancedGray, gradX, CV_32F, 1, 0);
    Scharr(enhancedGray, gradY, CV_32F, 0, 1);
    magnitude(gradX, gradY, gradMag);

    normalize(gradMag, gradMag, 0, 255, NORM_MINMAX);
    gradMag.convertTo(gradMag, CV_8U);

    Mat strataEdges;
    threshold(gradMag, strataEdges, 30, 255, THRESH_BINARY);

    // Group irregular stratigraphic edges
    morphologyEx(strataEdges, strataEdges, MORPH_CLOSE,
                 getStructuringElement(MORPH_RECT, Size(25,7)));
  
    
    Mat output = img.clone();
    
     vector<vector<Point>> contours;

    // ---- ICE PATCH BOXES (BLUE) ----
    findContours(iceMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (auto &cnt : contours) {
        if (contourArea(cnt) < CNTR_AREA_THR) continue;

        Rect box = boundingRect(cnt);
        rectangle(output, box, Scalar(255,0,0), 2);
        putText(output, "Ice",
                Point(box.x, box.y - 5),
                FONT_HERSHEY_SIMPLEX, 0.5,
                Scalar(255,0,0), 1);
    }

    // ---- IRON OXIDE BOXES (RED) ----
    contours.clear();
    findContours(ironOxideMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (auto &cnt : contours) {
        if (contourArea(cnt) < CNTR_AREA_THR) 
            continue;

        Rect box = boundingRect(cnt);
        rectangle(output, box, Scalar(0,0,255), 2);
        putText(output, "Iron Oxide",Point(box.x, box.y - 5),FONT_HERSHEY_SIMPLEX, 0.5,Scalar(0,0,255), 1);
    }
    
 // Find stratigraphic regions
    vector<vector<Point>> strataContours;
    findContours(strataEdges, strataContours,
                 RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (const auto& c : strataContours) {
        if (contourArea(c) < 800) continue;
        Rect box = boundingRect(c);
        rectangle(output, box, Scalar(0,255,0), 2);
        putText(output, "Strata", Point(box.x, box.y-5),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 1);
    }

    // ---------- Visualization ----------
    imshow("Original Image", img);
    imshow("Ice Mask", iceMask);
    imshow("Iron Oxide Mask", ironOxideMask);
    //imshow("Iron Shadow Mask", ironShadowMask);
    imshow("Final Detection Output", output);
    //imshow("Edges (Stratigraphy)", edges);

    waitKey(0);
    return 0;
}

