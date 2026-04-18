# Mars_Feature_Detection
This repository consists of OpenCV - C++ Application code which will detect feature detection (Ice Cap region detection, Iron Oxide region detection) on Mars. There are two approaches. One is from feature detection from Run-time image data and another one is from camera frame data based feature detection. 

1. Feature detection from Run-time image data: 
Compile Command : g++ Prog_updated.cpp -o Prog_updated `pkg-config --cflags --libs opencv4`
Executable Command: ./Prog_updated <image_jpg_file>

2. Feature detection from Live Video frame data:
Compile Command: g++ real_time_feature_detection.cpp -o real_time_feature_detection `pkg-config --cflags --libs opencv4`
Exectuable Command: ./real_time_feature_detection
