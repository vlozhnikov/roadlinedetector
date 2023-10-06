#ifndef HELPER_H
#define HELPER_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Helper
{
public:
    static const Mat lookupTable(float levels);
    static const Mat colorReduce(const Mat& image, float levels = 255.0);
    static const Mat isolateColorMask(const Mat &hls, Scalar low, Scalar high);
    static const Mat getAoi(const Mat& image);
};

#endif // HELPER_H
