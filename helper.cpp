#include "helper.h"

const Mat Helper::lookupTable(float levels) {
    float factor = 256.0 / levels;
    Mat table(1.0, 256.0, CV_8U);
    uchar *p = table.data;

    for(int i = 0; i < 128; ++i) {
        p[i] = factor * (i / factor);
    }

    for(int i = 128; i < 256; ++i) {
        p[i] = factor * (1 + (i / factor)) - 1;
    }

    return table;
}

const Mat Helper::colorReduce(const Mat& image, float levels) {
    Mat table = lookupTable(levels);

    std::vector<Mat> c;
    cv::split(image, c);
    for (std::vector<Mat>::iterator i = c.begin(), n = c.end(); i != n; ++i) {
        Mat &channel = *i;
        LUT(channel.clone(), table, channel);
    }

    Mat reduced;
    merge(c, reduced);

    return reduced;
}

const Mat Helper::isolateColorMask(const Mat &hls, Scalar low, Scalar high) {
    Mat mask;
    inRange(hls, low, high, mask);

    return mask;
}

const Mat Helper::getAoi(const Mat& image) {
    auto cols = image.cols;
    auto rows = image.rows;
    auto mask = image.clone().setTo(Scalar::all(0));

    auto leftBottom = Point(cols * 0.05, rows);
    auto rightBottom = Point(cols * 0.95, rows);
    auto leftTop = Point(cols * 0.3, rows * 0.5);
    auto rightTop = Point(cols * 0.7, rows * 0.5);

    auto pts = std::vector<Point> {leftBottom, rightBottom, rightTop, leftTop};
    fillPoly(mask, pts, Scalar(255));

    Mat aoi;
    bitwise_and(image, mask, aoi);

    return  aoi;
}
