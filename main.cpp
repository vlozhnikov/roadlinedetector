#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "helper.h"

#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

const Mat process(const Mat& mat);

int main()
{
     VideoCapture cap("../roadlinedetector/resources/video.mp4");

     if (!cap.isOpened()) {
          cout << "Error opening video stream or file" << endl;
          return -1;
     }

     namedWindow("origin", WINDOW_NORMAL);
     namedWindow("result", WINDOW_NORMAL);

     auto first = true;

     while (1) {

         Mat origin;

         // Capture frame-by-frame
         cap >> origin;

         // If the frame is empty, break immediately
         if (origin.empty())
             break;

         auto result = process(origin);

         // Display the resulting frame

         imshow("origin", origin);
         if (first)
            resizeWindow("origin", 600, 350);

         imshow("result", result);
         if (first)
            resizeWindow("result", 600, 350);

         first = false;

         // Press  ESC on keyboard to exit
         char c = (char) waitKey(25);
         if (c == 27)
             break;
     }

     // When everything done, release the video capture object
     cap.release();

     // Closes all the frames
     destroyAllWindows();

    return 0;
}

// https://www.reg.ru/blog/simple-algorithm-for-road-marking-detection/
const Mat process(const Mat& origin) {

    // Преобразовать исходное изображение в  grayscale.
    Mat gray;
    cvtColor(origin, gray, COLOR_BGR2GRAY);

    // Затемнить полученное изображение (это помогает уменьшить контраст от обесцвеченных участков дороги).
    auto darkened = Helper::colorReduce(gray);

    // Преобразовать исходное изображение в  цветовое пространство HLS (Hue, Lightness, Saturation — тон, свет, насыщенность).
    Mat hsv;
    cvtColor(origin, hsv, COLOR_BGR2HLS);

    // Изолировать жёлтый цвет из HLS для получения маски (для жёлтой разметки).
    auto yellow = Helper::isolateColorMask(hsv, Scalar(20, 100, 80), Scalar(45, 200, 255));

    // Изолировать  белый цвет из HLS (для белой разметки).
    auto white = Helper::isolateColorMask(hsv, Scalar(0, 200, 0), Scalar(200, 255, 255));

    // Выполнить побитовое «ИЛИ» жёлтой и белой масок для получения общей маски.
    Mat mask;
    bitwise_or(yellow, white, mask);

    // Выполнить побитовое «И» маски и затемнённого изображения.
    Mat darkenedAnd;
    bitwise_and(darkened, mask, darkenedAnd);

    // Применить Гауссово размытие.
    Mat gauss;
    GaussianBlur(darkenedAnd, gauss, Size(3, 3), 0.0);

    // Применить детектор границ Canny (пороги устанавливаются методом проб и ошибок).
    Mat canny;
    Canny(gauss, canny, 70.0, 140.0);

    // Определить область интереса (помогает отсеять нежелательные края, обнаруженные детектором Canny).
    auto aoi = Helper::getAoi(canny);

    // Получить линии Хафа.
    std::vector<cv::Vec4i> lines;
    HoughLinesP(aoi, lines, 1, CV_PI/180, 20, 20, 300);
    for (auto l : lines) {
        auto p1 = Point(l[0], l[1]);
        auto p2 = Point(l[2], l[3]);

//        auto ps = std::vector<Point> {p1, p2};
//        print(ps);

        auto angle = atan2((l[3] - l[1]), (l[2]  - l[0]));
        auto degree = angle * (180 / CV_PI);

//        std::cout << degree << endl;

        if (abs(degree) > 15) {
            line(origin, p1, p2, Scalar(0, 0, 255), 2);
        }

    }

    // Объединить и экстраполировать линии Хафа; отобразить их на исходном изображении.
    // оставить только те линии, углы наклонов которых попадают в заданные пределы

    return aoi;
}
