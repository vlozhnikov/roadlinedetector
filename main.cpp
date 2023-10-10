#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "helper.h"

#include <iostream>
#include <numeric>
#include <algorithm>

using namespace cv;
using namespace std;

const Mat process(const Mat& mat);
const std::map<int, std::vector<cv::Vec4i>> groupingLines(const std::vector<cv::Vec4i>& lines, const unsigned int count = 1);
const std::pair<cv::Point, cv::Point> pointsFromLine(const cv::Vec4i& line, const float kOffset = 0.2);
int distance(const Point& x1y1, const Point& x2y2);

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

    // отфильтровать линии по углу наклона.
    std::vector<cv::Vec4i> filteredLines;
    std::copy_if(lines.begin(), lines.end(), std::back_inserter(filteredLines), [](const cv::Vec4i& l) {

        auto angle = atan2((l[3] - l[1]), (l[2]  - l[0]));
        auto degree = angle * (180 / CV_PI);

        return (abs(degree) > 15);
    });

    auto groupedBy = groupingLines(filteredLines, 10);

//    for (const auto& e : groupedBy) {
//        auto a = e.second;
//        for (auto v : a) {
//            std::cout << v << ", ";
//        }
//        std::cout << endl;
//    }

    // из каждой объединенной группы получить среднюю линию

    for (const auto& e : groupedBy) {

        auto v = e.second;

        for (auto l : v) {

            if (v.size() < 2) continue;

            auto p1 = Point(l[0], l[1]);
            auto p2 = Point(l[2], l[3]);

            line(origin, p1, p2, Scalar(0, 255, 0), 5);
            break;
        }
    }

    return aoi;
}

const std::map<int, std::vector<cv::Vec4i>> groupingLines(const std::vector<cv::Vec4i>& lines, const unsigned int count) {

    auto localCount = count;
    if (localCount < 1) return {};

    auto first = true;
    std::vector<cv::Vec4i> localLines;

    while (localCount > 0) {

        std::map<int, std::vector<cv::Vec4i>> groupedBy;

        if (first) {
            for (const auto& e : lines) {
                localLines.push_back(e);
            }
        }
        first = false;

        // сгруппировать линии по дельте (расстоянию) между ними
        using groups = std::map<int, std::vector<cv::Vec4i>>;
        groupedBy = std::reduce(localLines.begin(), localLines.end(), groups{}, [](groups acc, const cv::Vec4i& item) {

    //        std::cout << endl << item << endl;

            // find all deltas with grouped lines before
            if (acc.empty()) {
                acc[1] = std::vector<cv::Vec4i>{item}; // first index is 1
                return acc;
            }

            // получить все ключи
            auto keys = std::vector<int>();
            for (const auto& [key, _] : acc) {
                keys.push_back(key);
            }

            // получить максимальный ключ
            auto maxKey = *std::max_element(keys.begin(), keys.end());

            // получить лве точки линии
            auto itemPoints = pointsFromLine(item);

            for (const auto& key : keys) {
                auto deltas = std::vector<int>();

                // вычислить расстояния между item и line в двух точках
                auto foundLines = std::vector<cv::Vec4i>();

                std::copy_if(acc[key].begin(), acc[key].end(), std::back_inserter(foundLines), [itemPoints](const auto& line) {

                    // получить две точки
                    auto midPoints = pointsFromLine(line);

                    // вычислить расстояния
                    const auto delta = 5;

                    auto firstDelta = false;
                    auto secondDelta = false;

                    auto d1 = distance(itemPoints.first, midPoints.first);
                    if (d1 <= delta) firstDelta = true;
                    auto d2 = distance(itemPoints.first, midPoints.second);
                    if (d2 <= delta) {
                        if (firstDelta) secondDelta = true;
                        else firstDelta = true;
                    }
                    auto d3 = distance(itemPoints.second, midPoints.first);
                    if (d3 <= delta) {
                        if (firstDelta) secondDelta = true;
                        else firstDelta = true;
                    }
                    auto d4 = distance(itemPoints.second, midPoints.second);
                    if (d4 <= delta) {
                        if (firstDelta) secondDelta = true;
                        else firstDelta = true;
                    }

                    return firstDelta && secondDelta;
                });

    //            std::for_each(foundLines.begin(), foundLines.end(), [](const auto &e) {
    //                std::cout << "found line: " << e << endl;
    //            });

                // если результат пусто, то создаем новую группу с новым ключем
                if (foundLines.empty()) {
                    acc[maxKey+1] = std::vector<cv::Vec4i>{item};
                }
                // иначе, добавляем текущую линию к найденной группе
                else {
                    acc[key].push_back(item);
                }
            }

            return acc;
        });

        std::cout << "localLines size: " << localLines.size() << endl;

        if (localLines.empty()) {
            return groupedBy;
        }

        localLines.erase(localLines.begin(), localLines.end());
        for (const auto& e : groupedBy) {
            for (const auto& v : e.second) {
                localLines.push_back(v);
            }
        }

        localCount -= 1;
        if (localCount == 0) {
            return  groupedBy;
        }
    };

    return {};
}

// получить две точки на отрезке с заданным смещением kOffset, относительно краев
const std::pair<cv::Point, cv::Point> pointsFromLine(const cv::Vec4i& line, const float kOffset) {
    auto xa = line[0];
    auto ya = line[1];
    auto xb = line[2];
    auto yb = line[3];

    auto x1 = xa + (xb - xa) * kOffset;
    auto y1 = ya + (yb - ya) * kOffset;

    auto x2 = xa + (xb - xa) * (1 - kOffset);
    auto y2 = ya + (yb - ya) * (1 - kOffset);

    return std::pair<cv::Point, cv::Point>{Point(x1, y1), Point(x2, y2)};
}

// вычислить расстояние между двумя точками
int distance(const Point& x1y1, const Point& x2y2) {
    return abs(sqrt(pow(x1y1.x - x2y2.x, 2) + pow(x1y1.y - x2y2.y, 2)));
}
