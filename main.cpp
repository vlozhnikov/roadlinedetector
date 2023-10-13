#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "helper.h"

#include <iostream>
#include <numeric>
#include <algorithm>

using namespace cv;
using namespace std;

const Mat process(const Mat& mat);
const std::map<int, std::vector<cv::Vec4i>> groupingLines(const std::vector<cv::Vec4i>& lines, const unsigned delta = 20);
const std::pair<cv::Point, cv::Point> pointsFromLine(const cv::Vec4i& line, const float kOffset = 0.2);
int distance(const Point& x1y1, const Point& x2y2);
void algorithm1(const Mat& mat, std::vector<cv::Vec4i>& lines);

int main()
{
     VideoCapture cap("/Users/user/MyProjects/opencv/roadlinedetector/resources/video.mp4");

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
    
    // обработка алгоритмом 1
    algorithm1(origin, lines);

    return aoi;
}

const std::map<int, std::vector<cv::Vec4i>> groupingLines(const std::vector<cv::Vec4i>& lines, const unsigned delta) {

    std::map<int, std::vector<cv::Vec4i>> groupedBy;

    // сгруппировать линии по дельте (расстоянию) между ними
    using groups = std::map<int, std::vector<cv::Vec4i>>;
    groupedBy = std::reduce(lines.begin(), lines.end(), groups{}, [delta, lines](groups acc, const cv::Vec4i& item) {

        // получить все ключи
        auto keys = std::vector<int>();
        for (const auto& [key, _] : acc) {
            keys.push_back(key);
        }
        
        auto maxKey = 0;
        if (!keys.empty()) {
            maxKey = *std::max_element(keys.begin(), keys.end());
        }
        
        // смотрим, есть ли текущая линия в уже найденной группе? если есть, то шаг пропускаем
        for (const auto& a : acc) {
            if (std::find(a.second.begin(), a.second.end(), item) != a.second.end()) {
                return acc;
            }
        }
        
        // найти линии, которые находятся в одной группе с текущей
        auto foundLines = std::vector<cv::Vec4i>();
        std::copy_if(lines.begin(), lines.end(), std::back_inserter(foundLines), [delta, maxKey, item](const auto& line) {
            
            if (line == item) return false; // не анализировать самого себя

            // получить две точки
            auto itemPoints = pointsFromLine(item);
            auto linePoints = pointsFromLine(line);

            // вычислить расстояния
            auto firstDelta = false;
            auto secondDelta = false;

            auto d1 = distance(itemPoints.first, linePoints.first);
            auto d2 = distance(itemPoints.first, linePoints.second);
            auto d3 = distance(itemPoints.second, linePoints.first);
            auto d4 = distance(itemPoints.second, linePoints.second);
            
            if (d1 <= delta) firstDelta = true;
            if (d2 <= delta) {
                if (firstDelta) secondDelta = true;
                else firstDelta = true;
            }
            if (d3 <= delta) {
                if (firstDelta) secondDelta = true;
                else firstDelta = true;
            }
            if (d4 <= delta) {
                if (firstDelta) secondDelta = true;
                else firstDelta = true;
            }

            return firstDelta && secondDelta;
        });
        
        // добавить найденные линии в текущую группу
        for (auto const& f : foundLines) {
            acc[maxKey + 1].push_back(f);
        }

        return acc;
    });
    
    return  groupedBy;
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

void algorithm1(const Mat& mat, std::vector<cv::Vec4i>& lines) {
    
    // отфильтровать линии по углу наклона.
    std::vector<cv::Vec4i> filteredLines;
    std::copy_if(lines.begin(), lines.end(), std::back_inserter(filteredLines), [](const cv::Vec4i& l) {

        auto angle = atan2((l[3] - l[1]), (l[2]  - l[0]));
        auto degree = angle * (180 / CV_PI);

        return (abs(degree) > 15);
    });
    
    // сгруппировать найденные линии
    auto groupedBy = groupingLines(filteredLines, 50);
    
    // еще раз прогнать алгоритм для объединения похожих групп
    filteredLines.clear();
    for (const auto& g : groupedBy) {
        for (const auto& l : g.second) {
            filteredLines.push_back(l);
        }
    }
    
    groupedBy.clear();
    groupedBy = groupingLines(filteredLines, 50);
    
    // из каждой объединенной группы получить среднюю линию
    std::map<int, std::vector<cv::Vec4i>> averageGroups;
    for (const auto& e : groupedBy) {
        
        auto v = e.second;
        
        auto averageX1 = 0;
        auto averageY1 = 0;
        auto averageX2 = 0;
        auto averageY2 = 0;
        
        for (const auto& g : v) {
            averageX1 += g[0];
            averageY1 += g[1];
            averageX2 += g[2];
            averageY2 += g[3];
        }
        
        averageX1 /= v.size();
        averageY1 /= v.size();
        averageX2 /= v.size();
        averageY2 /= v.size();
        
        averageGroups[e.first] = {cv::Vec4i{averageX1, averageY1, averageX2, averageY2}};
    }

//    for (const auto& e : averageGroups) {
//        std::cout << "key: " << e.first << endl;
//        auto a = e.second;
//        for (auto v : a) {
//            std::cout << v << ", ";
//        }
//        std::cout << endl;
//    }
//
//    std::cout << "--------" << endl << endl;

    auto c = 0;
    for (const auto& e : averageGroups) {

        auto v = e.second;
        for (auto l : v) {

            auto p1 = Point(l[0], l[1]);
            auto p2 = Point(l[2], l[3]);

            line(mat, p1, p2, Scalar(0, 255, 0), 15);
        }
    }
}
