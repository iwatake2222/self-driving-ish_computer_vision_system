/* Copyright 2021 iwatake2222

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef CURVE_FITTING_
#define CURVE_FITTING_

/*** Include ***/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define _USE_MATH_DEFINES
#include <cmath>
#include <string>
#include <vector>
#include <array>

#include <opencv2/opencv.hpp>

class CurveFitting
{
public:
    /* y = ax + b */
    /* point = (x, y) */
    template <typename T = int32_t>
    static bool SolveLinearRegression(const std::vector<cv::Point_<T>>& point_list, double& a, double& b)
    {
        if (point_list.size() < 2) return false;
        double x_avg = 0;
        double y_avg = 0;
        for (const auto& p : point_list) {
            x_avg += p.x;
            y_avg += p.y;
        }
        x_avg /= point_list.size();
        y_avg /= point_list.size();

        double s_xy = 0;
        double s_xx = 0;
        for (const auto& p : point_list) {
            s_xy += (p.x - x_avg) * (p.y - y_avg);
            s_xx += (p.x - x_avg) * (p.x - x_avg);
        }

        a = s_xy / s_xx;
        b = y_avg - a * x_avg;
        return true;
    }

    template <typename T = int32_t>
    static double ErrorAvgLinearRegression(const std::vector<cv::Point_<T>>& point_list, double a, double b)
    {
        double error = 0;
        for (const auto& p : point_list) {
            double y = p.y;
            double y_est = a * p.x + b;
            error += std::abs(y - y_est);
        }

        error /= point_list.size();
        return error;
    }

    template <typename T = int32_t>
    static double ErrorMaxLinearRegression(const std::vector<cv::Point_<T>>& point_list, double a, double b)
    {
        double error = 0;
        for (const auto& p : point_list) {
            double y = p.y;
            double y_est = a * p.x + b;
            error = std::max(error, std::abs(y - y_est));
        }

        return error;
    }

    /* y = ax^2 + bx + c */
    /* point = (x, y) */
    template <typename T>
    static bool SolveQuadraticRegression(const std::vector<cv::Point_<T>>& point_list, double& a, double& b, double& c)
    {
        if (point_list.size() < 3) return false;
        /*
        | sum(x^4) sum(x^3) sum(x^2) |   | a |   | sum(x*x*y) |
        | sum(x^3) sum(x^2) sum(x^1) | * | b | = | sum(x*y)   |
        | sum(x^2) sum(x^1) sum(x^0) |   | c |   | sum(y)     |
        */
        double sum_x4 = 0;
        double sum_x3 = 0;
        double sum_x2 = 0;
        double sum_x1 = 0;
        double sum_x0 = static_cast<double>(point_list.size());
        double sum_xxy = 0;
        double sum_xy = 0;
        double sum_y = 0;
        for (const auto& p : point_list) {
            sum_y += p.y;
            double xy = p.x * p.y;
            sum_xy += xy;
            sum_xxy += p.x * xy;

            double x1 = p.x;
            double x2 = x1 * x1;
            double x3 = x1 * x2;
            double x4 = x2 * x2;
            sum_x1 += x1;
            sum_x2 += x2;
            sum_x3 += x3;
            sum_x4 += x4;
        }

        cv::Mat LEFT = (cv::Mat_<double>(3, 3) << sum_x4, sum_x3, sum_x2, sum_x3, sum_x2, sum_x1, sum_x2, sum_x1, sum_x0);
        cv::Mat RIGHT = (cv::Mat_<double>(3, 1) << sum_xxy, sum_xy, sum_y);

        cv::Mat LEFT_inv;
        if (cv::invert(LEFT, LEFT_inv) == 0) return false;

        cv::Mat mat = LEFT_inv * RIGHT;
        a = mat.at<double>(0);
        b = mat.at<double>(1);
        c = mat.at<double>(2);

        return true;
    }
};


#endif
