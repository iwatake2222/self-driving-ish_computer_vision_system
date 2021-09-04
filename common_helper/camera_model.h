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

#ifndef M_PI
#define M_PI 3.141592653f
#endif

class CameraModel {
public:
    class Parameter {
    public:
        float& pitch() { return rvec.at<float>(0); }
        float& yaw() { return rvec.at<float>(1); }
        float& roll() { return rvec.at<float>(2); }
        float& x() { return tvec.at<float>(0); }
        float& y() { return tvec.at<float>(1); }
        float& z() { return tvec.at<float>(2); }
        float& fx() { return K.at<float>(0); }
        float& cx() { return K.at<float>(2); }
        float& fy() { return K.at<float>(4); }
        float& cy() { return K.at<float>(5); }

        /* float, 3 x 1, pitch,  yaw, roll */
        cv::Mat rvec;

        /* float, 3 x 1, (X, Y, Z): horizontal, vertical, depth */
        cv::Mat tvec;

        /* float, 3 x 3 */
        cv::Mat K;

        /* float, 5 x 1 */
        cv::Mat dist_coeff;
        
        int32_t width;
        int32_t height;

        /* Default Parameters */
        Parameter() {
            SetIntrinsic(1280, 720, 500.0f);
            SetExtrinsic({ 0, 0, 0 }, { 0, 0, 0 });
            dist_coeff = (cv::Mat_<float>(5, 1) << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            //dist_coeff = (cv::Mat_<float>(5, 1) << -0.1f, 0.01f, -0.005f, -0.001f, 0.0f);
        }

        void SetIntrinsic(int32_t _width, int32_t _height, float focal_length) {
            width = _width;
            height = _height;
            float f = focal_length;
            K = (cv::Mat_<float>(3, 3) <<
                focal_length,            0,  width / 2.f,
                           0, focal_length, height / 2.f,
                           0,            0,            1);
        }

        void SetExtrinsic(const std::array<float, 3>& r_deg, const std::array<float, 3>& t) {
            rvec = (cv::Mat_<float>(3, 1) << Deg2Rad(r_deg[0]), Deg2Rad(r_deg[1]), Deg2Rad(r_deg[2]));
            tvec = (cv::Mat_<float>(3, 1) << t[0], t[1], t[2]);
        }

        void GetExtrinsic(std::array<float, 3>& r_deg, std::array<float, 3>& t) {
            r_deg = { Rad2Deg(rvec.at<float>(0)), Rad2Deg(rvec.at<float>(1)) , Rad2Deg(rvec.at<float>(2)) };
            t = { tvec.at<float>(0), tvec.at<float>(1), tvec.at<float>(2) };
        }
    };

    Parameter parameter;

    static inline float Deg2Rad(float deg) { return static_cast<float>(deg * M_PI / 180.0); }
    static inline float Rad2Deg(float rad) { return static_cast<float>(rad * 180.0 / M_PI); }

    static float FocalLength(int32_t image_size, float fov)
    {
        /* (w/2) / f = tan(fov/2) */
        return (image_size / 2) / std::tanf(Deg2Rad(fov / 2));
    }

    static cv::Mat MakeRotateMat(float x_deg, float y_deg, float z_deg)
    {
        float x_rad = Deg2Rad(x_deg);
        float y_rad = Deg2Rad(y_deg);
        float z_rad = Deg2Rad(z_deg);
        cv::Mat R_x = (cv::Mat_<float>(3, 3) <<
            1, 0, 0,
            0, std::cos(x_rad), -std::sin(x_rad),
            0, std::sin(x_rad), std::cos(x_rad));

        cv::Mat R_y = (cv::Mat_<float>(3, 3) <<
            std::cos(y_rad), 0, std::sin(y_rad),
            0, 1, 0,
            -std::sin(y_rad), 0, std::cos(y_rad));

        cv::Mat R_z = (cv::Mat_<float>(3, 3) <<
            std::cos(z_rad), -std::sin(z_rad), 0,
            std::sin(z_rad), std::cos(z_rad), 0,
            0, 0, 1);
        cv::Mat R = R_z * R_y * R_x;
        return R;
    }

    static void RotateObject(float x_deg, float y_deg, float z_deg, std::vector<cv::Point3f>& object_point_list)
    {
        cv::Mat R = MakeRotateMat(x_deg, y_deg, z_deg);
        for (auto& object_point : object_point_list) {
            cv::Mat p = (cv::Mat_<float>(3, 1) << object_point.x, object_point.y, object_point.z);
            p = R * p;
            object_point.x = p.at<float>(0);
            object_point.y = p.at<float>(1);
            object_point.z = p.at<float>(2);
        }
    }

    static void MoveObject(float x, float y, float z, std::vector<cv::Point3f>& object_point_list)
    {
        for (auto& object_point : object_point_list) {
            object_point.x += x;
            object_point.y += y;
            object_point.z += z;
        }
    }

    void ProjectWorld2Image(const cv::Point3f& object_point, cv::Point2f& image_point)
    {
        /* todo: the conversion result is diferent from cv::projectPoints when more than two angles changes */
        /*** Projection ***/
        /* s[u, v, 1] = K * [R t] * [M, 1]  */
        cv::Mat K = parameter.K;
        cv::Mat R = MakeRotateMat(Rad2Deg(parameter.pitch()), Rad2Deg(parameter.yaw()), Rad2Deg(parameter.roll()));
        cv::Mat Rt = (cv::Mat_<float>(3, 4) <<
            R.at<float>(0), R.at<float>(1), R.at<float>(2), parameter.x(),
            R.at<float>(3), R.at<float>(4), R.at<float>(5), parameter.y(),
            R.at<float>(6), R.at<float>(7), R.at<float>(8), parameter.z());

        cv::Mat M = (cv::Mat_<float>(4, 1) << object_point.x, object_point.y, object_point.z, 1);

        cv::Mat UV = K * Rt * M;

        float u = UV.at<float>(0);
        float v = UV.at<float>(1);
        float s = UV.at<float>(2);
        u /= s;
        v /= s;
        /*** Undistort ***/
        float uu = (u - parameter.cx()) / parameter.fx();  /* from optical center*/
        float vv = (v - parameter.cy()) / parameter.fy();  /* from optical center*/
        float r2 = uu * uu + vv * vv;
        float r4 = r2 * r2;
        float k1 = parameter.dist_coeff.at<float>(0);
        float k2 = parameter.dist_coeff.at<float>(1);
        float p1 = parameter.dist_coeff.at<float>(3);
        float p2 = parameter.dist_coeff.at<float>(4);
        uu = uu + uu * (k1 * r2 + k2 * r4 /*+ k3 * r6 */) + (2 * p1 * uu * vv) + p2 * (r2 + 2 * uu * uu);
        vv = vv + vv * (k1 * r2 + k2 * r4 /*+ k3 * r6 */) + (2 * p2 * uu * vv) + p1 * (r2 + 2 * vv * vv);

        image_point.x = uu * parameter.fx() + parameter.cx();
        image_point.y = vv * parameter.fy() + parameter.cy();
    }

    void PRINT_MAT_FLOAT(const cv::Mat& mat, int32_t size)
    {
        for (int32_t i = 0; i < size; i++) {
            printf("%d: %.3f\n", i, mat.at<float>(i));
        }
    }

    void ProjectImage2GroundPlane(const cv::Point2f& image_point, cv::Point3f& object_point)
    {
        /*** Undistort image point ***/
        std::vector<cv::Point2f> original_uv{ image_point };
        std::vector<cv::Point2f> image_point_undistort;
        cv::undistortPoints(original_uv, image_point_undistort, parameter.K, parameter.dist_coeff, parameter.K);
        float u = image_point_undistort[0].x;
        float v = image_point_undistort[0].y;


        /*** Calculate point in ground plane (in world coordinate) ***/
        /* Main idea:*/
        /*   s * [u, v, 1] = K * [R t] * [M, 1]  */
        /*   s * [u, v, 1] = K * R * M + K * t */
        /*   s * Kinv * [u, v, 1] = R * M + t */
        /*   s * Kinv * [u, v, 1] - t = R * M */
        /*   Rinv * (s * Kinv * [u, v, 1] - t) = M */
        /* calculate s */
        /*   s * Rinv * Kinv * [u, v, 1] = M + R_inv * t */
        /*      where, M = (X, Y, Z) and Y = camera_height(ground_plane) */
        /*      so , we can solve left[1] = M[1](camera_height) */

        cv::Mat K = parameter.K;
        cv::Mat R = MakeRotateMat(Rad2Deg(parameter.pitch()), Rad2Deg(parameter.yaw()), Rad2Deg(parameter.roll()));
        cv::Mat K_inv;
        cv::invert(K, K_inv);
        cv::Mat R_inv;
        cv::invert(R, R_inv);
        cv::Mat t = parameter.tvec;
        cv::Mat UV = (cv::Mat_<float>(3, 1) << u, v, 1);

        /* calculate s */
        cv::Mat LEFT_WO_S = R_inv * K_inv * UV;
        cv::Mat RIGHT_WO_M = R_inv * t;
        float s = (t.at<float>(1) + RIGHT_WO_M.at<float>(1)) / LEFT_WO_S.at<float>(1);

        /* calculate M */
        cv::Mat TEMP = R_inv * (s * K_inv * UV - t);

        object_point.x = TEMP.at<float>(0);
        object_point.y = TEMP.at<float>(1);
        object_point.z = TEMP.at<float>(2);

        //PRINT_MAT_FLOAT(TEMP, 3);
    }
};