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
#ifndef CAMERA_MODEL_
#define CAMERA_MODEL_

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
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.141592653f
#endif

static inline float Deg2Rad(float deg) { return static_cast<float>(deg * M_PI / 180.0); }
static inline float Rad2Deg(float rad) { return static_cast<float>(rad * 180.0 / M_PI); }
static inline float FocalLength(int32_t image_size, float fov)
{
    /* (w/2) / f = tan(fov/2) */
    return (image_size / 2) / std::tan(Deg2Rad(fov / 2));
}

class CameraModel {
    /***
    * s[x, y, 1] = K * [R t] * [Mw, 1]
    *     K: カメラの内部パラメータ
    *     [R t]: カメラの外部パラメータ
    *     R: ワールド座標上でのカメラの回転行列 (カメラの姿勢)
    *     t: カメラ座標上での、カメラ位置(Oc)からワールド座標原点(Ow)へのベクトル= Ow - Oc
    *         = -RT (T = ワールド座標上でのOwからOcへのベクトル)
    *     Mw: ワールド座標上での対象物体の座標 (Xw, Yw, Zw)
    * s[x, y, 1] = K * Mc
    *     Mc: カメラ座標上での対象物体の座標 (Xc, Yc, Zc)
    *         = [R t] * [Mw, 1]
    *
    * 理由: Mc = R(Mw - T) = RMw - RT = RMw + t   (t = -RT)
    *
    * 注意1: tはカメラ座標上でのベクトルである。そのため、Rを変更した場合はtを再計算する必要がある
    *
    * 注意2: 座標系は右手系。X+ = 右、Y+ = 下、Z+ = 奥  (例. カメラから見て物体が上にある場合、Ycは負値)
    ***/

public:
    /*** Intrinsic parameters ***/
    /* float, 3 x 3 */
    cv::Mat K;
    cv::Mat K_new;

    int32_t width;
    int32_t height;

    /* float, 5 x 1 */
    cv::Mat dist_coeff;

    /*** Extrinsic parameters ***/
    /* float, 3 x 1, pitch(rx),  yaw(ry), roll(rz) [rad] */
    cv::Mat rvec;

    /* float, 3 x 1, (tx, ty, tz): horizontal, vertical, depth (Camera location: Ow - Oc in camera coordinate) */
    cv::Mat tvec;

public:
    CameraModel() {
        /* Default Parameters */
        SetIntrinsic(1280, 720, 500.0f);
        SetDist({ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
        //SetDist({ -0.1f, 0.01f, -0.005f, -0.001f, 0.0f });
        SetExtrinsic({ 0, 0, 0 }, { 0, 0, 0 });
    }

    /*** Accessor for camera parameters ***/
    float& rx() { return rvec.at<float>(0); }   /* pitch */
    float& ry() { return rvec.at<float>(1); }   /* yaw */
    float& rz() { return rvec.at<float>(2); }   /* roll */
    float& tx() { return tvec.at<float>(0); }
    float& ty() { return tvec.at<float>(1); }
    float& tz() { return tvec.at<float>(2); }
    float& fx() { return K.at<float>(0); }
    float& cx() { return K.at<float>(2); }
    float& fy() { return K.at<float>(4); }
    float& cy() { return K.at<float>(5); }


    /*** Methods for camera parameters ***/
    void SetIntrinsic(int32_t width, int32_t height, float focal_length) {
        this->width = width;
        this->height = height;
        this->K = (cv::Mat_<float>(3, 3) <<
            focal_length, 0, width / 2.f,
            0, focal_length, height / 2.f,
            0, 0, 1);
        UpdateNewCameraMatrix();
    }

    void SetDist(const std::array<float, 5>& dist) {
        this->dist_coeff = (cv::Mat_<float>(5, 1) << dist[0], dist[1], dist[2], dist[3], dist[4]);
        UpdateNewCameraMatrix();
    }

    void UpdateNewCameraMatrix()
    {
        if (!this->K.empty() && !this->dist_coeff.empty()) {
            this->K_new = cv::getOptimalNewCameraMatrix(this->K, this->dist_coeff, cv::Size(this->width, this->height), 0.0);
        }
    }

    void SetExtrinsic(const std::array<float, 3>& rvec_deg, const std::array<float, 3>& tvec, bool is_t_on_world = true)
    {
        this->rvec = (cv::Mat_<float>(3, 1) << Deg2Rad(rvec_deg[0]), Deg2Rad(rvec_deg[1]), Deg2Rad(rvec_deg[2]));
        this->tvec = (cv::Mat_<float>(3, 1) << tvec[0], tvec[1], tvec[2]);

        /*
            is_t_on_world == true: tvec = T (Oc - Ow in world coordinate)
            is_t_on_world == false: tvec = tvec (Ow - Oc in camera coordinate)
        */
        if (is_t_on_world) {
            cv::Mat R = MakeRotationMat(Rad2Deg(rx()), Rad2Deg(ry()), Rad2Deg(rz()));
            this->tvec = -R * this->tvec;   /* t = -RT */
        }
    }

    void GetExtrinsic(std::array<float, 3>& rvec_deg, std::array<float, 3>& tvec, bool is_t_on_world = true)
    {
        rvec_deg = { Rad2Deg(this->rvec.at<float>(0)), Rad2Deg(this->rvec.at<float>(1)) , Rad2Deg(this->rvec.at<float>(2)) };
        tvec = { this->tvec.at<float>(0), this->tvec.at<float>(1), this->tvec.at<float>(2) };
        /*
            is_t_on_world == true: tvec = T (Oc - Ow in world coordinate)
            is_t_on_world == false: tvec = tvec (Ow - Oc in camera coordinate)
        */
        if (is_t_on_world) {
            /* t = -RT -> T = -R^1 t */
            cv::Mat R = MakeRotationMat(Rad2Deg(rx()), Rad2Deg(ry()), Rad2Deg(rz()));
            cv::Mat R_inv;
            cv::invert(R, R_inv);
            cv::Mat T = -R_inv * this->tvec;
            tvec = { T.at<float>(0), T.at<float>(1), T.at<float>(2) };
        } else {
            tvec = { this->tvec.at<float>(0), this->tvec.at<float>(1), this->tvec.at<float>(2) };
        }
        
    }

    void SetCameraPos(float tx, float ty, float tz, bool is_on_world = true)    /* Oc - Ow */
    {
        this->tvec = (cv::Mat_<float>(3, 1) << tx, ty, tz);
        if (is_on_world) {
            cv::Mat R = MakeRotationMat(Rad2Deg(rx()), Rad2Deg(ry()), Rad2Deg(rz()));
            this->tvec = -R * this->tvec;   /* t = -RT */
        } else {
            /* Oc - Ow -> Ow - Oc */
            this->tvec *= -1;
        }
    }

    void MoveCameraPos(float dtx, float dty, float dtz, bool is_on_world = true)    /* Oc - Ow */
    {
        cv::Mat tvec_delta = (cv::Mat_<float>(3, 1) << dtx, dty, dtz);
        if (is_on_world) {
            cv::Mat R = MakeRotationMat(Rad2Deg(rx()), Rad2Deg(ry()), Rad2Deg(rz()));
            tvec_delta = -R * tvec_delta;
        } else {
            /* Oc - Ow -> Ow - Oc */
            tvec_delta *= -1;
        }
        this->tvec += tvec_delta;
    }

    void SetCameraAngle(float pitch_deg, float yaw_deg, float roll_deg)
    {
        /* t vec is vector in camera coordinate, so need to re-calculate it when rvec is updated */
        cv::Mat R_old = MakeRotationMat(Rad2Deg(rx()), Rad2Deg(ry()), Rad2Deg(rz()));
        cv::Mat T = -R_old.inv() * this->tvec;      /* T is tvec in world coordinate.  t = -RT */
        this->rvec = (cv::Mat_<float>(3, 1) << Deg2Rad(pitch_deg), Deg2Rad(yaw_deg), Deg2Rad(roll_deg));
        cv::Mat R_new = MakeRotationMat(Rad2Deg(rx()), Rad2Deg(ry()), Rad2Deg(rz()));
        this->tvec = -R_new * T;   /* t = -RT */
    }

    void RotateCameraAngle(float dpitch_deg, float dyaw_deg, float droll_deg)
    {
        /* t vec is vector in camera coordinate, so need to re-calculate it when rvec is updated */
        cv::Mat R_old = MakeRotationMat(Rad2Deg(rx()), Rad2Deg(ry()), Rad2Deg(rz()));
        cv::Mat T = -R_old.inv() * this->tvec;      /* T is tvec in world coordinate.  t = -RT */
        cv::Mat R_delta = MakeRotationMat(dpitch_deg, dyaw_deg, droll_deg);
        cv::Mat R_new = R_delta * R_old;
        this->tvec = -R_new * T;   /* t = -RT */
        cv::Rodrigues(R_new, this->rvec); /* Rotation matrix -> rvec */
    }

    /*** Methods for projection ***/
    void ConvertWorld2Image(const cv::Point3f& object_point, cv::Point2f& image_point)
    {
        std::vector<cv::Point3f> object_point_list = { object_point };
        std::vector<cv::Point2f> image_point_list;
        ConvertWorld2Image(object_point_list, image_point_list);
        image_point = image_point_list[0];
    }

    void ConvertWorld2Image(const std::vector<cv::Point3f>& object_point_list, std::vector<cv::Point2f>& image_point_list)
    {
        /*** Mw -> Image ***/
        /* the followings get exactly the same result */
#if 1
        /*** Projection ***/
        /* s[x, y, 1] = K * [R t] * [M, 1] = K * M_from_cam */
        cv::Mat K = this->K;
        cv::Mat R = MakeRotationMat(Rad2Deg(this->rx()), Rad2Deg(this->ry()), Rad2Deg(this->rz()));
        cv::Mat Rt = (cv::Mat_<float>(3, 4) <<
            R.at<float>(0), R.at<float>(1), R.at<float>(2), this->tx(),
            R.at<float>(3), R.at<float>(4), R.at<float>(5), this->ty(),
            R.at<float>(6), R.at<float>(7), R.at<float>(8), this->tz());

        image_point_list.resize(object_point_list.size());

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int32_t i = 0; i < object_point_list.size(); i++) {
            const auto& object_point = object_point_list[i];
            auto& image_point = image_point_list[i];
            cv::Mat Mw = (cv::Mat_<float>(4, 1) << object_point.x, object_point.y, object_point.z, 1);
            cv::Mat Mc = Rt * Mw;
            float Zc = Mc.at<float>(2);
            if (Zc <= 0) {
                /* Do not project points behind the camera */
                image_point = cv::Point2f(-1, -1);
                continue;
            }

            cv::Mat XY = K * Mc;
            float x = XY.at<float>(0);
            float y = XY.at<float>(1);
            float s = XY.at<float>(2);
            x /= s;
            y /= s;

            if (this->dist_coeff.empty() || this->dist_coeff.at<float>(0) == 0) {
                image_point.x = x;
                image_point.y = y;
            } else {
                /*** Distort ***/
                float u = (x - this->cx()) / this->fx();  /* from optical center*/
                float v = (y - this->cy()) / this->fy();  /* from optical center*/
                float r2 = u * u + v * v;
                float r4 = r2 * r2;
                float k1 = this->dist_coeff.at<float>(0);
                float k2 = this->dist_coeff.at<float>(1);
                float p1 = this->dist_coeff.at<float>(3);
                float p2 = this->dist_coeff.at<float>(4);
                u = u + u * (k1 * r2 + k2 * r4 /*+ k3 * r6 */) + (2 * p1 * u * v) + p2 * (r2 + 2 * u * u);
                v = v + v * (k1 * r2 + k2 * r4 /*+ k3 * r6 */) + (2 * p2 * u * v) + p1 * (r2 + 2 * v * v);
                image_point.x = u * this->fx() + this->cx();
                image_point.y = v * this->fy() + this->cy();
            }
        }
#else
        cv::projectPoints(object_point_list, this->rvec, this->tvec, this->K, this->dist_coeff, image_point_list);
#endif
    }

    void ConvertWorld2Camera(const std::vector<cv::Point3f>& object_point_in_world_list, std::vector<cv::Point3f>& object_point_in_camera_list)
    {
        /*** Mw -> Mc ***/
        /* Mc = [R t] * [M, 1] */
        cv::Mat K = this->K;
        cv::Mat R = MakeRotationMat(Rad2Deg(this->rx()), Rad2Deg(this->ry()), Rad2Deg(this->rz()));
        cv::Mat Rt = (cv::Mat_<float>(3, 4) <<
            R.at<float>(0), R.at<float>(1), R.at<float>(2), this->tx(),
            R.at<float>(3), R.at<float>(4), R.at<float>(5), this->ty(),
            R.at<float>(6), R.at<float>(7), R.at<float>(8), this->tz());

        object_point_in_camera_list.resize(object_point_in_world_list.size());

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int32_t i = 0; i < object_point_in_world_list.size(); i++) {
            const auto& object_point_in_world = object_point_in_world_list[i];
            auto& object_point_in_camera = object_point_in_camera_list[i];
            cv::Mat Mw = (cv::Mat_<float>(4, 1) << object_point_in_world.x, object_point_in_world.y, object_point_in_world.z, 1);
            cv::Mat Mc = Rt * Mw;
            object_point_in_camera.x = Mc.at<float>(0);
            object_point_in_camera.y = Mc.at<float>(1);
            object_point_in_camera.z = Mc.at<float>(2);
        }
    }

    void ConvertCamera2World(const std::vector<cv::Point3f>& object_point_in_camera_list, std::vector<cv::Point3f>& object_point_in_world_list)
    {
        /*** Mc -> Mw ***/
        /* Mc = [R t] * [Mw, 1] */
        /* -> [M, 1] = [R t]^1 * Mc <- Unable to get the inverse of [R t] because it's 4x3 */
        /* So, Mc = R * Mw + t */
        /* -> Mw = R^1 * (Mc - t) */
        cv::Mat R = MakeRotationMat(Rad2Deg(this->rx()), Rad2Deg(this->ry()), Rad2Deg(this->rz()));
        cv::Mat R_inv;
        cv::invert(R, R_inv);
        cv::Mat t = this->tvec;


        object_point_in_world_list.resize(object_point_in_camera_list.size());

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int32_t i = 0; i < object_point_in_camera_list.size(); i++) {
            const auto& object_point_in_camera = object_point_in_camera_list[i];
            auto& object_point_in_world = object_point_in_world_list[i];
            cv::Mat Mc = (cv::Mat_<float>(3, 1) << object_point_in_camera.x, object_point_in_camera.y, object_point_in_camera.z);
            cv::Mat Mw = R_inv * (Mc - t);
            object_point_in_world.x = Mw.at<float>(0);
            object_point_in_world.y = Mw.at<float>(1);
            object_point_in_world.z = Mw.at<float>(2);
        }
    }

    void ConvertImage2GroundPlane(const std::vector<cv::Point2f>& image_point_list, std::vector<cv::Point3f>& object_point_list)
    {
        /*** Image -> Mw ***/
        /*** Calculate point in ground plane (in world coordinate) ***/
        /* Main idea:*/
        /*   s * [x, y, 1] = K * [R t] * [M, 1]  */
        /*   s * [x, y, 1] = K * R * M + K * t */
        /*   s * Kinv * [x, y, 1] = R * M + t */
        /*   s * Kinv * [x, y, 1] - t = R * M */
        /*   Rinv * (s * Kinv * [x, y, 1] - t) = M */
        /* calculate s */
        /*   s * Rinv * Kinv * [x, y, 1] = M + R_inv * t */
        /*      where, M = (X, Y, Z), and we assume Y = 0(ground_plane) */
        /*      so , we can solve left[1] = R_inv * t[1](camera_height) */

        if (image_point_list.size() == 0) return;

        cv::Mat K = this->K;
        cv::Mat R = MakeRotationMat(Rad2Deg(this->rx()), Rad2Deg(this->ry()), Rad2Deg(this->rz()));
        cv::Mat K_inv;
        cv::invert(K, K_inv);
        cv::Mat R_inv;
        cv::invert(R, R_inv);
        cv::Mat t = this->tvec;

        /*** Undistort image point ***/
        std::vector<cv::Point2f> image_point_undistort;
        if (this->dist_coeff.empty() || this->dist_coeff.at<float>(0) == 0) {
            image_point_undistort = image_point_list;
        } else {
            cv::undistortPoints(image_point_list, image_point_undistort, this->K, this->dist_coeff, this->K);    /* don't use K_new */
        }

        object_point_list.resize(image_point_list.size());
        for (int32_t i = 0; i < object_point_list.size(); i++) {
            const auto& image_point = image_point_list[i];
            auto& object_point = object_point_list[i];

            float x = image_point_undistort[i].x;
            float y = image_point_undistort[i].y;
            if (y < EstimateVanishmentY()) {
                object_point.x = 999;
                object_point.y = 999;
                object_point.z = 999;
                continue;
            }

            cv::Mat XY = (cv::Mat_<float>(3, 1) << x, y, 1);

            /* calculate s */
            cv::Mat LEFT_WO_S = R_inv * K_inv * XY;
            cv::Mat RIGHT_WO_M = R_inv * t;         /* no need to add M because M[1] = 0 (ground plane)*/
            float s = RIGHT_WO_M.at<float>(1) / LEFT_WO_S.at<float>(1);

            /* calculate M */
            cv::Mat TEMP = R_inv * (s * K_inv * XY - t);

            object_point.x = TEMP.at<float>(0);
            object_point.y = TEMP.at<float>(1);
            object_point.z = TEMP.at<float>(2);
            if (object_point.z < 0) object_point.z = 999;
        }
    }

    void ConvertImage2Camera(std::vector<cv::Point2f>& image_point_list, const std::vector<float>& z_list, std::vector<cv::Point3f>& object_point_list)
    {
        /*** Image -> Mc ***/
        if (image_point_list.size() == 0) {
            /* Convert for all pixels on image, when image_point_list = empty */
            if (z_list.size() != this->width * this->height) {
                printf("[ConvertImage2Camera] Invalid z_list size\n");
                return;
            }
            /* Generate the original image point mat */
            /* todo: no need to generate every time */
            for (int32_t y = 0; y < this->height; y++) {
                for (int32_t x = 0; x < this->width; x++) {
                    image_point_list.push_back(cv::Point2f(float(x), float(y)));
                }
            }
        } else {
            /* Convert for the input pixels only */
            if (z_list.size() != image_point_list.size()) {
                printf("[ConvertImage2Camera] Invalid z_list size\n");
                return;
            }
        }

        /*** Undistort image point ***/
        std::vector<cv::Point2f> image_point_undistort;
        if (this->dist_coeff.empty() || this->dist_coeff.at<float>(0) == 0) {
            image_point_undistort = image_point_list;   /* todo: redundant copy. I do this just because to make code simple */
        } else {
            cv::undistortPoints(image_point_list, image_point_undistort, this->K, this->dist_coeff, this->K);    /* don't use K_new */
        }

        object_point_list.resize(image_point_list.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int32_t i = 0; i < object_point_list.size(); i++) {
            const auto& Zc = z_list[i];
            auto& object_point = object_point_list[i];

            float x = image_point_undistort[i].x;
            float y = image_point_undistort[i].y;

            float u = x - this->cx();
            float v = y - this->cy();
            float Xc = Zc * u / this->fx();
            float Yc = Zc * v / this->fy();
            object_point.x = Xc;
            object_point.y = Yc;
            object_point.z = Zc;
        }
    }

    void ConvertImage2World(std::vector<cv::Point2f>& image_point_list, const std::vector<float>& z_list, std::vector<cv::Point3f>& object_point_list)
    {
        /*** Image -> Mw ***/
        std::vector<cv::Point3f> object_point_in_camera_list;
        ConvertImage2Camera(image_point_list, z_list, object_point_in_camera_list);
        ConvertCamera2World(object_point_in_camera_list, object_point_list);
    }


    /* tan(theta) = delta / f */
    float EstimatePitch(float vanishment_y)
    {
        float pitch = std::atan2(this->cy() - vanishment_y, this->fy());
        return Rad2Deg(pitch);
    }

    float EstimateYaw(float vanishment_x)
    {
        float yaw = std::atan2(this->cx() - vanishment_x, this->fx());
        return Rad2Deg(yaw);
    }

    int32_t EstimateVanishmentY()
    {
        float fy = this->fy();
        float cy = this->cy();
        //float fy = this->K_new.at<float>(4);
        //float cy = this->K_new.at<float>(5);
        float px_from_center = std::tan(this->rx()) * fy;
        float vanishment_y = cy - px_from_center;
        return static_cast<int32_t>(vanishment_y);
    }

    int32_t EstimateVanishmentX()
    {
        float px_from_center = std::tan(this->ry()) * this->fx();
        float vanishment_x = this->cx() - px_from_center;
        return static_cast<int32_t>(vanishment_x);
    }


    /*** Other methods ***/
    template <typename T = float>
    static void PRINT_MAT_FLOAT(const cv::Mat& mat, int32_t size)
    {
        for (int32_t i = 0; i < size; i++) {
            printf("%d: %.3f\n", i, mat.at<T>(i));
        }
    }

    template <typename T = float>
    static cv::Mat MakeRotationMat(T x_deg, T y_deg, T z_deg)
    {
        T x_rad = Deg2Rad(x_deg);
        T y_rad = Deg2Rad(y_deg);
        T z_rad = Deg2Rad(z_deg);
#if 0
        /* Rotation Matrix with Euler Angle */
        cv::Mat R_x = (cv::Mat_<T>(3, 3) <<
            1, 0, 0,
            0, std::cos(x_rad), -std::sin(x_rad),
            0, std::sin(x_rad), std::cos(x_rad));

        cv::Mat R_y = (cv::Mat_<T>(3, 3) <<
            std::cos(y_rad), 0, std::sin(y_rad),
            0, 1, 0,
            -std::sin(y_rad), 0, std::cos(y_rad));

        cv::Mat R_z = (cv::Mat_<T>(3, 3) <<
            std::cos(z_rad), -std::sin(z_rad), 0,
            std::sin(z_rad), std::cos(z_rad), 0,
            0, 0, 1);

        cv::Mat R = R_z * R_x * R_y;
#else
        /* Rodrigues */
        cv::Mat rvec = (cv::Mat_<T>(3, 1) << x_rad, y_rad, z_rad);
        cv::Mat R;
        cv::Rodrigues(rvec, R);
#endif
        return R;
    }

    static void RotateObject(float x_deg, float y_deg, float z_deg, std::vector<cv::Point3f>& object_point_list)
    {
        cv::Mat R = MakeRotationMat(x_deg, y_deg, z_deg);
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
};

#endif
