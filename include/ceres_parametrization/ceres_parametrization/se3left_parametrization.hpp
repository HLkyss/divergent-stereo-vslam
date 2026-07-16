/**
*    This file is part of OV²SLAM.
*    
*    Copyright (C) 2020 ONERA
*
*    For more information see <https://github.com/ov2slam/ov2slam>
*
*    OV²SLAM is free software: you can redistribute it and/or modify
*    it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    OV²SLAM is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU General Public License for more details.
*
*    You should have received a copy of the GNU General Public License
*    along with OV²SLAM.  If not, see <https://www.gnu.org/licenses/>.
*
*    Authors: Maxime Ferrera     <maxime.ferrera at gmail dot com> (ONERA, DTIS - IVA),
*             Alexandre Eudes    <first.last at onera dot fr>      (ONERA, DTIS - IVA),
*             Julien Moras       <first.last at onera dot fr>      (ONERA, DTIS - IVA),
*             Martial Sanfourche <first.last at onera dot fr>      (ONERA, DTIS - IVA)
*/
#pragma once


#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <ceres/ceres.h>

/*
    SE(3) Parametrization such as:
    1. T + dT = Exp(dT) * T 
    2. T o X = T^(-1) * X (i.e. T: cam -> world)  
*/

// 添加一个约束，确保 map_id_posespar_l_ 和 map_id_posespar_r_ 相等（即左目和右目的位姿必须相同）
// 计算左目和右目的位姿之间的差异：使用 ceres::SizedCostFunction 来定义该函数，它接收两个位姿作为输入，输出它们之间的差异（残差）。
struct CostFunctor {
    template <typename T>
    bool operator()(const T* const x, const T* const y, const T* const z, T* residual) const {
        residual[0] = 10.0 - x[0];
        residual[1] = 10.0 - y[0];
        residual[2] = 10.0 - z[0];
        return true;
    }
};

struct CostFunctor2 {
    template <typename T>
    bool operator()(T const* const parameters, T* residual) const {
        residual[0] = 10.0 - parameters[0];
        residual[1] = 10.0 - parameters[1];
        residual[2] = 10.0 - parameters[2];
        return true;
    }
};

//class PoseEqualityConstraint : public ceres::SizedCostFunction<7, 7, 7> {
struct PoseEqualityConstraint {

    PoseEqualityConstraint() {}

    template <typename T>
    bool operator()(const T* const* parameters, T* residuals) const {

        // 外参位姿变换  [tx, ty, tz, qw, qx, qy, qz]
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> trl(parameters[0]); // 外参位姿变换
        Eigen::Map<const Eigen::Quaternion<T>> qrl(parameters[0]+3);
        Sophus::SE3<T> Trl(qrl, trl); // 左目到右目的变换

        // 左相机位姿（待优化）
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> twc_l(parameters[1]); // 左相机位姿（左到世界）
        Eigen::Map<const Eigen::Quaternion<T>> qwc_l(parameters[1]+3);
        Sophus::SE3<T> Twc_l(qwc_l,twc_l);
        Sophus::SE3<T> Tcw_l = Twc_l.inverse();
        // 右相机位姿（待优化）
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> twc_r(parameters[2]); // 右相机位姿（右到世界）
        Eigen::Map<const Eigen::Quaternion<T>> qwc_r(parameters[2]+3);
        Sophus::SE3<T> Twc_r(qwc_r,twc_r);
        Sophus::SE3<T> Tcw_r = Twc_r.inverse();
        Sophus::SE3<T> Twc_l_pred = Twc_r * Trl;

        // 误差
        Sophus::SE3<T> T_diff = Twc_l_pred.inverse() * Tcw_l; // 计算左目位姿和计算得到的左目位姿的误差
        Eigen::Matrix<T, 3, 1> translation_error = T_diff.translation(); // 平移误差
        Eigen::Quaternion<T> rotation_error = T_diff.unit_quaternion(); // 旋转误差

        // 将平移和旋转误差放入残差项
        residuals[0] = translation_error.x(); // x 平移误差
        residuals[1] = translation_error.y(); // y 平移误差
        residuals[2] = translation_error.z(); // z 平移误差
        residuals[3] = rotation_error.w();    // 旋转误差的四元数 w 分量
        residuals[4] = rotation_error.x();    // 旋转误差的四元数 x 分量
        residuals[5] = rotation_error.y();    // 旋转误差的四元数 y 分量
        residuals[6] = rotation_error.z();    // 旋转误差的四元数 z 分量

        return true;
    }

};

struct PoseEqualityConstraint2 {

    PoseEqualityConstraint2() {}

    template <typename T>
    bool operator()(const T* const parameters0, const T* const parameters1, const T* const parameters2, T* residuals) const {

        // 外参位姿变换  [tx, ty, tz, qw, qx, qy, qz]
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> trl(parameters0); // 外参位姿变换
        Eigen::Map<const Eigen::Quaternion<T>> qrl(parameters0+3);
        Sophus::SE3<T> Trl(qrl, trl); // 左目到右目的变换
//        std::cout << "Trl: " << Trl.matrix() << std::endl;

        // 左相机位姿（待优化）
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> twc_l(parameters1); // 左相机位姿（左到世界）
        Eigen::Map<const Eigen::Quaternion<T>> qwc_l(parameters1+3);
        Sophus::SE3<T> Twc_l(qwc_l,twc_l);
        Sophus::SE3<T> Tcw_l = Twc_l.inverse();
        // 右相机位姿（待优化）
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> twc_r(parameters2); // 右相机位姿（右到世界）
        Eigen::Map<const Eigen::Quaternion<T>> qwc_r(parameters2+3);
        Sophus::SE3<T> Twc_r(qwc_r,twc_r);
        Sophus::SE3<T> Tcw_r = Twc_r.inverse();
        Sophus::SE3<T> Twc_l_pred = Twc_r * Trl;

        // 误差
        Sophus::SE3<T> T_diff = Twc_l_pred.inverse() * Twc_l; // 计算左目位姿和计算得到的左目位姿的误差
//        std::cout << "T_diff: " << T_diff.matrix() << std::endl;
        Eigen::Matrix<T, 3, 1> translation_error = T_diff.translation(); // 平移误差
        Eigen::Quaternion<T> rotation_error = T_diff.unit_quaternion(); // 旋转误差

        // 将平移和旋转误差放入残差项
        residuals[0] = translation_error.x(); // x 平移误差
        residuals[1] = translation_error.y(); // y 平移误差
        residuals[2] = translation_error.z(); // z 平移误差
        residuals[3] = rotation_error.w();    // 旋转误差的四元数 w 分量
        residuals[4] = rotation_error.x();    // 旋转误差的四元数 x 分量
        residuals[5] = rotation_error.y();    // 旋转误差的四元数 y 分量
        residuals[6] = rotation_error.z();    // 旋转误差的四元数 z 分量

        return true;
    }

};

class PoseEqualityConstraint3 : public ceres::SizedCostFunction<2, 4, 7, 7, 3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PoseEqualityConstraint3(const Sophus::SE3d Trl, const double sigma = 1.)
    : Trl_(Trl)
    {
        sqrt_cov_ = sigma * Eigen::Matrix2d::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix2d sqrt_cov_, sqrt_info_;
private:
    Sophus::SE3d Trl_;
};

class SE3LeftParameterization : public ceres::LocalParameterization {
public:
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const 
    {
        Eigen::Map<const Eigen::Vector3d> t(x);
        Eigen::Map<const Eigen::Quaterniond> q(x+3);

        Eigen::Map<const Eigen::Matrix<double,6,1>> vdelta(delta);

        // Left update
        Sophus::SE3d upT = Sophus::SE3d::exp(vdelta) * Sophus::SE3d(q,t);

        Eigen::Map<Eigen::Vector3d> upt(x_plus_delta);
        Eigen::Map<Eigen::Quaterniond> upq(x_plus_delta+3);

        upt = upT.translation();
        upq = upT.unit_quaternion();

        return true;
    }

    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const
    {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > J(jacobian);
        J.topRows<6>().setIdentity();
        J.bottomRows<1>().setZero();
        return true;
    }

    virtual int GlobalSize() const { return 7; }
    virtual int LocalSize() const { return 6; }
};


class LeftSE3RelativePoseError : public ceres::SizedCostFunction<6, 7, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LeftSE3RelativePoseError(const Sophus::SE3d &Tc0c1,
                            const double sigma = 1.)
        : Tc0c1_(Tc0c1)
    {
        sqrt_cov_ = sigma * Eigen::Matrix<double,6,6>::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix<double,6,6> sqrt_cov_, sqrt_info_;
private:
    Sophus::SE3d Tc0c1_;
};


// Cost functions with SE(3) pose parametrized as
// T cam -> world
namespace DirectLeftSE3 {

class ReprojectionErrorKSE3XYZ : public ceres::SizedCostFunction<2, 4, 7, 3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionErrorKSE3XYZ(const double u, const double v,
                            const double sigma = 1.)
        : unpx_(u,v)
    {
        sqrt_cov_ = sigma * Eigen::Matrix2d::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix2d sqrt_cov_, sqrt_info_;
private:
    Eigen::Vector2d unpx_;
};

class ReprojectionErrorRightCamKSE3XYZ : public ceres::SizedCostFunction<2, 4, 7, 7, 3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionErrorRightCamKSE3XYZ(const double u, const double v,
                            const double sigma = 1.)
        : unpx_(u,v)
    {
        sqrt_cov_ = sigma * Eigen::Matrix2d::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix2d sqrt_cov_, sqrt_info_;
private:
    Eigen::Vector2d unpx_;
};


class ReprojectionErrorSE3 : public ceres::SizedCostFunction<2, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionErrorSE3(const double u, const double v,
            double fx, double fy, double cx, double cy, 
            const Eigen::Vector3d &wpt, const double sigma = 1.)
        : unpx_(u,v), wpt_(wpt), fx_(fx), fy_(fy), cx_(cx), cy_(cy)
    {
        sqrt_cov_ = sigma * Eigen::Matrix2d::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix2d sqrt_cov_, sqrt_info_;
private:
    Eigen::Vector2d unpx_;
    Eigen::Vector3d wpt_;
    double fx_, fy_, cx_, cy_;
};


class ReprojectionErrorKSE3AnchInvDepth : public ceres::SizedCostFunction<2, 4, 7, 7, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionErrorKSE3AnchInvDepth(
                            const double u, const double v,
                            const double uanch, const double vanch,
                            const double sigma = 1.)
        : unpx_(u,v), anchpx_(uanch,vanch,1.)
    {
        sqrt_cov_ = sigma * Eigen::Matrix2d::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix2d sqrt_cov_, sqrt_info_;
private:
    Eigen::Vector2d unpx_;
    Eigen::Vector3d anchpx_;
};

//class ReprojectionErrorKSE3AnchInvDepth_r : public ceres::SizedCostFunction<2, 4, 7, 7, 1>
class ReprojectionErrorKSE3AnchInvDepth_r : public ceres::SizedCostFunction<2, 4, 7, 7, 1, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionErrorKSE3AnchInvDepth_r(
            const double u, const double v,
            const double uanch, const double vanch,
            const double sigma = 1.)
            : runpx_(u,v), ranchpx_(uanch,vanch,1.)
    {
        sqrt_cov_ = sigma * Eigen::Matrix2d::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix2d sqrt_cov_, sqrt_info_;
private:
    Eigen::Vector2d runpx_;
    Eigen::Vector3d ranchpx_;
};

class ReprojectionErrorRightAnchCamKSE3AnchInvDepth : public ceres::SizedCostFunction<2, 4, 4, 7, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionErrorRightAnchCamKSE3AnchInvDepth(
                    const double ur, const double vr,
                    const double uanch, const double vanch,
                    const double  sigmar = 1.)
        : runpx_(ur,vr), anchpx_(uanch,vanch,1.)
    {
        sqrt_cov_.setZero();
        sqrt_cov_ = sigmar * Eigen::Matrix2d::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,  //具体定义重投影误差求解方法
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated
    // in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix2d sqrt_cov_, sqrt_info_;
private:
    Eigen::Vector2d runpx_;
    Eigen::Vector3d anchpx_;
};

class ReprojectionErrorRightAnchCamKSE3AnchInvDepth_r : public ceres::SizedCostFunction<2, 4, 4, 7, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionErrorRightAnchCamKSE3AnchInvDepth_r(
            const double ur, const double vr,
            const double uanch, const double vanch,
            const double  sigmar = 1.)
            : lunpx_(ur,vr), ranchpx_(uanch,vanch,1.)
    {
        sqrt_cov_.setZero();
        sqrt_cov_ = sigmar * Eigen::Matrix2d::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,  //具体定义重投影误差求解方法
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated
    // in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix2d sqrt_cov_, sqrt_info_;
private:
    Eigen::Vector2d lunpx_;
    Eigen::Vector3d ranchpx_;
};


class ReprojectionErrorRightCamKSE3AnchInvDepth : public ceres::SizedCostFunction<2, 4, 4, 7, 7, 7, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionErrorRightCamKSE3AnchInvDepth(
                    const double ur, const double vr,
                    const double uanch, const double vanch,
                    const double  sigmar = 1.)
        : runpx_(ur,vr), anchpx_(uanch,vanch,1.)
    {
        sqrt_cov_.setZero();
        sqrt_cov_ = sigmar * Eigen::Matrix2d::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated
    // in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix2d sqrt_cov_, sqrt_info_;
private:
    Eigen::Vector2d runpx_;
    Eigen::Vector3d anchpx_;
};

class ReprojectionErrorRightCamKSE3AnchInvDepth_r : public ceres::SizedCostFunction<2, 4, 4, 7, 7, 7, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionErrorRightCamKSE3AnchInvDepth_r(
            const double ur, const double vr,
            const double uanch, const double vanch,
            const double  sigmar = 1.)
            : unpx_(ur,vr), ranchpx_(uanch,vanch,1.)
    {
        sqrt_cov_.setZero();
        sqrt_cov_ = sigmar * Eigen::Matrix2d::Identity();
        sqrt_info_ = sqrt_cov_.inverse();
    }

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    // Mutable var. that will be updated
    // in const Evaluate()
    mutable double chi2err_;
    mutable bool isdepthpositive_;
    Eigen::Matrix2d sqrt_cov_, sqrt_info_;
private:
    Eigen::Vector2d unpx_;
    Eigen::Vector3d ranchpx_;
};


} // end namespace