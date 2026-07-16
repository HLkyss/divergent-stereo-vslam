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

#include "ceres_parametrization.hpp"



bool PoseEqualityConstraint3::Evaluate(double const* const* parameters,
                                        double* residuals, double** jacobians) const
{
    // 外参位姿变换  [tx, ty, tz, qw, qx, qy, qz]
    Sophus::SE3d Trl = Trl_; // 左目到右目的变换
    // 左相机位姿（待优化）
    Eigen::Map<const Eigen::Vector3d> twc_l(parameters[0]); // 左相机位姿（左到世界）
    Eigen::Map<const Eigen::Quaterniond> qwc_l(parameters[0]+3);
    Sophus::SE3d Twc_l(qwc_l,twc_l);
    Sophus::SE3d Tcw_l = Twc_l.inverse();
    // 右相机位姿（待优化）
    Eigen::Map<const Eigen::Vector3d> twc_r(parameters[1]); // 右相机位姿（右到世界）
    Eigen::Map<const Eigen::Quaterniond> qwc_r(parameters[1]+3);
    Sophus::SE3d Twc_r(qwc_r,twc_r);
    Sophus::SE3d Tcw_r = Twc_r.inverse();
    Sophus::SE3d Twc_l_pred = Twc_r * Trl;
    // 误差
    Sophus::SE3d T_diff = Twc_l_pred.inverse() * Twc_l; // 计算左目位姿和计算得到的左目位姿的误差
//        std::cout << "T_diff: " << T_diff.matrix() << std::endl;
    Eigen::Vector3d translation_error = T_diff.translation(); // 平移误差
    Eigen::Quaterniond rotation_error = T_diff.unit_quaternion(); // 旋转误差

    Eigen::Map<Sophus::SE3d> werr(residuals);
    werr = T_diff;
    // 将平移和旋转误差放入残差项
    residuals[0] = translation_error.x(); // x 平移误差
    residuals[1] = translation_error.y(); // y 平移误差
    residuals[2] = translation_error.z(); // z 平移误差
    residuals[3] = rotation_error.w();    // 旋转误差的四元数 w 分量
    residuals[4] = rotation_error.x();    // 旋转误差的四元数 x 分量
    residuals[5] = rotation_error.y();    // 旋转误差的四元数 y 分量
    residuals[6] = rotation_error.z();    // 旋转误差的四元数 z 分量

//    if(jacobians != NULL)
//    {
//        const double linvz2 = linvz * linvz;
//
//        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_lcam;
//        J_lcam << linvz * lcalib(0), 0., -lcampt.x() * linvz2 * lcalib(0),
//                0., linvz * lcalib(1), -lcampt.y() * linvz2 * lcalib(1);
//
//        Eigen::Matrix<double,2,3> J_lRcw;
//
//        if( jacobians[1] != NULL || jacobians[2] != NULL )
//        {
//            J_lRcw.noalias() = J_lcam * Tcw.rotationMatrix();
//        }
//
//        if(jacobians[0] != NULL)
//        {
//            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_lcalib(jacobians[0]);
//            J_lcalib.setZero();
//            J_lcalib(0,0) = lcampt.x() * linvz;
//            J_lcalib(0,2) = 1.;
//            J_lcalib(1,1) = lcampt.y() * linvz;
//            J_lcalib(1,3) = 1.;
//
//            J_lcalib = sqrt_info_ * J_lcalib.eval();
//        }
//        if(jacobians[1] != NULL)
//        {
//            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3pose(jacobians[1]);
//            J_se3pose.setZero();
//
//            J_se3pose.block<2,3>(0,0) = -1. * J_lRcw;
//            J_se3pose.block<2,3>(0,3).noalias() = J_lRcw * Sophus::SO3d::hat(wpt);
//
//            J_se3pose = sqrt_info_ * J_se3pose.eval();
//        }
//        if(jacobians[2] != NULL)
//        {
//            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J_wpt(jacobians[2]);
//            J_wpt.setZero();
//            J_wpt.block<2,3>(0,0) = J_lRcw;
//
//            J_wpt = sqrt_info_ * J_wpt.eval();
//        }
//    }

    return true;
}

bool LeftSE3RelativePoseError::Evaluate(double const* const* parameters,
                double* residuals, double** jacobians) const
{
    // [tx, ty, tz, qw, qx, qy, qz]
    Eigen::Map<const Eigen::Vector3d> twc0(parameters[0]);
    Eigen::Map<const Eigen::Quaterniond> qwc0(parameters[0]+3);

    // [tx, ty, tz, qw, qx, qy, qz]
    Eigen::Map<const Eigen::Vector3d> twc1(parameters[1]);
    Eigen::Map<const Eigen::Quaterniond> qwc1(parameters[1]+3);

    Sophus::SE3d Twc0(qwc0,twc0);
    Sophus::SE3d Twc1(qwc1,twc1);
    Sophus::SE3d Tc1w = Twc1.inverse();

    Sophus::SE3d Tc1c0 = Tc1w * Twc0;

    Sophus::SE3d err = (Tc1c0 * Tc0c1_);

    Eigen::Matrix<double,6,1> verr = err.log();

    Eigen::Map<Eigen::Matrix<double,6,1>> werr(residuals);
    werr = sqrt_info_ * verr;

    // Update chi2err info for 
    // post optim checking
    chi2err_ = 0.;
    for( int i = 0 ; i < 6 ; i++ ) 
        chi2err_ += std::pow(residuals[i],2);

    if(jacobians != NULL)
    {
        Eigen::Matrix3d skew_rho = Sophus::SO3d::hat(verr.block<3,1>(0,0));
        Eigen::Matrix3d skew_omega = Sophus::SO3d::hat(err.so3().log());

        Eigen::Matrix<double,6,6> I6x6 = Eigen::Matrix<double,6,6>::Identity();

        if(jacobians[0] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J_Tc0(jacobians[0]);
            J_Tc0.setZero();

            // Adapted from Strasdat PhD Appendix

            Eigen::Matrix<double,6,6> J_c0;
            J_c0.setZero();
            J_c0.block<3,3>(0,0).noalias() = -1. * skew_omega;
            J_c0.block<3,3>(0,3).noalias() = -1. * skew_rho;
            J_c0.block<3,3>(3,3).noalias() = -1. * skew_omega;

            J_Tc0.block<6,6>(0,0).noalias() = (I6x6 + 0.5 * J_c0) * Tc1w.Adj();

            J_Tc0 = sqrt_info_ * J_Tc0.eval();
        }
        if(jacobians[1] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J_Tc1(jacobians[1]);
            J_Tc1.setZero();

            Eigen::Matrix<double,6,6> J_c1;
            J_c1.setZero();
            J_c1.block<3,3>(0,0).noalias() = skew_omega;
            J_c1.block<3,3>(0,3).noalias() = skew_rho;
            J_c1.block<3,3>(3,3).noalias() = skew_omega;

            J_Tc1.block<6,6>(0,0).noalias() = -1. * (I6x6 + 0.5 * J_c1) * (Twc0 * Tc0c1_).inverse().Adj();

            J_Tc1 = sqrt_info_ * J_Tc1.eval();
        }
    }

    return true;
}

namespace DirectLeftSE3 {


bool ReprojectionErrorKSE3XYZ::Evaluate(double const* const* parameters,
                double* residuals, double** jacobians) const
{
    // [fx, fy, cx, cy]
    Eigen::Map<const Eigen::Vector4d> lcalib(parameters[0]);

    // [tx, ty, tz, qw, qx, qy, qz]
    Eigen::Map<const Eigen::Vector3d> twc(parameters[1]);
    Eigen::Map<const Eigen::Quaterniond> qwc(parameters[1]+3);

    Sophus::SE3d Twc(qwc,twc);
    Sophus::SE3d Tcw = Twc.inverse();

    // [x,y,z]
    Eigen::Map<const Eigen::Vector3d> wpt(parameters[2]);

    // Compute left/right reproj err
    Eigen::Vector2d pred;

    Eigen::Vector3d lcampt = Tcw * wpt;

    const double linvz = 1. / lcampt.z();

    pred << lcalib(0) * lcampt.x() * linvz + lcalib(2),
            lcalib(1) * lcampt.y() * linvz + lcalib(3);
    
    Eigen::Map<Eigen::Vector2d> werr(residuals);
    werr = sqrt_info_ * (pred - unpx_);

    // Update chi2err and depthpos info for 
    // post optim checking
    chi2err_ = 0.;
    for( int i = 0 ; i < 2 ; i++ ) 
        chi2err_ += std::pow(residuals[i],2);
    
    // std::cout << "\n chi2 err : " << chi2err_;

    isdepthpositive_ = true;    
    if( lcampt.z() <= 0 )
        isdepthpositive_ = false;

    if(jacobians != NULL)
    {
        const double linvz2 = linvz * linvz;

        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_lcam;
        J_lcam << linvz * lcalib(0), 0., -lcampt.x() * linvz2 * lcalib(0),
                0., linvz * lcalib(1), -lcampt.y() * linvz2 * lcalib(1);
        
        Eigen::Matrix<double,2,3> J_lRcw; 

        if( jacobians[1] != NULL || jacobians[2] != NULL )
        {
            J_lRcw.noalias() = J_lcam * Tcw.rotationMatrix();
        }

        if(jacobians[0] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_lcalib(jacobians[0]);
            J_lcalib.setZero();
            J_lcalib(0,0) = lcampt.x() * linvz;
            J_lcalib(0,2) = 1.;
            J_lcalib(1,1) = lcampt.y() * linvz;
            J_lcalib(1,3) = 1.;

            J_lcalib = sqrt_info_ * J_lcalib.eval();        
        }
        if(jacobians[1] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3pose(jacobians[1]);
            J_se3pose.setZero();

            J_se3pose.block<2,3>(0,0) = -1. * J_lRcw;
            J_se3pose.block<2,3>(0,3).noalias() = J_lRcw * Sophus::SO3d::hat(wpt);

            J_se3pose = sqrt_info_ * J_se3pose.eval();
        }
        if(jacobians[2] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J_wpt(jacobians[2]);
            J_wpt.setZero();
            J_wpt.block<2,3>(0,0) = J_lRcw;

            J_wpt = sqrt_info_ * J_wpt.eval();
        }
    }

    return true;
}


bool ReprojectionErrorRightCamKSE3XYZ::Evaluate(double const* const* parameters,
                double* residuals, double** jacobians) const
{
    // [fx, fy, cx, cy]
    Eigen::Map<const Eigen::Vector4d> rcalib(parameters[0]);

    // [tx, ty, tz, qw, qx, qy, qz]
    Eigen::Map<const Eigen::Vector3d> twc(parameters[1]);
    Eigen::Map<const Eigen::Quaterniond> qwc(parameters[1]+3);

    Eigen::Map<const Eigen::Vector3d> trl(parameters[2]);
    Eigen::Map<const Eigen::Quaterniond> qrl(parameters[2]+3);

    Sophus::SE3d Twc(qwc,twc);
    Sophus::SE3d Tcw = Twc.inverse();
    Sophus::SE3d Trl(qrl,trl);

    // [x,y,z]
    Eigen::Map<const Eigen::Vector3d> wpt(parameters[3]);

    // Compute left/right reproj err
    Eigen::Vector2d pred;

    Eigen::Vector3d rcampt = Trl * Tcw * wpt;

    const double rinvz = 1. / rcampt.z();

    pred << rcalib(0) * rcampt.x() * rinvz + rcalib(2),
            rcalib(1) * rcampt.y() * rinvz + rcalib(3);
    
    Eigen::Map<Eigen::Vector2d> werr(residuals);
    werr = sqrt_info_ * (pred - unpx_);

    // Update chi2err and depthpos info for 
    // post optim checking
    chi2err_ = 0.;
    for( int i = 0 ; i < 2 ; i++ ) 
        chi2err_ += std::pow(residuals[i],2);

    isdepthpositive_ = true;    
    if( rcampt.z() <= 0 )
        isdepthpositive_ = false;

    if(jacobians != NULL)
    {
        const double rinvz2 = rinvz * rinvz;

        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_rcam;
        J_rcam << rinvz * rcalib(0), 0., -rcampt.x() * rinvz2 * rcalib(0),
                0., rinvz * rcalib(1), -rcampt.y() * rinvz2 * rcalib(1);

        Eigen::Matrix<double,2,3> J_rRcw; 

        if( jacobians[1] != NULL || jacobians[3] != NULL )
        {
            Eigen::Matrix3d Rcw = Tcw.rotationMatrix();
            J_rRcw.noalias() = J_rcam * Trl.rotationMatrix() * Rcw;
        }

        if(jacobians[0] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_rcalib(jacobians[0]);
            J_rcalib.setZero();
            J_rcalib(0,0) = rcampt.x() * rinvz;
            J_rcalib(0,2) = 1.;
            J_rcalib(1,1) = rcampt.y() * rinvz;
            J_rcalib(1,3) = 1.;

            J_rcalib = sqrt_info_ * J_rcalib.eval();   
        }
        if(jacobians[1] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3pose(jacobians[1]);
            J_se3pose.setZero();

            Eigen::Matrix3d skew_wpt = Sophus::SO3d::hat(wpt);

            J_se3pose.block<2,3>(0,0) = -1. * J_rRcw;
            J_se3pose.block<2,3>(0,3).noalias() = J_rRcw * skew_wpt;

            J_se3pose = sqrt_info_ * J_se3pose.eval();
        }
        if(jacobians[2] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3extrin(jacobians[2]);
            J_se3extrin.setZero();

            // TODO
        }
        if(jacobians[3] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J_wpt(jacobians[3]);
            J_wpt.setZero();
            J_wpt.block<2,3>(0,0) = J_rRcw;

            J_wpt = sqrt_info_ * J_wpt.eval();
        }
    }

    return true;
}


bool ReprojectionErrorSE3::Evaluate(double const* const* parameters,
                double* residuals, double** jacobians) const
{
    // [tx, ty, tz, qw, qx, qy, qz]
    Eigen::Map<const Eigen::Vector3d> twc(parameters[0]);
    Eigen::Map<const Eigen::Quaterniond> qwc(parameters[0]+3);

    Sophus::SE3d Twc(qwc,twc);
    Sophus::SE3d Tcw = Twc.inverse();

    // Compute left/right reproj err
    Eigen::Vector2d pred;

    Eigen::Vector3d lcampt = Tcw * wpt_;

    const double linvz = 1. / lcampt.z();

    pred << fx_ * lcampt.x() * linvz + cx_,
            fy_ * lcampt.y() * linvz + cy_;
    
    Eigen::Map<Eigen::Vector2d> werr(residuals);
    werr = sqrt_info_ * (pred - unpx_);

    // Update chi2err and depthpos info for 
    // post optim checking
    chi2err_ = 0.;
    for( int i = 0 ; i < 2 ; i++ ) 
        chi2err_ += std::pow(residuals[i],2);

    isdepthpositive_ = true;    
    if( lcampt.z() <= 0 )
        isdepthpositive_ = false;

    if(jacobians != NULL)
    {
        const double linvz2 = linvz * linvz;

        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_lcam;
        J_lcam << linvz * fx_, 0., -lcampt.x() * linvz2 * fx_,
                0., linvz * fy_, -lcampt.y() * linvz2 * fy_;
        
        Eigen::Matrix<double,2,3> J_lRcw; 
        J_lRcw.noalias() = J_lcam * Tcw.rotationMatrix();

        if(jacobians[0] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3pose(jacobians[0]);
            J_se3pose.setZero();

            J_se3pose.block<2,3>(0,0).noalias() = -1. * J_lRcw;
            J_se3pose.block<2,3>(0,3).noalias() = J_lRcw * Sophus::SO3d::hat(wpt_);

            J_se3pose = sqrt_info_ * J_se3pose.eval();
        }
    }

    return true;
}


bool ReprojectionErrorKSE3AnchInvDepth::Evaluate(double const* const* parameters,
                double* residuals, double** jacobians) const
{
    // [fx, fy, cx, cy]
    Eigen::Map<const Eigen::Vector4d> lcalib(parameters[0]);

    // [tx, ty, tz, qw, qx, qy, qz]
    Eigen::Map<const Eigen::Vector3d> twanch(parameters[1]);
    Eigen::Map<const Eigen::Quaterniond> qwanch(parameters[1]+3);

    Eigen::Map<const Eigen::Vector3d> twc(parameters[2]);
    Eigen::Map<const Eigen::Quaterniond> qwc(parameters[2]+3);

    Sophus::SE3d Twanch(qwanch,twanch);
    Sophus::SE3d Twc(qwc,twc);
    Sophus::SE3d Tcw = Twc.inverse();

    // [1/z_anch]
    const double zanch = 1. / parameters[3][0];

    Eigen::Matrix3d invK, K;
    K << lcalib(0), 0., lcalib(2),
        0., lcalib(1), lcalib(3),
        0., 0., 1.;
    invK = K.inverse();

    Eigen::Vector3d anchpt = zanch * invK * anchpx_;
    Eigen::Vector3d wpt = Twanch * anchpt;

    // Compute left/right reproj err
    Eigen::Vector2d pred;

    Eigen::Vector3d lcampt = Tcw * wpt;

    const double linvz = 1. / lcampt.z();

    pred << lcalib(0) * lcampt.x() * linvz + lcalib(2),
            lcalib(1) * lcampt.y() * linvz + lcalib(3);

    Eigen::Map<Eigen::Vector2d> werr(residuals);
    werr = sqrt_info_ * (pred - unpx_);

    // Update chi2err and depthpos info for 
    // post optim checking
    chi2err_ = 0.;
    for( int i = 0 ; i < 2 ; i++ ) 
        chi2err_ += std::pow(residuals[i],2);


    isdepthpositive_ = true;    
    if( lcampt.z() <= 0 )
        isdepthpositive_ = false;

    if(jacobians != NULL)
    {
        const double linvz2 = linvz * linvz;

        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_lcam;
        J_lcam << linvz * lcalib(0), 0., -lcampt.x() * linvz2 * lcalib(0),
                0., linvz * lcalib(1), -lcampt.y() * linvz2 * lcalib(1);

        Eigen::Matrix<double,2,3> J_lRcw; 

        if( jacobians[1] != NULL || jacobians[2] != NULL 
            || jacobians[3] != NULL )
        {
            Eigen::Matrix3d Rcw = Tcw.rotationMatrix();
            J_lRcw.noalias() = J_lcam * Rcw;
        }

        if(jacobians[0] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_lcalib(jacobians[0]);
            J_lcalib.setZero();
            
            // TODO       
        }
        if(jacobians[1] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3anch(jacobians[1]);
            J_se3anch.setZero();

            Eigen::Matrix3d skew_wpt = Sophus::SO3d::hat(wpt);

            J_se3anch.block<2,3>(0,0) = J_lRcw;
            J_se3anch.block<2,3>(0,3).noalias() = -1. * J_lRcw * skew_wpt;

            J_se3anch = sqrt_info_ * J_se3anch.eval();
        }
        if(jacobians[2] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3pose(jacobians[2]);
            J_se3pose.setZero();

            Eigen::Matrix3d skew_wpt = Sophus::SO3d::hat(wpt);

            J_se3pose.block<2,3>(0,0) = -1. * J_lRcw;
            J_se3pose.block<2,3>(0,3).noalias() = J_lRcw * skew_wpt;

            J_se3pose = sqrt_info_ * J_se3pose.eval();
        }
        if(jacobians[3] != NULL)
        {
            Eigen::Map<Eigen::Vector2d> J_invpt(jacobians[3]);
            Eigen::Vector3d J_lambda = -1. * zanch * Twanch.rotationMatrix() * anchpt;
            J_invpt.noalias() = J_lRcw * J_lambda;

            J_invpt = sqrt_info_ * J_invpt.eval();
        }
    }

    return true;
}

bool ReprojectionErrorKSE3AnchInvDepth_r::Evaluate(double const* const* parameters,
                                                 double* residuals, double** jacobians) const
{
    // [fx, fy, cx, cy]
//    Eigen::Map<const Eigen::Vector4d> lcalib(parameters[0]);
    Eigen::Map<const Eigen::Vector4d> rcalib(parameters[0]);

    // [tx, ty, tz, qw, qx, qy, qz]
    Eigen::Map<const Eigen::Vector3d> twanch(parameters[1]);//锚定关键帧的位姿 左目到世界
    Eigen::Map<const Eigen::Quaterniond> qwanch(parameters[1]+3);

    Eigen::Map<const Eigen::Vector3d> twc(parameters[2]);//当前关键帧的位姿 左目到世界
    Eigen::Map<const Eigen::Quaterniond> qwc(parameters[2]+3);

    Eigen::Map<const Eigen::Vector3d> trl(parameters[4]);
    Eigen::Map<const Eigen::Quaterniond> qrl(parameters[4]+3);
    Sophus::SE3d Trl(qrl,trl);
    Sophus::SE3d Tlr = Trl.inverse();

    Sophus::SE3d Twanch(qwanch,twanch);//锚定关键帧的位姿 左目到世界
    Twanch = Twanch*Tlr;//转换成右目位姿（左目到世界*右到左）
    Sophus::SE3d Twc(qwc,twc);//当前关键帧的位姿 左目到世界
    Twc = Twc*Tlr;//转换成右目位姿（左目到世界*右到左）
    Sophus::SE3d Tcw = Twc.inverse();

    // [1/z_anch]
    const double zanch_r = 1. / parameters[3][0];//右目深度

    Eigen::Matrix3d invK_r, K_r;
    K_r << rcalib(0), 0., rcalib(2),
            0., rcalib(1), rcalib(3),
            0., 0., 1.;
    invK_r = K_r.inverse();

    Eigen::Vector3d ranchpt = zanch_r * invK_r * ranchpx_;
//    Eigen::Vector3d wpt = Twanch * anchpt;
    Eigen::Vector3d wpt = Twanch * ranchpt;

    // Compute left/right reproj err
    Eigen::Vector2d pred;

//    Eigen::Vector3d lcampt = Tcw * wpt;
    Eigen::Vector3d rcampt = Tcw * wpt;

//    const double linvz = 1. / lcampt.z();
    const double rinvz = 1. / rcampt.z();

    pred << rcalib(0) * rcampt.x() * rinvz + rcalib(2),
            rcalib(1) * rcampt.y() * rinvz + rcalib(3);

    Eigen::Map<Eigen::Vector2d> werr(residuals);
    werr = sqrt_info_ * (pred - runpx_);

    // Update chi2err and depthpos info for
    // post optim checking
    chi2err_ = 0.;
    for( int i = 0 ; i < 2 ; i++ )
        chi2err_ += std::pow(residuals[i],2);


    isdepthpositive_ = true;
    if( rcampt.z() <= 0 )
        isdepthpositive_ = false;

    if(jacobians != NULL)
    {
        const double rinvz2 = rinvz * rinvz;

//        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_lcam;
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_rcam;
        J_rcam << rinvz * rcalib(0), 0., -rcampt.x() * rinvz2 * rcalib(0),
                0., rinvz * rcalib(1), -rcampt.y() * rinvz2 * rcalib(1);

//        Eigen::Matrix<double,2,3> J_lRcw;
        Eigen::Matrix<double,2,3> J_rRcw;

        if( jacobians[1] != NULL || jacobians[2] != NULL
            || jacobians[3] != NULL )
        {
//            Eigen::Matrix3d Rcw = Tcw.rotationMatrix();
//            J_lRcw.noalias() = J_lcam * Rcw;
            Eigen::Matrix3d Rcw = Tcw.rotationMatrix();
            J_rRcw.noalias() = J_rcam * Rcw;
        }

        if(jacobians[0] != NULL)
        {
//            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_lcalib(jacobians[0]);
//            J_lcalib.setZero();
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_rcalib(jacobians[0]);
            J_rcalib.setZero();

            // TODO
        }
        if(jacobians[1] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3anch(jacobians[1]);// 锚定关键帧的位姿 右目到世界
            J_se3anch.setZero();

            Eigen::Matrix3d skew_wpt = Sophus::SO3d::hat(wpt);

            J_se3anch.block<2,3>(0,0) = J_rRcw;
            J_se3anch.block<2,3>(0,3).noalias() = -1. * J_rRcw * skew_wpt;

            J_se3anch = sqrt_info_ * J_se3anch.eval();
        }
        if(jacobians[2] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3pose(jacobians[2]);// 当前关键帧的位姿 右目到世界
            J_se3pose.setZero();

            Eigen::Matrix3d skew_wpt = Sophus::SO3d::hat(wpt);

            J_se3pose.block<2,3>(0,0) = -1. * J_rRcw;
            J_se3pose.block<2,3>(0,3).noalias() = J_rRcw * skew_wpt;

            J_se3pose = sqrt_info_ * J_se3pose.eval();
        }
        if(jacobians[3] != NULL)
        {
            Eigen::Map<Eigen::Vector2d> J_invpt(jacobians[3]);//地图点的逆深度参数
            Eigen::Vector3d J_lambda = -1. * zanch_r * Twanch.rotationMatrix() * ranchpt;
            J_invpt.noalias() = J_rRcw * J_lambda;

            J_invpt = sqrt_info_ * J_invpt.eval();
        }
    }

    return true;
}


bool ReprojectionErrorRightAnchCamKSE3AnchInvDepth::Evaluate(double const* const* parameters,
                double* residuals, double** jacobians) const
{
    // [fx, fy, cx, cy]
    Eigen::Map<const Eigen::Vector4d> lcalib(parameters[0]);
    Eigen::Map<const Eigen::Vector4d> rcalib(parameters[1]);

    // [tx, ty, tz, qw, qx, qy, qz]
    Eigen::Map<const Eigen::Vector3d> trl(parameters[2]);
    Eigen::Map<const Eigen::Quaterniond> qrl(parameters[2]+3);

    // [1/z_anch]
    const double zanch = 1. / parameters[3][0];

    // Sophus::SE3d Twanch(qwanch,twanch);
    Sophus::SE3d Trl(qrl,trl);

    Eigen::Matrix3d invK, K;
    K << lcalib(0), 0., lcalib(2),
        0., lcalib(1), lcalib(3),
        0., 0., 1.;
    invK = K.inverse();

    Eigen::Vector3d anchpt = zanch * invK * anchpx_;
    // Eigen::Vector3d wpt = Twanch * anchpt;

    // Compute left/right reproj err
    Eigen::Vector2d pred;

    Eigen::Vector3d rcampt = Trl * anchpt;

    const double rinvz = 1. / rcampt.z();

    pred << rcalib(0) * rcampt.x() * rinvz + rcalib(2),
            rcalib(1) * rcampt.y() * rinvz + rcalib(3);
    
    Eigen::Map<Eigen::Vector2d> werr(residuals);
    werr = sqrt_info_ * (pred - runpx_);

    // Update chi2err and depthpos info for 
    // post optim checking
    chi2err_ = 0.;
    for( int i = 0 ; i < 2 ; i++ ) 
        chi2err_ += std::pow(residuals[i],2);

    isdepthpositive_ = true;    
    if( rcampt.z() <= 0 )
        isdepthpositive_ = false;

    if(jacobians != NULL)
    {
        const double rinvz2 = rinvz * rinvz;

        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_rcam;

        J_rcam << rinvz * rcalib(0), 0., -rcampt.x() * rinvz2 * rcalib(0),
                0., rinvz * rcalib(1), -rcampt.y() * rinvz2 * rcalib(1);

        Eigen::Matrix<double,2,3> J_rRcw; 

        if( jacobians[3] != NULL )
        {
            J_rRcw.noalias() = J_rcam * Trl.rotationMatrix();
        }

        if(jacobians[0] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_lcalib(jacobians[0]);
            J_lcalib.setZero();
            
            // TODO 
        }
        if(jacobians[1] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_rcalib(jacobians[1]);
            J_rcalib.setZero();
            J_rcalib(0,0) = rcampt.x() * rinvz;
            J_rcalib(0,2) = 1.;
            J_rcalib(1,1) = rcampt.y() * rinvz;
            J_rcalib(1,3) = 1.;

            J_rcalib = sqrt_info_ * J_rcalib.eval();   
        }
        if(jacobians[2] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3extrin(jacobians[2]);
            J_se3extrin.setZero();
            
            // TODO
        }
        if(jacobians[3] != NULL)
        {
            Eigen::Map<Eigen::Vector2d> J_invpt(jacobians[3]);
            Eigen::Vector3d J_lambda = -1. * zanch * anchpt;
            J_invpt.noalias() = J_rRcw * J_lambda;

            J_invpt = sqrt_info_ * J_invpt.eval();
        }
    }

    return true;
}

bool ReprojectionErrorRightAnchCamKSE3AnchInvDepth_r::Evaluate(double const* const* parameters,
                                                             double* residuals, double** jacobians) const
{
    // [fx, fy, cx, cy]
    Eigen::Map<const Eigen::Vector4d> lcalib(parameters[0]);
    Eigen::Map<const Eigen::Vector4d> rcalib(parameters[1]);

    // [tx, ty, tz, qw, qx, qy, qz]
    Eigen::Map<const Eigen::Vector3d> trl(parameters[2]);
    Eigen::Map<const Eigen::Quaterniond> qrl(parameters[2]+3);

    // [1/z_anch]
    const double zanch_r = 1. / parameters[3][0];

    // Sophus::SE3d Twanch(qwanch,twanch);
    Sophus::SE3d Trl(qrl,trl);
    Sophus::SE3d Tlr = Trl.inverse();

    Eigen::Matrix3d invK, K;
    K << lcalib(0), 0., lcalib(2),
            0., lcalib(1), lcalib(3),
            0., 0., 1.;
    invK = K.inverse();
    Eigen::Matrix3d invK_r, K_r;
    K_r << rcalib(0), 0., rcalib(2),
            0., rcalib(1), rcalib(3),
            0., 0., 1.;
    invK_r = K_r.inverse();

//    Eigen::Vector3d anchpt = zanch * invK * anchpx_;
    Eigen::Vector3d ranchpt = zanch_r * invK_r * ranchpx_;
    // Eigen::Vector3d wpt = Twanch * anchpt;

    // Compute left/right reproj err
    Eigen::Vector2d pred;

//    Eigen::Vector3d rcampt = Trl * anchpt;
    Eigen::Vector3d lcampt = Tlr * ranchpt;

//    const double rinvz = 1. / rcampt.z();
    const double linvz = 1. / lcampt.z();

    pred << lcalib(0) * lcampt.x() * linvz + lcalib(2),
            lcalib(1) * lcampt.y() * linvz + lcalib(3);

    Eigen::Map<Eigen::Vector2d> werr(residuals);
    werr = sqrt_info_ * (pred - lunpx_);

    // Update chi2err and depthpos info for
    // post optim checking
    chi2err_ = 0.;
    for( int i = 0 ; i < 2 ; i++ )
        chi2err_ += std::pow(residuals[i],2);

    isdepthpositive_ = true;
    if( lcampt.z() <= 0 )
        isdepthpositive_ = false;

    if(jacobians != NULL)
    {
        const double linvz2 = linvz * linvz;

//        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_rcam;
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_lcam;

        J_lcam << linvz * lcalib(0), 0., -lcampt.x() * linvz2 * lcalib(0),
                0., linvz * lcalib(1), -lcampt.y() * linvz2 * lcalib(1);

//        Eigen::Matrix<double,2,3> J_rRcw;
        Eigen::Matrix<double,2,3> J_lRcw;

        if( jacobians[3] != NULL )
        {
//            J_rRcw.noalias() = J_rcam * Trl.rotationMatrix();
            J_lRcw.noalias() = J_lcam * Tlr.rotationMatrix();//
        }

        if(jacobians[1] != NULL)
        {
//            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_lcalib(jacobians[0]);
//            J_lcalib.setZero();
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_rcalib(jacobians[1]);
            J_rcalib.setZero();

            // TODO
        }
        if(jacobians[0] != NULL)
        {
//            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_rcalib(jacobians[1]);
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_lcalib(jacobians[0]);
            J_lcalib.setZero();
            J_lcalib(0,0) = lcampt.x() * linvz;
            J_lcalib(0,2) = 1.;
            J_lcalib(1,1) = lcampt.y() * linvz;
            J_lcalib(1,3) = 1.;

            J_lcalib = sqrt_info_ * J_lcalib.eval();
        }
        if(jacobians[2] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3extrin(jacobians[2]);// todo 用不用改，转置之类的
            J_se3extrin.setZero();

            // TODO
        }
        if(jacobians[3] != NULL)
        {
//            Eigen::Map<Eigen::Vector2d> J_invpt(jacobians[3]);
//            Eigen::Vector3d J_lambda = -1. * zanch * anchpt;
//            J_invpt.noalias() = J_rRcw * J_lambda;
            Eigen::Map<Eigen::Vector2d> J_invpt(jacobians[3]);
            Eigen::Vector3d J_lambda = -1. * zanch_r * ranchpt;//与逆深度相关的梯度（即影响重投影误差的梯度）
            J_invpt.noalias() = J_lRcw * J_lambda;//逆深度参数的雅可比矩阵 不知道对不对

            J_invpt = sqrt_info_ * J_invpt.eval();
        }
    }

    return true;
}

bool ReprojectionErrorRightCamKSE3AnchInvDepth::Evaluate(double const* const* parameters,
                double* residuals, double** jacobians) const
{
    // [fx, fy, cx, cy]
    Eigen::Map<const Eigen::Vector4d> lcalib(parameters[0]);
    Eigen::Map<const Eigen::Vector4d> rcalib(parameters[1]);

    // [tx, ty, tz, qw, qx, qy, qz]
    Eigen::Map<const Eigen::Vector3d> twanch(parameters[2]);
    Eigen::Map<const Eigen::Quaterniond> qwanch(parameters[2]+3);

    Eigen::Map<const Eigen::Vector3d> twc(parameters[3]);
    Eigen::Map<const Eigen::Quaterniond> qwc(parameters[3]+3);

    Eigen::Map<const Eigen::Vector3d> trl(parameters[4]);
    Eigen::Map<const Eigen::Quaterniond> qrl(parameters[4]+3);

    // [1/z_anch]
    const double zanch = 1. / parameters[5][0];

    Sophus::SE3d Twanch(qwanch,twanch);
    Sophus::SE3d Twc(qwc,twc);
    Sophus::SE3d Tcw = Twc.inverse();
    Sophus::SE3d Trl(qrl,trl);

    Eigen::Matrix3d invK, K;
    K << lcalib(0), 0., lcalib(2),
        0., lcalib(1), lcalib(3),
        0., 0., 1.;
    invK = K.inverse();

    Eigen::Vector3d anchpt = zanch * invK * anchpx_;
    Eigen::Vector3d wpt = Twanch * anchpt;

    // Compute left/right reproj err
    Eigen::Vector2d pred;

    Eigen::Vector3d lcampt = Tcw * wpt;
    Eigen::Vector3d rcampt = Trl * lcampt;

    const double rinvz = 1. / rcampt.z();

    pred << rcalib(0) * rcampt.x() * rinvz + rcalib(2),
            rcalib(1) * rcampt.y() * rinvz + rcalib(3);

    Eigen::Map<Eigen::Vector2d> werr(residuals);
    werr = sqrt_info_ * (pred - runpx_);

    // Update chi2err and depthpos info for 
    // post optim checking
    chi2err_ = 0.;
    for( int i = 0 ; i < 2 ; i++ ) 
        chi2err_ += std::pow(residuals[i],2);

    isdepthpositive_ = true;    
    if( rcampt.z() <= 0 )
        isdepthpositive_ = false;

    if(jacobians != NULL)
    {
        const double rinvz2 = rinvz * rinvz;

        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_rcam;

        J_rcam << rinvz * rcalib(0), 0., -rcampt.x() * rinvz2 * rcalib(0),
                0., rinvz * rcalib(1), -rcampt.y() * rinvz2 * rcalib(1);

        Eigen::Matrix<double,2,3> J_rRcw; 
        Eigen::Matrix3d skew_wpt;

        if( jacobians[2] != NULL || jacobians[3] != NULL 
            || jacobians[5] != NULL )
        {
            J_rRcw.noalias() = J_rcam * Trl.rotationMatrix() * Tcw.rotationMatrix();
            skew_wpt = Sophus::SO3d::hat(wpt);
        }

        if(jacobians[0] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_lcalib(jacobians[0]);
            J_lcalib.setZero();

            // TODO       
        }
        if(jacobians[1] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_rcalib(jacobians[1]);
            J_rcalib.setZero();
            J_rcalib(0,0) = rcampt.x() * rinvz;
            J_rcalib(0,2) = 1.;
            J_rcalib(1,1) = rcampt.y() * rinvz;
            J_rcalib(1,3) = 1.;

            J_rcalib = sqrt_info_ * J_rcalib.eval();   
        }
        if(jacobians[2] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3anch(jacobians[2]);
            J_se3anch.setZero();

            J_se3anch.block<2,3>(0,0) = J_rRcw;
            J_se3anch.block<2,3>(0,3).noalias() = -1. * J_rRcw * skew_wpt;

            J_se3anch = sqrt_info_ * J_se3anch.eval();
        }
        if(jacobians[3] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3pose(jacobians[3]);
            J_se3pose.setZero();

            J_se3pose.block<2,3>(0,0) = -1. * J_rRcw;
            J_se3pose.block<2,3>(0,3).noalias() = J_rRcw * skew_wpt;

            J_se3pose = sqrt_info_ * J_se3pose.eval();
        }
        if(jacobians[4] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3extrin(jacobians[4]);
            J_se3extrin.setZero();

            // TODO
        }
        if(jacobians[5] != NULL)
        {
            Eigen::Map<Eigen::Vector2d> J_invpt(jacobians[5]);
            Eigen::Vector3d J_lambda = -1. * zanch *  Twanch.rotationMatrix() * anchpt;
            J_invpt.block<2,1>(0,0).noalias() = J_rRcw * J_lambda;

            J_invpt = sqrt_info_ * J_invpt.eval();
        }
    }

    return true;
}

bool ReprojectionErrorRightCamKSE3AnchInvDepth_r::Evaluate(double const* const* parameters,
                                                         double* residuals, double** jacobians) const
{
    // [fx, fy, cx, cy]
    Eigen::Map<const Eigen::Vector4d> lcalib(parameters[0]);
    Eigen::Map<const Eigen::Vector4d> rcalib(parameters[1]);

    // [tx, ty, tz, qw, qx, qy, qz]
    Eigen::Map<const Eigen::Vector3d> twanch(parameters[2]);//锚定关键帧的位姿 左目到世界
    Eigen::Map<const Eigen::Quaterniond> qwanch(parameters[2]+3);

    Eigen::Map<const Eigen::Vector3d> twc(parameters[3]);//当前关键帧的位姿 左目到世界
    Eigen::Map<const Eigen::Quaterniond> qwc(parameters[3]+3);

    Eigen::Map<const Eigen::Vector3d> trl(parameters[4]);
    Eigen::Map<const Eigen::Quaterniond> qrl(parameters[4]+3);

    // [1/z_anch]
    const double zanch_r = 1. / parameters[5][0];

    Sophus::SE3d Trl(qrl,trl);
    Sophus::SE3d Tlr = Trl.inverse();
    Sophus::SE3d Twanch(qwanch,twanch);//锚定关键帧的位姿 左目到世界
    Twanch = Twanch*Tlr;//转换成右目位姿（左目到世界*右到左）
    Sophus::SE3d Twc(qwc,twc);
    Twc = Twc*Tlr;
    Sophus::SE3d Tcw = Twc.inverse();

    Eigen::Matrix3d invK_r, K_r;
    K_r << rcalib(0), 0., rcalib(2),
            0., rcalib(1), rcalib(3),
            0., 0., 1.;
    invK_r = K_r.inverse();

    Eigen::Vector3d ranchpt = zanch_r * invK_r * ranchpx_;
    Eigen::Vector3d wpt = Twanch * ranchpt;

    // Compute left/right reproj err
    Eigen::Vector2d pred;

//    Eigen::Vector3d lcampt = Tcw * wpt;
//    Eigen::Vector3d rcampt = Trl * lcampt;
    Eigen::Vector3d rcampt = Tcw * wpt;
    Eigen::Vector3d lcampt = Tlr * rcampt;

//    const double rinvz = 1. / rcampt.z();
    const double linvz = 1. / lcampt.z();

    pred << lcalib(0) * lcampt.x() * linvz + lcalib(2),
            lcalib(1) * lcampt.y() * linvz + lcalib(3);

    Eigen::Map<Eigen::Vector2d> werr(residuals);
    werr = sqrt_info_ * (pred - unpx_);

    // Update chi2err and depthpos info for
    // post optim checking
    chi2err_ = 0.;
    for( int i = 0 ; i < 2 ; i++ )
        chi2err_ += std::pow(residuals[i],2);

    isdepthpositive_ = true;
    if( rcampt.z() <= 0 )
        isdepthpositive_ = false;

    if(jacobians != NULL)
    {
        const double linvz2 = linvz * linvz;

//        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_rcam;
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_lcam;

        J_lcam << linvz * lcalib(0), 0., -lcampt.x() * linvz2 * lcalib(0),
                0., linvz * lcalib(1), -lcampt.y() * linvz2 * lcalib(1);

//        Eigen::Matrix<double,2,3> J_rRcw;
        Eigen::Matrix<double,2,3> J_lRcw;
        Eigen::Matrix3d skew_wpt;

        if( jacobians[2] != NULL || jacobians[3] != NULL
            || jacobians[5] != NULL )
        {
//            J_rRcw.noalias() = J_rcam * Trl.rotationMatrix() * Tcw.rotationMatrix();
            J_lRcw.noalias() = J_lcam * Tlr.rotationMatrix() * Tcw.rotationMatrix();
            skew_wpt = Sophus::SO3d::hat(wpt);
        }

        if(jacobians[1] != NULL)
        {
//            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_lcalib(jacobians[0]);
//            J_lcalib.setZero();
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_rcalib(jacobians[1]);
            J_rcalib.setZero();

            // TODO
        }
        if(jacobians[0] != NULL)
        {
//            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_rcalib(jacobians[1]);
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_lcalib(jacobians[0]);
            J_lcalib.setZero();
            J_lcalib(0,0) = lcampt.x() * linvz;
            J_lcalib(0,2) = 1.;
            J_lcalib(1,1) = lcampt.y() * linvz;
            J_lcalib(1,3) = 1.;

            J_lcalib = sqrt_info_ * J_lcalib.eval();
        }
        if(jacobians[2] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3anch(jacobians[2]);//锚定关键帧的位姿 右目到世界
            J_se3anch.setZero();

            J_se3anch.block<2,3>(0,0) = J_lRcw;
            J_se3anch.block<2,3>(0,3).noalias() = -1. * J_lRcw * skew_wpt;

            J_se3anch = sqrt_info_ * J_se3anch.eval();
        }
        if(jacobians[3] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3pose(jacobians[3]);//当前关键帧的位姿 右目到世界
            J_se3pose.setZero();

            J_se3pose.block<2,3>(0,0) = -1. * J_lRcw;
            J_se3pose.block<2,3>(0,3).noalias() = J_lRcw * skew_wpt;

            J_se3pose = sqrt_info_ * J_se3pose.eval();
        }
        if(jacobians[4] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3extrin(jacobians[4]);
            J_se3extrin.setZero();

            // TODO
        }
        if(jacobians[5] != NULL)
        {
            Eigen::Map<Eigen::Vector2d> J_invpt(jacobians[5]);
            Eigen::Vector3d J_lambda = -1. * zanch_r *  Twanch.rotationMatrix() * ranchpt;
            J_invpt.block<2,1>(0,0).noalias() = J_lRcw * J_lambda;

            J_invpt = sqrt_info_ * J_invpt.eval();
        }
    }

    return true;
}

}