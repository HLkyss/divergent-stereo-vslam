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


#include <iostream>
#include <string>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include <sophus/se3.hpp>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "profiler.hpp"

class SlamParams {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//    EIGEN_DONT_VECTORIZE
//    EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT

    SlamParams() {}
    
    SlamParams(const cv::FileStorage &fsSettings);

    void reset();

    //=====================================================
    // Variables relative to the current state of the SLAM
    //=====================================================

    bool blocalba_is_on_ = false;
    bool blc_is_on_ = false;
    bool bvision_init_ = false;
    bool breset_req_ = false;
    bool bforce_realtime_ = false;

    //=====================================================
    // Variables relative to the setup used for the SLAM
    //=====================================================

    // Calibration parameters (TODO: Get Ready to store all of these in a vector to handle N camera)
    std::string cam_left_topic_, cam_right_topic_;
    std::string cam_left_model_, cam_right_model_;

    double fxl_, fyl_, cxl_, cyl_;
    double k1l_, k2l_, p1l_, p2l_;

    double fxr_, fyr_, cxr_, cyr_;
    double k1r_, k2r_, p1r_, p2r_;

    double img_left_w_, img_left_h_;
    double img_right_w_, img_right_h_;
    //单双目区域图像大小
    double img_leftm_w_, img_leftm_h_;
    double img_rightm_w_, img_rightm_h_;
    double img_lefts_w_, img_lefts_h_;
    double img_rights_w_, img_rights_h_;
    double img_w_crop_;

    Eigen::Matrix3d R_sl, R_sr, R_ml, R_mr;//左双目区虚拟相机-左相机；右双目区虚拟相机-右相机；左单目区虚拟相机-左相机；右单目区虚拟相机-右相机
    bool use_cuda_ = false;

    // Extrinsic parameters
    Sophus::SE3d T_left_right_;     //右相机到左相机
    Sophus::SE3d T_right_left_;     //add:左单目区虚拟相机到左相机
    //虚拟相机外参
    Sophus::SE3d T_left_lefts_;     //add:左双目区虚拟相机到左相机
    Sophus::SE3d T_left_leftm_;     //add:左单目区虚拟相机到左相机
    Sophus::SE3d T_right_rights_;   //add:右双目区虚拟相机到右相机
    Sophus::SE3d T_right_rightm_;   //add:右单目区虚拟相机到右相机

    // SLAM settings
    bool debug_, log_timings_;

    bool mono_, stereo_;
    bool mono_stereo_;

    double fov;
    double cos_half_fov;
    double theta;
    double theta_m;//单目区域偏转角绝对值（虚拟相机单目区方向与原相机朝向的夹角）
    double theta_s;//双目区域偏转角绝对值（虚拟相机双目区方向与原相机朝向的夹角）
    double angle_m;//单目区视野
    double angle_s;//双目区视野

    bool slam_mode_;

    bool buse_loop_closer_;
    int lckfid_ = -1;

    float finit_parallax_;
    
    bool bdo_stereo_rect_;
    double alpha_;

    bool bdo_undist_;

    // Keypoints Extraction
    bool use_fast_, use_shi_tomasi_, use_brief_;
    bool use_singlescale_detector_;
    
    int nfast_th_;
    int nbmaxkps_, nmaxdist_;
    double dmaxquality_;
    int nbmaxkps_m_, nbmaxkps_s_;

    // Image Processing
    bool use_clahe_;
    float fclahe_val_;

    // KLT Parameters
    bool do_klt_, klt_use_prior_;
    bool btrack_keyframetoframe_;
    int nklt_win_size_, nklt_pyr_lvl_;
    cv::Size klt_win_size_;

    float fmax_fbklt_dist_;
    int nmax_iter_;
    float fmax_px_precision_;

    int nklt_err_;

    // Matching th.
    bool bdo_track_localmap_;
    
    float fmax_desc_dist_;
    float fmax_proj_pxdist_;

    // Error thresholds
    bool doepipolar_;
    bool dop3p_;
    bool bdo_random; // RANDOMIZE RANSAC?
    float fransac_err_;
    int nransac_iter_;
    float fepi_th_;

    float fmax_reproj_err_;
    bool buse_inv_depth_;

    // Bundle Adjustment Parameters
    // (mostly related to Ceres options)
    float robust_mono_th_;
    float robust_stereo_th_;

    bool use_sparse_schur_; // If False, Dense Schur used
    bool use_dogleg_; // If False, Lev.-Marq. used
    bool use_subspace_dogleg_; // If False, Powell's trad. Dogleg used
    bool use_nonmonotic_step_;

    // Estimator parameters
    bool apply_l2_after_robust_; // If true, a L2 optim is applied to refine the results from robust cost function

    int nmin_covscore_; // Number of common observations req. for opt. a KF in localBA

    // Map Filtering parameters
    float fkf_filtering_ratio_;

    // Final BA
    bool do_full_ba_;
};
