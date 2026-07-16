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

#include "slam_params.hpp"

SlamParams::SlamParams(const cv::FileStorage &fsSettings) {

    std::cout << "\nSLAM Parameters are being setup...\n";

    // READ THE SETTINGS
    debug_ = static_cast<int>(fsSettings["debug"]);;
    log_timings_ = static_cast<int>(fsSettings["log_timings"]);;

    mono_ =  static_cast<int>(fsSettings["mono"]);
    stereo_ = static_cast<int>(fsSettings["stereo"]);
    mono_stereo_ = static_cast<int>(fsSettings["mono_stereo"]);

    bforce_realtime_ = static_cast<int>(fsSettings["force_realtime"]);

    slam_mode_ = static_cast<int>(fsSettings["slam_mode"]);

    buse_loop_closer_ = static_cast<int>(fsSettings["buse_loop_closer"]);

    cam_left_topic_.assign(fsSettings["Camera.topic_left"]);
    cam_left_model_.assign(fsSettings["Camera.model_left"]);
    img_left_w_ = fsSettings["Camera.left_nwidth"];
    img_left_h_ = fsSettings["Camera.left_nheight"];

    fxl_ = fsSettings["Camera.fxl"];
    fyl_ = fsSettings["Camera.fyl"];
    cxl_ = fsSettings["Camera.cxl"];
    cyl_ = fsSettings["Camera.cyl"];

    k1l_ = fsSettings["Camera.k1l"];
    k2l_ = fsSettings["Camera.k2l"];
    p1l_ = fsSettings["Camera.p1l"];
    p2l_ = fsSettings["Camera.p2l"];

    use_cuda_ = static_cast<int>(fsSettings["cuda"]);

    if( stereo_ || mono_stereo_ ) {
        cam_right_topic_.assign(fsSettings["Camera.topic_right"]);
        cam_right_model_.assign(fsSettings["Camera.model_right"]);

        img_right_w_ = fsSettings["Camera.right_nwidth"];
        img_right_h_ = fsSettings["Camera.right_nheight"];

        fxr_ = fsSettings["Camera.fxr"];
        fyr_ = fsSettings["Camera.fyr"];
        cxr_ = fsSettings["Camera.cxr"];
        cyr_ = fsSettings["Camera.cyr"];

        k1r_ = fsSettings["Camera.k1r"];
        k2r_ = fsSettings["Camera.k2r"];
        p1r_ = fsSettings["Camera.p1r"];
        p2r_ = fsSettings["Camera.p2r"];

        cv::Mat cvTbc0, cvTbc1;
        Eigen::Matrix4d Tbc0, Tbc1;

        fsSettings["body_T_cam0"] >> cvTbc0;//y轴朝下
        fsSettings["body_T_cam1"] >> cvTbc1;

        cv::cv2eigen(cvTbc0,Tbc0);
        cv::cv2eigen(cvTbc1,Tbc1);

        T_left_right_ = Sophus::SE3d(Tbc0.inverse() * Tbc1);//从右相机到左相机的变换  c1到c0=body到c0*c1到body
        T_right_left_ = Sophus::SE3d(Tbc1.inverse() * Tbc0);//从左相机到右相机的变换  c0到c1=body到c1*c0到body
    }
    if(mono_stereo_){
        fov = fsSettings["fov"];
        double fov_rad = fov * CV_PI / 180.0;
        cos_half_fov = cos(fov_rad/2.);
        theta = fsSettings["theta"];
        theta_m  = fov/2 - theta;
        theta_s = theta;
        double theta_m_rad = theta_m * CV_PI / 180.0;
        double theta_s_rad = theta_s * CV_PI / 180.0;
        angle_s = 2 * (fov/2-theta);
        angle_m = fov-angle_s;
        double angle_m_rad = angle_m * CV_PI / 180.0;
        double angle_s_rad = angle_s * CV_PI / 180.0;
        img_leftm_w_ = int(img_left_w_*tan(angle_m_rad/2)/tan(fov_rad/2));
        img_leftm_h_ = img_left_h_;
        img_rightm_w_ = int(img_right_w_*tan(angle_m_rad/2)/tan(fov_rad/2));
        img_rightm_h_ = img_right_h_;
        img_lefts_w_ = int(img_left_w_*tan(angle_s_rad/2)/tan(fov_rad/2));
        img_lefts_h_ = img_left_h_;
        img_rights_w_ = int(img_right_w_*tan(angle_s_rad/2)/tan(fov_rad/2));
        img_rights_h_ = img_right_h_;
        img_w_crop_ = fxl_*(tan(fov_rad/2)-tan((fov_rad-angle_s_rad)/2));
        std::cout << "fov:" << fov << std::endl;
        std::cout << "theta:" << theta << std::endl;
        std::cout << "angle_m:" << angle_m << std::endl;
        std::cout << "angle_s:" << angle_s << std::endl;
        std::cout << "theta_s:" << theta_s << std::endl;
        std::cout << "theta_m:" << theta_m << std::endl;
        std::cout << "img_leftm_w_:" << img_leftm_w_ << std::endl;
        std::cout << "img_leftm_h_:" << img_leftm_h_ << std::endl;
        std::cout << "img_rightm_w_:" << img_rightm_w_ << std::endl;
        std::cout << "img_rightm_h_:" << img_rightm_h_ << std::endl;
        std::cout << "img_lefts_w_:" << img_lefts_w_ << std::endl;
        std::cout << "img_lefts_h_:" << img_lefts_h_ << std::endl;
        std::cout << "img_rights_w_:" << img_rights_w_ << std::endl;
        std::cout << "img_rights_h_:" << img_rights_h_ << std::endl;
        std::cout << "img_w_crop_:" << img_w_crop_ << std::endl;

//        R_sl << cos(-1*theta_s_rad), 0, sin(-1*theta_s_rad),//左双目区虚拟相机到左相机
//                0,          1, 0,
//                -sin(-1*theta_s_rad), 0, cos(-1*theta_s_rad);
//        R_sr << cos(theta_s_rad), 0, sin(theta_s_rad),//右双目区虚拟相机到右相机
//                0,          1, 0,
//                -sin(theta_s_rad), 0, cos(theta_s_rad);
//        R_ml << cos(theta_m_rad), 0, sin(theta_m_rad),
//                0,          1, 0,
//                -sin(theta_m_rad), 0, cos(theta_m_rad);
//        R_mr << cos(-1*theta_m_rad), 0, sin(-1*theta_m_rad),
//                0,          1, 0,
//                -sin(-1*theta_m_rad), 0, cos(-1*theta_m_rad);
        R_sl << cos(theta_s_rad), 0, sin(theta_s_rad),//左双目区点变换到左目（世界系下左双目向左转变成左目），绕y轴30度，相机y轴朝下
                0,          1, 0,
                -sin(theta_s_rad), 0, cos(theta_s_rad);
        R_sr << cos(-1*theta_s_rad), 0, sin(-1*theta_s_rad),//右双目区到右目
                0,          1, 0,
                -sin(-1*theta_s_rad), 0, cos(-1*theta_s_rad);
        R_ml << cos(-1*theta_m_rad), 0, sin(-1*theta_m_rad),
                0,          1, 0,
                -sin(-1*theta_m_rad), 0, cos(-1*theta_m_rad);
        R_mr << cos(theta_m_rad), 0, sin(theta_m_rad),
                0,          1, 0,
                -sin(theta_m_rad), 0, cos(theta_m_rad);
        T_left_leftm_=Sophus::SE3d(R_ml,Eigen::Vector3d(0,0,0));
        T_left_lefts_=Sophus::SE3d(R_sl,Eigen::Vector3d(0,0,0));
        T_right_rightm_=Sophus::SE3d(R_mr,Eigen::Vector3d(0,0,0));
        T_right_rights_=Sophus::SE3d(R_sr,Eigen::Vector3d(0,0,0));
        std::cout << "R_sl:" << R_sl << std::endl;
        std::cout << "R_sr:" << R_sr << std::endl;
        std::cout << "R_ml:" << R_ml << std::endl;
        std::cout << "R_mr:" << R_mr << std::endl;
        std::cout<<"T_left_leftm_:"<<T_left_leftm_.matrix()<<std::endl;
        std::cout<<"T_left_lefts_:"<<T_left_lefts_.matrix()<<std::endl;
        std::cout<<"T_right_rightm_:"<<T_right_rightm_.matrix()<<std::endl;
        std::cout<<"T_right_rights_:"<<T_right_rights_.matrix()<<std::endl;
    }

    finit_parallax_ = fsSettings["finit_parallax"];

    bdo_stereo_rect_ = static_cast<int>(fsSettings["bdo_stereo_rect"]);
    alpha_ = fsSettings["alpha"];

    bdo_undist_ = static_cast<int>(fsSettings["bdo_undist"]);
    
    bdo_random = static_cast<int>(fsSettings["bdo_random"]);

    use_shi_tomasi_ = static_cast<int>(fsSettings["use_shi_tomasi"]);
    use_fast_ = static_cast<int>(fsSettings["use_fast"]);
    use_brief_ = static_cast<int>(fsSettings["use_brief"]);
    use_singlescale_detector_ = static_cast<int>(fsSettings["use_singlescale_detector"]);

    nfast_th_ = fsSettings["nfast_th"];
    dmaxquality_ = fsSettings["dmaxquality"];

    nmaxdist_ = fsSettings["nmaxdist"];
    float nbwcells = ceil( (float)img_left_w_ / nmaxdist_ );
    float nbhcells = ceil( (float)img_left_h_ / nmaxdist_ );
    nbmaxkps_ = nbwcells * nbhcells;
    float nbwcells_m = ceil( (float)img_leftm_w_ / nmaxdist_ );
    float nbhcells_m = ceil( (float)img_leftm_h_ / nmaxdist_ );
    nbmaxkps_m_ = nbwcells_m * nbhcells_m;
    float nbwcells_s = ceil( (float)img_lefts_w_ / nmaxdist_ );
    float nbhcells_s = ceil( (float)img_lefts_h_ / nmaxdist_ );
    nbmaxkps_s_ = nbwcells_s * nbhcells_s;

    use_clahe_ = static_cast<int>(fsSettings["use_clahe"]);
    fclahe_val_ = fsSettings["fclahe_val"];

    do_klt_ = static_cast<int>(fsSettings["do_klt"]);
    klt_use_prior_ = static_cast<int>(fsSettings["klt_use_prior"]);

    btrack_keyframetoframe_ = static_cast<int>(fsSettings["btrack_keyframetoframe"]);
    
    nklt_win_size_ = fsSettings["nklt_win_size"];
    nklt_pyr_lvl_ = fsSettings["nklt_pyr_lvl"];

    klt_win_size_ = cv::Size(nklt_win_size_, nklt_win_size_);

    fmax_fbklt_dist_ = fsSettings["fmax_fbklt_dist"];
    nmax_iter_ = fsSettings["nmax_iter"];
    fmax_px_precision_ = fsSettings["fmax_px_precision"];

    
    nklt_err_ = fsSettings["nklt_err"];

    // Matching th.
    bdo_track_localmap_ = static_cast<int>(fsSettings["bdo_track_localmap"]);

    fmax_desc_dist_ = fsSettings["fmax_desc_dist"];
    fmax_proj_pxdist_ = fsSettings["fmax_proj_pxdist"];

    doepipolar_ = static_cast<int>(fsSettings["doepipolar"]);
    dop3p_ = static_cast<int>(fsSettings["dop3p"]);

    fransac_err_ = fsSettings["fransac_err"];
    fepi_th_ = fransac_err_;
    nransac_iter_ = fsSettings["nransac_iter"];

    fmax_reproj_err_ = fsSettings["fmax_reproj_err"];
    buse_inv_depth_ = static_cast<int>(fsSettings["buse_inv_depth"]);

    // Bundle Adjustment Parameters
    // (mostly related to Ceres options)
    robust_mono_th_ = fsSettings["robust_mono_th"];
    robust_stereo_th_ = fsSettings["robust_stereo_th"];

    use_sparse_schur_ = static_cast<int>(fsSettings["use_sparse_schur"]);
    use_dogleg_ = static_cast<int>(fsSettings["use_dogleg"]);
    use_subspace_dogleg_ = static_cast<int>(fsSettings["use_subspace_dogleg"]);
    use_nonmonotic_step_ = static_cast<int>(fsSettings["use_nonmonotic_step"]);

    apply_l2_after_robust_ = static_cast<int>(fsSettings["apply_l2_after_robust"]);

    nmin_covscore_ = fsSettings["nmin_covscore"];

    // Map Filtering parameters
    fkf_filtering_ratio_ = fsSettings["fkf_filtering_ratio"]; 

    // Apply Full BA?
    do_full_ba_ = static_cast<int>(fsSettings["do_full_ba"]);
}

void SlamParams::reset() {
    blocalba_is_on_ = false;
    blc_is_on_ = false;
    bvision_init_ = false;
    breset_req_ = false;
}