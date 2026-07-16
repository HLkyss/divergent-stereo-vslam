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


#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "slam_params.hpp"
#include "map_manager.hpp"
#include "feature_tracker.hpp"

class MotionModel {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void applyMotionModel(Sophus::SE3d &Twc, double time) {
        if( prev_time_ > 0 ) 
        {
            // Provided Twc and prevTwc should be equal here
            // as prevTwc is updated right after pose computation
            if( !(Twc * prevTwc_.inverse()).log().isZero(1.e-5) )
            {
                // Might happen in case of LC!
                // So update prevPose to stay consistent
                prevTwc_ = Twc;
            }

            double dt = (time - prev_time_);
            Twc = Twc * Sophus::SE3d::exp(log_relT_ * dt);
        }
    }

    void updateMotionModel(const Sophus::SE3d &Twc, double time) {
        if( prev_time_ < 0. ) {
            prev_time_ = time;
            prevTwc_ = Twc;
        } else {
            double dt = time - prev_time_;

            prev_time_ = time;
            
            if( dt < 0. ) {
                std::cerr << "\nGot image older than previous image! LEAVING!\n";
                exit(-1);
            }

            Sophus::SE3d Tprevcur = prevTwc_.inverse() * Twc;
            log_relT_ = Tprevcur.log() / dt;// todo
//            std::cout<<"Twc = "<<Twc.matrix()<<std::endl;// 有问题：单位矩阵
//            std::cout<<"prevTwc_ = "<<prevTwc_.matrix()<<std::endl;// 有问题：单位矩阵
//            std::cout<<"dt = "<<dt<<std::endl;// 有问题：0
//            std::cout<<"update log_relT_= ="<<log_relT_<<std::endl;// 有问题：-nan
            prevTwc_ = Twc;
        }
    }

    void reset() {
        prev_time_ = -1.;
        log_relT_ = Eigen::Matrix<double, 6, 1>::Zero();
    }
    

    double prev_time_ = -1.;

    Sophus::SE3d prevTwc_;
    Eigen::Matrix<double, 6, 1> log_relT_ = Eigen::Matrix<double, 6, 1>::Zero();
};

class VisualFrontEnd {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VisualFrontEnd() {}
    VisualFrontEnd(std::shared_ptr<SlamParams> pstate, std::shared_ptr<Frame> pframe,
        std::shared_ptr<MapManager> pmap, std::shared_ptr<FeatureTracker> ptracker);
    VisualFrontEnd(std::shared_ptr<SlamParams> pstate, std::shared_ptr<Frame> pframe, std::shared_ptr<Frame> pframe_l, std::shared_ptr<Frame> pframe_r, std::shared_ptr<Frame> pframe_lm, std::shared_ptr<Frame> pframe_ls, std::shared_ptr<Frame> pframe_rm, std::shared_ptr<Frame> pframe_rs,
                   std::shared_ptr<MapManager> pmap, std::shared_ptr<MapManager> pmap_l, std::shared_ptr<MapManager> pmap_r, std::shared_ptr<MapManager> pmap_lm, std::shared_ptr<MapManager> pmap_ls, std::shared_ptr<MapManager> pmap_rm, std::shared_ptr<MapManager> pmap_rs, std::shared_ptr<MapManager> pmap_l_to_r, std::shared_ptr<MapManager> pmap_r_to_l, std::shared_ptr<FeatureTracker> ptracker);//mono_stereo

    bool visualTracking(cv::Mat &iml, double time);
    bool visualTracking(cv::Mat &iml, cv::Mat &imr, cv::Mat &imlm, cv::Mat &imls, cv::Mat &imrm, cv::Mat &imrs, double time);
    bool visualTracking(cv::Mat &iml, cv::Mat &imr, cv::Mat &imlm, cv::Mat &imls, cv::Mat &imrm, cv::Mat &imrs, double time, bool isleft);

    bool trackMono(cv::Mat &im, double time);
    bool trackMonoStereo(cv::Mat &iml, cv::Mat &imr, cv::Mat &imlm, cv::Mat &imls, cv::Mat &imrm, cv::Mat &imrs, double time);
    bool trackMonoStereo(cv::Mat &iml, cv::Mat &imr, cv::Mat &imlm, cv::Mat &imls, cv::Mat &imrm, cv::Mat &imrs, double time, bool isleft);

    void switchMapKeypoint(std::shared_ptr<Frame>& pframe_l,  std::shared_ptr<Frame>& pframe_r, std::shared_ptr<MapManager>& pmap_l, std::shared_ptr<MapManager>& pmap_r, std::shared_ptr<MapManager>& pmap_l_to_r, std::shared_ptr<MapManager>& pmap_r_to_l, cv::Mat &iml, cv::Mat &imr);
    void switchMapKeypoint_cuda(std::shared_ptr<Frame>& pframe_l,  std::shared_ptr<Frame>& pframe_r, std::shared_ptr<MapManager>& pmap_l, std::shared_ptr<MapManager>& pmap_r, std::shared_ptr<MapManager>& pmap_l_to_r, std::shared_ptr<MapManager>& pmap_r_to_l, cv::Mat &iml, cv::Mat &imr);

    bool trackStereo(cv::Mat &iml, cv::Mat &imr, cv::Mat &imlm, cv::Mat &imls, cv::Mat &imrm, cv::Mat &imrs, double time);

    void preprocessImage(cv::Mat &img_raw);
    void preprocessImage(cv::Mat &iml_raw, cv::Mat &imr_raw, cv::Mat &imlm_raw, cv::Mat &imls_raw, cv::Mat &imrm_raw, cv::Mat &imrs_raw);

    void kltTracking();
    void kltTracking(std::shared_ptr<Frame>& pframe, const std::vector<cv::Mat> &prevpyr, const std::vector<cv::Mat> &curpyr, std::shared_ptr<MapManager>& pmap, bool isleft);//mono_stereo
    void kltTrackingFromKF();

    void epipolar2d2dFiltering();
    void epipolar2d2dFiltering(std::shared_ptr<Frame>& pframe, std::shared_ptr<MapManager>& pmap, bool isleft);//mono_stereo

    void computePose();
    void computePose(std::shared_ptr<Frame>& pframe, std::shared_ptr<MapManager>& pmap, const std::vector<cv::Mat> &curpyl, const std::vector<cv::Mat> &curpyr, bool isleft);

    float computeParallax(const int kfid, bool do_unrot=true, bool bmedian=true, bool b2donly=false);
    float computeParallax(const int kfid, std::shared_ptr<Frame>& pframe, std::shared_ptr<MapManager>& pmap, bool isleft, bool do_unrot=true, bool bmedian=true, bool b2donly=false);

    bool checkReadyForInit();
    bool checkNewKfReq();
    bool checkNewKfReq(std::shared_ptr<Frame>& pframe, std::shared_ptr<MapManager>& pmap, bool isleft);
    bool checkNewKfReq_diverg(std::shared_ptr<Frame>& pframe_l, std::shared_ptr<Frame>& pframe_r, std::shared_ptr<MapManager>& pmap_l, std::shared_ptr<MapManager>& pmap_r);

    void createKeyframe();

    void applyMotion();
    void updateMotion();

    void resetFrame();
    void resetFrame(std::shared_ptr<Frame>& pframe, std::shared_ptr<MapManager>& pmap);
    void reset();

    std::shared_ptr<SlamParams> pslamstate_;
    std::shared_ptr<Frame> pcurframe_;
    std::shared_ptr<Frame> pcurframe_l_, pcurframe_r_, pcurframe_lm_, pcurframe_ls_, pcurframe_rm_, pcurframe_rs_;
    std::shared_ptr<MapManager> pmap_;
    std::shared_ptr<MapManager> pmap_l_, pmap_r_, pmap_lm_, pmap_ls_, pmap_rm_, pmap_rs_;//mono_stereo
    std::shared_ptr<MapManager> pmap_l_to_r_, pmap_r_to_l_;

    std::shared_ptr<FeatureTracker> ptracker_;

    cv::Mat left_raw_img_;
    cv::Mat cur_img_, prev_img_;//后面可以把这个注释掉，原本所有用到的地方都要修改替换
    cv::Mat cur_imgl_, prev_imgl_, cur_imgr_, prev_imgr_, cur_imglm_, prev_imglm_, cur_imgls_, prev_imgls_, cur_imgrm_, prev_imgrm_, cur_imgrs_, prev_imgrs_;//mono_stereo 不知道具体加谁，先都加着吧
    std::vector<cv::Mat> cur_pyr_, prev_pyr_;//后面可以把这个注释掉，原本所有用到的地方都要修改替换
    std::vector<cv::Mat> cur_pyrl_, prev_pyrl_, cur_pyrr_, prev_pyrr_, cur_pyrlm_, prev_pyrlm_, cur_pyrls_, prev_pyrls_, cur_pyrrm_, prev_pyrrm_, cur_pyrrs_, prev_pyrrs_;//mono_stereo 不知道具体加谁，先都加着吧
    std::vector<cv::Mat> kf_pyr_;
    std::vector<cv::Mat> kf_pyrl_, kf_pyrr_, kf_pyrlm_, kf_pyrls_, kf_pyrrm_, kf_pyrrs_;//mono_stereo 不知道具体加谁，先都加着吧

    MotionModel motion_model_, motion_model_r_;

    bool bp3preq_ = false; 
    bool bp3preq_r_ = false;
};
