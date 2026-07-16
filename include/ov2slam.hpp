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
#include <queue>
#include <mutex>

#include "slam_params.hpp"
#include "ros_visualizer.hpp"

#include "logger.hpp"

#include "camera_calibration.hpp"
#include "feature_extractor.hpp"
#include "feature_tracker.hpp"

#include "frame.hpp"
#include "map_manager.hpp"
#include "visual_front_end.hpp"
#include "mapper.hpp"
#include "estimator.hpp"

class SlamManager {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SlamManager(std::shared_ptr<SlamParams> pstate, std::shared_ptr<RosVisualizer> pviz);

    void run();
    void run2(bool isleft);

    bool getNewImage(cv::Mat &iml, cv::Mat &imr, double &time);
    bool getNewImage(cv::Mat &iml, cv::Mat &imr, cv::Mat &iml_m, cv::Mat &iml_s, cv::Mat &imr_m, cv::Mat &imr_s, double &time);
    bool getNewImage(cv::Mat &iml, cv::Mat &imr, cv::Mat &iml_m, cv::Mat &iml_s, cv::Mat &imr_m, cv::Mat &imr_s, double &time, bool isleft);

    void addNewStereoImages(const double time, cv::Mat &im0, cv::Mat &im1);
    void addNewMonoImage(const double time, cv::Mat &im0);

    void setupCalibration();
    void setupStereoCalibration();
    
    void reset();
    void reset(bool isleft);

    void writeResults();

    void writeFullTrajectoryLC();

    void visualizeAtFrameRate(const double time);
    void visualizeFrame(const cv::Mat &imleft, const double time);
    void visualizeFrame_left(const cv::Mat &imleft, const double time);
    void visualizeFrame_right(const cv::Mat &imright, const double time);
    void visualizeVOTraj(const double time);

    void visualizeAtKFsRate(const double time);
    void visualizeCovisibleKFs(const double time);
    void visualizeFullKFsTraj(const double time);
    
    void visualizeFinalKFsTraj();

    cv::Mat generateVirtualImage(const cv::Mat& realImage, const int& virtualImageWidth, const int& virtualImageHeight, const Eigen::Matrix3d& R, const Eigen::Matrix3d& virtualCameraK, const Eigen::Matrix3d& realCameraK, const Eigen::Vector3d& T, const bool& cuda);
    Eigen::Vector2d projectToRealCamera(const Eigen::Vector3d& point3D, const Eigen::Matrix3d& realCameraK);

    int frame_id_ = -1;
    bool bnew_img_available_ = false;

    bool bexit_required_ = false;
    bool bis_on_ = false;
    
    bool bframe_viz_ison_ = false;
    bool bkf_viz_ison_ = false;

    std::shared_ptr<SlamParams> pslamstate_;
    std::shared_ptr<RosVisualizer> prosviz_;

    std::shared_ptr<CameraCalibration> pcalib_model_left_;
    std::shared_ptr<CameraCalibration> pcalib_model_right_;

    //111-添加4个虚拟相机
    std::shared_ptr<CameraCalibration> pcalib_model_left_mono_;
    std::shared_ptr<CameraCalibration> pcalib_model_right_mono_;
    std::shared_ptr<CameraCalibration> pcalib_model_left_stereo_;
    std::shared_ptr<CameraCalibration> pcalib_model_right_stereo_;

    std::shared_ptr<Frame> pcurframe_;
    std::shared_ptr<Frame> pcurframe_l_, pcurframe_r_, pcurframe_lm_, pcurframe_ls_, pcurframe_rm_, pcurframe_rs_;

    std::shared_ptr<MapManager> pmap_;
    std::shared_ptr<MapManager> pmap_l_, pmap_r_, pmap_lm_, pmap_ls_, pmap_rm_, pmap_rs_;
    std::shared_ptr<MapManager> pmap_l_to_r_, pmap_r_to_l_;

    std::unique_ptr<VisualFrontEnd> pvisualfrontend_;
    std::unique_ptr<Mapper> pmapper_;

    std::shared_ptr<FeatureExtractor> pfeatextract_;
    std::shared_ptr<FeatureTracker> ptracker_;

    std::queue<cv::Mat> qimg_left_, qimg_right_;
    std::queue<double> qimg_time_;

    std::mutex img_mutex_;
    std::mutex img_mutex2_;
};