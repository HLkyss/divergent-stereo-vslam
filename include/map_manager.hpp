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


#include <mutex>
#include <unordered_map>

#include <pcl_ros/point_cloud.h>

#include "slam_params.hpp"
#include "frame.hpp"
#include "map_point.hpp"
#include "feature_extractor.hpp"
#include "feature_tracker.hpp"


class MapManager {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MapManager() {}

    MapManager(std::shared_ptr<SlamParams> pstate, std::shared_ptr<Frame> pframe, std::shared_ptr<FeatureExtractor> pfeatextract, std::shared_ptr<FeatureTracker> ptracker);
    MapManager(std::shared_ptr<SlamParams> pstate, std::shared_ptr<Frame> pframe, std::shared_ptr<Frame> pframe_l, std::shared_ptr<Frame> pframe_r, std::shared_ptr<Frame> pframe_lm, std::shared_ptr<Frame> pframe_ls, std::shared_ptr<Frame> pframe_rm, std::shared_ptr<Frame> pframe_rs, std::shared_ptr<FeatureExtractor> pfeatextract, std::shared_ptr<FeatureTracker> ptracker);//mono_stereo

    void prepareFrame();
    void prepareFrame(std::shared_ptr<Frame> &pframe);
    void prepareFrame_s(std::shared_ptr<Frame> &pframe);

    void addKeyframe();
    void addKeyframe(std::shared_ptr<Frame> &pframe);
    void addMapPoint(const cv::Scalar &color = cv::Scalar(200));
    void addMapPoint(const cv::Mat &desc, const cv::Scalar &color = cv::Scalar(200));

    std::shared_ptr<Frame> getKeyframe(const int kfid) const;
    std::shared_ptr<MapPoint> getMapPoint(const int lmid) const;

    void updateMapPoint(const int lmid, const Eigen::Vector3d &wpt, const double kfanch_invdepth=-1.);
    void updateMapPoint(const int lmid, const Eigen::Vector3d &wpt, bool isleft, const double kfanch_invdepth=-1.);
    void updateMapPoint_s(const int lmid, const Eigen::Vector3d &wpt, bool isleft, const double kfanch_invdepth=-1.);
    void addMapPointKfObs(const int lmid, const int kfid);

    bool setMapPointObs(const int lmid);

    void updateFrameCovisibility(Frame &frame);
    void updateFrameCovisibility(Frame &frame, bool isleft);
    void mergeMapPoints(const int prevlmid, const int newlmid);
//    void mergeMapPoints(const int prevlmid, const int newlmid, std::shared_ptr<Frame> &pframe);
    void mergeMapPoints(const int prevlmid, const int newlmid, bool isleft);

    void removeKeyframe(const int kfid);
    void removeMapPoint(const int lmid);
    void removeMapPointObs(const int lmid, const int kfid);
    void removeMapPointObs_s(const int lmid, const int kfid);
    void removeMapPointObs(MapPoint &lm, Frame &frame);

    void removeObsFromCurFrameById(const int lmid);
    void removeObsFromCurFrameById(const int lmid, std::shared_ptr<Frame> &pframe);
    void removeObsFromCurFrameByIdx(const int kpidx);

    void createKeyframe(const cv::Mat &im, const cv::Mat &imraw);
    void createKeyframe(const cv::Mat &im, const cv::Mat &imraw, std::shared_ptr<Frame> &pframe, bool isleft);
    void createKeyframe_s(const cv::Mat &im, const cv::Mat &imraw, std::shared_ptr<Frame> &pframe);
    void createKeyframe_s2(const cv::Mat &im, const cv::Mat &imraw, std::shared_ptr<Frame> &pframe);
    void addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, Frame &frame);
    void addKeypointsToFrame_s(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, Frame &frame);
    void addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts,
                const std::vector<int> &vscales, Frame &frame);
    void addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, 
                const std::vector<cv::Mat> &vdescs, Frame &frame);
    void addKeypointsToFrame_s(const cv::Mat &im, const std::vector<cv::Point2f> &vpts,
                             const std::vector<cv::Mat> &vdescs, Frame &frame);
    void addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, 
                const std::vector<int> &vscales, const std::vector<float> &vangles, 
                const std::vector<cv::Mat> &vdescs, Frame &frame);

    void extractKeypoints(const cv::Mat &im, const cv::Mat &imraw);
    void extractKeypoints(const cv::Mat &im, const cv::Mat &im_raw, std::shared_ptr<Frame> &pframe, bool isleft);
    void extractKeypoints(const cv::Mat &im, const cv::Mat &im_raw, std::shared_ptr<Frame> &pframe, std::shared_ptr<Frame> &pframe_add, bool isleft);
    void extractKeypoints_s(const cv::Mat &im, const cv::Mat &im_raw, std::shared_ptr<Frame> &pframe);

    void describeKeypoints(const cv::Mat &im, const std::vector<Keypoint> &vkps, 
                const std::vector<cv::Point2f> &vpts, 
                const std::vector<int> *pvscales = nullptr, 
                std::vector<float> *pvangles = nullptr);
    void describeKeypoints(const cv::Mat &im, const std::vector<Keypoint> &vkps,
                           const std::vector<cv::Point2f> &vpts,
                           std::shared_ptr<Frame> &pframe,
                           const std::vector<int> *pvscales = nullptr,
                           std::vector<float> *pvangles = nullptr);

    void kltStereoTracking(const std::vector<cv::Mat> &vleftpyr, 
                const std::vector<cv::Mat> &vrightpyr);

    void stereoMatching(Frame &frame, const std::vector<cv::Mat> &vleftpyr, 
                const std::vector<cv::Mat> &vrightpyr);
    void stereoMatching_s(Frame &frame, const std::vector<cv::Mat> &vleftpyr,
                        const std::vector<cv::Mat> &vrightpyr);// mono_stereo
    void stereoMatching_s2(Frame &frame_s, Frame &frame, Frame &frame_r, const std::vector<cv::Mat> &vleftpyr_origin, const std::vector<cv::Mat> &vrightpyr_origin, const std::vector<cv::Mat> &vleftpyr,
                          const std::vector<cv::Mat> &vrightpyr);// mono_stereo
    void stereoMatching_s2_r(Frame &frame_s, Frame &frame, Frame &frame_r, const std::vector<cv::Mat> &vleftpyr_origin, const std::vector<cv::Mat> &vrightpyr_origin, const std::vector<cv::Mat> &vleftpyr,
                           const std::vector<cv::Mat> &vrightpyr);// mono_stereo

    void guidedMatching(Frame &frame);

    void triangulate(Frame &frame);
    void triangulateTemporal(Frame &frame);
    void triangulateStereo(Frame &frame);

    Eigen::Vector3d computeTriangulation(const Sophus::SE3d &T, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr);

    void getDepthDistribution(const Frame &frame, double &mean_depth, double &std_depth);

    void reset();
    
    int nlmid_, nkfid_;
    int nblms_, nbkfs_;

    std::shared_ptr<SlamParams> pslamstate_;
    std::shared_ptr<FeatureExtractor> pfeatextract_;
    std::shared_ptr<FeatureTracker> ptracker_;

    std::shared_ptr<Frame> pcurframe_;
    std::shared_ptr<Frame> pcurframe_l_, pcurframe_r_, pcurframe_lm_, pcurframe_ls_, pcurframe_rm_, pcurframe_rs_;

    std::unordered_map<int, std::shared_ptr<Frame>> map_pkfs_;
//    std::unordered_map<int, std::shared_ptr<Frame>> map_pkfs_l_, map_pkfs_r_, map_pkfs_lm_, map_pkfs_ls_, map_pkfs_rm_, map_pkfs_rs_;
    std::unordered_map<int, std::shared_ptr<MapPoint>> map_plms_;
//    std::unordered_map<int, std::shared_ptr<MapPoint>> map_plms_l_, map_plms_r_, map_plms_lm_, map_plms_ls_, map_plms_rm_, map_plms_rs_;//map_plms_属于pmap_，pmap_已经单独分左右了，每个pmap_的map_plms_就不用再分左右了

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud_;

    mutable std::mutex kf_mutex_, lm_mutex_;
    mutable std::mutex curframe_mutex_;

    mutable std::mutex map_mutex_, optim_mutex_;
};