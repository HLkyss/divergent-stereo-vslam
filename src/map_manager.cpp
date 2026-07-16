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

#include <opencv2/highgui.hpp>

#include "multi_view_geometry.hpp"

#include "map_manager.hpp"

MapManager::MapManager(std::shared_ptr<SlamParams> pstate, std::shared_ptr<Frame> pframe, std::shared_ptr<FeatureExtractor> pfeatextract, std::shared_ptr<FeatureTracker> ptracker)
    : nlmid_(0), nkfid_(0), nblms_(0), nbkfs_(0), pslamstate_(pstate), pfeatextract_(pfeatextract), ptracker_(ptracker), pcurframe_(pframe)
{
    pcloud_.reset( new pcl::PointCloud<pcl::PointXYZRGB>() );
    pcloud_->points.reserve(1e5);
}

MapManager::MapManager(std::shared_ptr<SlamParams> pstate, std::shared_ptr<Frame> pframe, std::shared_ptr<Frame> pframe_l, std::shared_ptr<Frame> pframe_r, std::shared_ptr<Frame> pframe_lm, std::shared_ptr<Frame> pframe_ls, std::shared_ptr<Frame> pframe_rm, std::shared_ptr<Frame> pframe_rs, std::shared_ptr<FeatureExtractor> pfeatextract, std::shared_ptr<FeatureTracker> ptracker)//mono_stereo
        : nlmid_(0), nkfid_(0), nblms_(0), nbkfs_(0), pslamstate_(pstate), pfeatextract_(pfeatextract), ptracker_(ptracker), pcurframe_(pframe), pcurframe_l_(pframe_l), pcurframe_r_(pframe_r), pcurframe_lm_(pframe_lm), pcurframe_ls_(pframe_ls), pcurframe_rm_(pframe_rm), pcurframe_rs_(pframe_rs)
{
    pcloud_.reset( new pcl::PointCloud<pcl::PointXYZRGB>() );
    pcloud_->points.reserve(1e5);
}


// This function turn the current frame into a Keyframe.
// Keypoints extraction is performed and the related MPs and
// the new KF are added to the map.
void MapManager::createKeyframe(const cv::Mat &im, const cv::Mat &imraw)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.FE_createKeyframe");

    // Prepare Frame to become a KF
    // (Update observations between MPs / KFs)
    prepareFrame();

    // Detect in im and describe in imraw
    extractKeypoints(im, imraw);

    // Add KF to the map
    addKeyframe();

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.FE_createKeyframe");
}

void MapManager::createKeyframe(const cv::Mat &im, const cv::Mat &imraw, std::shared_ptr<Frame> &pframe, bool isleft)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.FE_createKeyframe");

    // Prepare Frame to become a KF
    // (Update observations between MPs / KFs)
//    prepareFrame();   // todo mono_stereo
    prepareFrame(pframe);
//    std::cout<<"prepareFrame done "<<pframe->nbwcells_s_<<std::endl;

    // Detect in im and describe in imraw
//    extractKeypoints(im, imraw); // todo mono_stereo
    extractKeypoints(im, imraw, pframe, isleft);
//    std::cout<<"extractKeypoints done "<<pframe->nbwcells_s_<<std::endl;

    // Add KF to the map
//    addKeyframe(); // todo mono_stereo
    addKeyframe(pframe);
//    std::cout<<"addKeyframe done "<<pframe->nbwcells_s_<<std::endl;

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.FE_createKeyframe");
}

void MapManager::createKeyframe_s(const cv::Mat &im, const cv::Mat &imraw, std::shared_ptr<Frame> &pframe)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.FE_createKeyframe");

    // Prepare Frame to become a KF
    // (Update observations between MPs / KFs)
//    prepareFrame();   // todo mono_stereo
    prepareFrame_s(pframe);

    // Detect in im and describe in imraw
//    extractKeypoints(im, imraw); // todo mono_stereo
    extractKeypoints_s(im, imraw, pframe);

    // Add KF to the map
//    addKeyframe(); // todo mono_stereo
    addKeyframe(pframe);

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.FE_createKeyframe");
}

void MapManager::createKeyframe_s2(const cv::Mat &im, const cv::Mat &imraw, std::shared_ptr<Frame> &pframe)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.FE_createKeyframe");

    // Prepare Frame to become a KF
    // (Update observations between MPs / KFs)
//    prepareFrame();   // todo mono_stereo
    prepareFrame_s(pframe);

    // Detect in im and describe in imraw
//    extractKeypoints(im, imraw); // todo mono_stereo
//    extractKeypoints_s(im, imraw, pframe);

    // Add KF to the map
//    addKeyframe(); // todo mono_stereo
    addKeyframe(pframe);

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.FE_createKeyframe");
}

// Prepare Frame to become a KF
// (Update observations between MPs / KFs)
void MapManager::prepareFrame()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_CF_prepareFrame");

    // Update new KF id
    pcurframe_->kfid_ = nkfid_;

    // Filter if too many kps
    if( (int)pcurframe_->nbkps_ > pslamstate_->nbmaxkps_ ) {
        for( const auto &vkpids : pcurframe_->vgridkps_ ) {
            if( vkpids.size() > 2 ) {
                int lmid2remove = -1;
                size_t minnbobs = std::numeric_limits<size_t>::max();
                for( const auto &lmid : vkpids ) {
                    auto plmit = map_plms_.find(lmid);
                    if( plmit != map_plms_.end() ) {
                        size_t nbobs = plmit->second->getKfObsSet().size();
                        if( nbobs < minnbobs ) {
                            lmid2remove = lmid;
                            minnbobs = nbobs;
                        }
                    } else {
                        removeObsFromCurFrameById(lmid);
                        break;
                    }
                }
                if( lmid2remove >= 0 ) {
                    removeObsFromCurFrameById(lmid2remove);
                }
            }
        }
    }

    for( const auto &kp : pcurframe_->getKeypoints() ) {

        // Get the related MP
        auto plmit = map_plms_.find(kp.lmid_);
        
        if( plmit == map_plms_.end() ) {
            removeObsFromCurFrameById(kp.lmid_);
            continue;
        }

        // Relate new KF id to the MP
        plmit->second->addKfObs(nkfid_);
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_CF_prepareFrame");
}

void MapManager::prepareFrame(std::shared_ptr<Frame> &pframe)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_CF_prepareFrame");

    // Update new KF id
    pframe->kfid_ = nkfid_;

    // Filter if too many kps
    if( (int)pframe->nbkps_ > pslamstate_->nbmaxkps_ ) {
        for( const auto &vkpids : pframe->vgridkps_ ) {
            if( vkpids.size() > 2 ) {
                int lmid2remove = -1;
                size_t minnbobs = std::numeric_limits<size_t>::max();
                for( const auto &lmid : vkpids ) {
                    auto plmit = map_plms_.find(lmid);
                    if( plmit != map_plms_.end() ) {
                        size_t nbobs = plmit->second->getKfObsSet().size();
                        if( nbobs < minnbobs ) {
                            lmid2remove = lmid;
                            minnbobs = nbobs;
                        }
                    } else {
//                        removeObsFromCurFrameById(lmid);
                        removeObsFromCurFrameById(lmid, pframe);
                        break;
                    }
                }
                if( lmid2remove >= 0 ) {
//                    removeObsFromCurFrameById(lmid2remove);
                    removeObsFromCurFrameById(lmid2remove, pframe);
                }
            }
        }
    }

    for( const auto &kp : pframe->getKeypoints() ) {

        // Get the related MP
        auto plmit = map_plms_.find(kp.lmid_);

        if( plmit == map_plms_.end() ) {
//            removeObsFromCurFrameById(kp.lmid_);
            removeObsFromCurFrameById(kp.lmid_, pframe);
            continue;
        }

        // Relate new KF id to the MP
        plmit->second->addKfObs(nkfid_);
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_CF_prepareFrame");
}

void MapManager::prepareFrame_s(std::shared_ptr<Frame> &pframe)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_CF_prepareFrame");

    // Update new KF id
    pframe->kfid_ = nkfid_;

    // Filter if too many kps
    if( (int)pframe->nbkps_ > pslamstate_->nbmaxkps_s_ ) {
        for( const auto &vkpids : pframe->vgridkps_s_ ) {
            if( vkpids.size() > 2 ) {
                int lmid2remove = -1;
                size_t minnbobs = std::numeric_limits<size_t>::max();
                for( const auto &lmid : vkpids ) {
                    auto plmit = map_plms_.find(lmid);
                    if( plmit != map_plms_.end() ) {
                        size_t nbobs = plmit->second->getKfObsSet().size();
                        if( nbobs < minnbobs ) {
                            lmid2remove = lmid;
                            minnbobs = nbobs;
                        }
                    } else {
//                        removeObsFromCurFrameById(lmid);
                        removeObsFromCurFrameById(lmid, pframe);
                        break;
                    }
                }
                if( lmid2remove >= 0 ) {
//                    removeObsFromCurFrameById(lmid2remove);
                    removeObsFromCurFrameById(lmid2remove, pframe);
                }
            }
        }
    }

    for( const auto &kp : pframe->getKeypoints() ) {

        // Get the related MP
        auto plmit = map_plms_.find(kp.lmid_);

        if( plmit == map_plms_.end() ) {
//            removeObsFromCurFrameById(kp.lmid_);
            removeObsFromCurFrameById(kp.lmid_, pframe);
            continue;
        }

        // Relate new KF id to the MP
        plmit->second->addKfObs(nkfid_);
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_CF_prepareFrame");
}

void MapManager::updateFrameCovisibility(Frame &frame)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_updateFrameCovisilbity");

    // Update the MPs and the covisilbe graph between KFs
    std::map<int,int> map_covkfs;
    std::unordered_set<int> set_local_mapids;

    for( const auto &kp : frame.getKeypoints() ) {

        // Get the related MP
        auto plmit = map_plms_.find(kp.lmid_);
        
        if( plmit == map_plms_.end() ) {
            removeMapPointObs(kp.lmid_, frame.kfid_);
            removeObsFromCurFrameById(kp.lmid_);
            continue;
        }

        // Get the set of KFs observing this KF to update 
        // covisible KFs
        for( const auto &kfid : plmit->second->getKfObsSet() ) 
        {
            if( kfid != frame.kfid_ ) 
            {
                auto it = map_covkfs.find(kfid);
                if( it != map_covkfs.end() ) {
                    it->second += 1;
                } else {
                    map_covkfs.emplace(kfid, 1);
                }
            }
        }
    }

    // Update covisibility for covisible KFs
    std::set<int> set_badkfids;
    for( const auto &kfid_cov : map_covkfs ) 
    {
        int kfid = kfid_cov.first;
        int covscore = kfid_cov.second;
        
        auto pkfit = map_pkfs_.find(kfid);
        if( pkfit != map_pkfs_.end() ) 
        {
            // Will emplace or update covisiblity
            pkfit->second->map_covkfs_[frame.kfid_] = covscore;

            // Set the unobserved local map for future tracking
            for( const auto &kp : pkfit->second->getKeypoints3d() ) {
                if( !frame.isObservingKp(kp.lmid_) ) {
                    set_local_mapids.insert(kp.lmid_);
                }
            }
        } else {
            set_badkfids.insert(kfid);
        }
    }

    for( const auto &kfid : set_badkfids ) {
        map_covkfs.erase(kfid);
    }
    
    // Update the set of covisible KFs
    frame.map_covkfs_.swap(map_covkfs);

    // Update local map of unobserved MPs
    if( set_local_mapids.size() > 0.5 * frame.set_local_mapids_.size() ) {
        frame.set_local_mapids_.swap(set_local_mapids);
    } else {
        frame.set_local_mapids_.insert(set_local_mapids.begin(), set_local_mapids.end());
    }
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_updateFrameCovisilbity");
}

void MapManager::updateFrameCovisibility(Frame &frame, bool isleft)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_updateFrameCovisilbity");

    // Update the MPs and the covisilbe graph between KFs
    std::map<int,int> map_covkfs;
    std::unordered_set<int> set_local_mapids;

    for( const auto &kp : frame.getKeypoints() ) {

        // Get the related MP
        auto plmit = map_plms_.find(kp.lmid_);

        if( plmit == map_plms_.end() ) {
            removeMapPointObs(kp.lmid_, frame.kfid_);
            if(isleft){
                removeObsFromCurFrameById(kp.lmid_, pcurframe_l_);//
            } else {
                removeObsFromCurFrameById(kp.lmid_, pcurframe_r_);//
            }
            continue;
        }

        // Get the set of KFs observing this KF to update
        // covisible KFs
        for( const auto &kfid : plmit->second->getKfObsSet() )
        {
            if( kfid != frame.kfid_ )
            {
                auto it = map_covkfs.find(kfid);
                if( it != map_covkfs.end() ) {
                    it->second += 1;
                } else {
                    map_covkfs.emplace(kfid, 1);
                }
            }
        }
    }

    // Update covisibility for covisible KFs
    std::set<int> set_badkfids;
    for( const auto &kfid_cov : map_covkfs )
    {
        int kfid = kfid_cov.first;
        int covscore = kfid_cov.second;

        auto pkfit = map_pkfs_.find(kfid);
        if( pkfit != map_pkfs_.end() )
        {
            // Will emplace or update covisiblity
            pkfit->second->map_covkfs_[frame.kfid_] = covscore;

            // Set the unobserved local map for future tracking
            for( const auto &kp : pkfit->second->getKeypoints3d() ) {
                if( !frame.isObservingKp(kp.lmid_) ) {
                    set_local_mapids.insert(kp.lmid_);
                }
            }
        } else {
            set_badkfids.insert(kfid);
        }
    }

    for( const auto &kfid : set_badkfids ) {
        map_covkfs.erase(kfid);
    }

    // Update the set of covisible KFs
    frame.map_covkfs_.swap(map_covkfs);

    // Update local map of unobserved MPs
    if( set_local_mapids.size() > 0.5 * frame.set_local_mapids_.size() ) {
        frame.set_local_mapids_.swap(set_local_mapids);
    } else {
        frame.set_local_mapids_.insert(set_local_mapids.begin(), set_local_mapids.end());
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_updateFrameCovisilbity");
}

void MapManager::addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    
    // Add keypoints + create MPs
    size_t nbpts = vpts.size();
    for( size_t i = 0 ; i < nbpts ; i++ ) {
        // Add keypoint to current frame
        frame.addKeypoint(vpts.at(i), nlmid_);

        // Create landmark with same id
        cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
        addMapPoint(col);
    }
}

void MapManager::addKeypointsToFrame_s(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);

    // Add keypoints + create MPs
    size_t nbpts = vpts.size();
    for( size_t i = 0 ; i < nbpts ; i++ ) {
        // Add keypoint to current frame
        frame.addKeypoint_s(vpts.at(i), nlmid_);//

        // Create landmark with same id
        cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
        addMapPoint(col);
    }
}

void MapManager::addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, 
    const std::vector<int> &vscales, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    
    // Add keypoints + create landmarks
    size_t nbpts = vpts.size();
    for( size_t i = 0 ; i < nbpts ; i++ )
    {
        // Add keypoint to current frame
        frame.addKeypoint(vpts.at(i), nlmid_, vscales.at(i));

        // Create landmark with same id
        cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
        addMapPoint(col);
    }
}

void MapManager::addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, const std::vector<cv::Mat> &vdescs, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    
    // Add keypoints + create landmarks
    size_t nbpts = vpts.size();
    for( size_t i = 0 ; i < nbpts ; i++ )
    {
        if( !vdescs.at(i).empty() ) {
            // Add keypoint to current frame
            frame.addKeypoint(vpts.at(i), nlmid_, vdescs.at(i));

            // Create landmark with same id
            cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
            addMapPoint(vdescs.at(i), col);
        } 
        else {
            // Add keypoint to current frame
            frame.addKeypoint(vpts.at(i), nlmid_);

            // Create landmark with same id
            cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
            addMapPoint(col);
        }
    }
}

void MapManager::addKeypointsToFrame_s(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, const std::vector<cv::Mat> &vdescs, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);

    // Add keypoints + create landmarks
    size_t nbpts = vpts.size();
    for( size_t i = 0 ; i < nbpts ; i++ )
    {
        if( !vdescs.at(i).empty() ) {
            // Add keypoint to current frame
            frame.addKeypoint_s(vpts.at(i), nlmid_, vdescs.at(i));//

            // Create landmark with same id
            cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
            addMapPoint(vdescs.at(i), col);
        }
        else {
            // Add keypoint to current frame
            frame.addKeypoint_s(vpts.at(i), nlmid_);//

            // Create landmark with same id
            cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
            addMapPoint(col);
        }
    }
}


void MapManager::addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, const std::vector<int> &vscales, const std::vector<float> &vangles, 
                        const std::vector<cv::Mat> &vdescs, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    
    // Add keypoints + create landmarks
    size_t nbpts = vpts.size();
    for( size_t i = 0 ; i < nbpts ; i++ )
    {
        if( !vdescs.at(i).empty() ) {
            // Add keypoint to current frame
            frame.addKeypoint(vpts.at(i), nlmid_, vdescs.at(i), vscales.at(i), vangles.at(i));

            // Create landmark with same id
            cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
            addMapPoint(vdescs.at(i), col);
        } 
        else {
            // Add keypoint to current frame
            frame.addKeypoint(vpts.at(i), nlmid_);

            // Create landmark with same id
            cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
            addMapPoint(col);
        }
    }
}

// Extract new kps into provided image and update cur. Frame
void MapManager::extractKeypoints(const cv::Mat &im, const cv::Mat &imraw)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_CF_extractKeypoints");

    std::vector<Keypoint> vkps = pcurframe_->getKeypoints();

    std::vector<cv::Point2f> vpts;
    std::vector<int> vscales;
    std::vector<float> vangles;

    for( auto &kp : vkps ) {
        vpts.push_back(kp.px_);
    }

    if( pslamstate_->use_brief_ ) {
        describeKeypoints(imraw, vkps, vpts);
    }

    int nb2detect = pslamstate_->nbmaxkps_ - pcurframe_->noccupcells_;

    if( nb2detect > 0 ) {
        // Detect kps in the provided images
        // using the cur kps and img roi to set a mask
        std::vector<cv::Point2f> vnewpts;

        if( pslamstate_->use_shi_tomasi_ ) {
            vnewpts = pfeatextract_->detectGFTT(im, vpts, pcurframe_->pcalib_leftcam_->roi_mask_, nb2detect);
        } 
        else if( pslamstate_->use_fast_ ) {
            vnewpts = pfeatextract_->detectGridFAST(im, pslamstate_->nmaxdist_, vpts, pcurframe_->pcalib_leftcam_->roi_rect_);
        } 
        else if ( pslamstate_->use_singlescale_detector_ ) {
            vnewpts = pfeatextract_->detectSingleScale(im, pslamstate_->nmaxdist_, vpts, pcurframe_->pcalib_leftcam_->roi_rect_);
        } else {
            std::cerr << "\n Choose a detector between : gftt / FAST / SingleScale detector!";
            exit(-1);
        }

        if( !vnewpts.empty() ) {
            if( pslamstate_->use_brief_ ) {
                std::vector<cv::Mat> vdescs;
                vdescs = pfeatextract_->describeBRIEF(imraw, vnewpts);
                addKeypointsToFrame(im, vnewpts, vdescs, *pcurframe_);
            } 
            else if( pslamstate_->use_shi_tomasi_ || pslamstate_->use_fast_ 
                || pslamstate_->use_singlescale_detector_ ) 
            {
                addKeypointsToFrame(im, vnewpts, *pcurframe_);
            }
        }
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_CF_extractKeypoints");
}

void MapManager::extractKeypoints(const cv::Mat &im, const cv::Mat &im_raw, std::shared_ptr<Frame> &pframe, bool isleft)   //mono_stereo
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_CF_extractKeypoints");

    std::vector<Keypoint> vkps = pframe->getKeypoints();
//    std::cout << "Nb of kps in cur frame : " << vkps.size() << "\n";

    std::vector<cv::Point2f> vpts;
    std::vector<int> vscales;
    std::vector<float> vangles;

    for( auto &kp : vkps ) {
        vpts.push_back(kp.px_);
    }

    if( pslamstate_->use_brief_ ) {
//        describeKeypoints(im_raw, vkps, vpts);//todo mono_stereo
        describeKeypoints(im_raw, vkps, vpts, pframe);
    }

    int nb2detect = pslamstate_->nbmaxkps_ - pframe->noccupcells_;
//    std::cout << "Nb of kps to detect : " << nb2detect << "\n";

    if( nb2detect > 0 ) {
        // Detect kps in the provided images
        // using the cur kps and img roi to set a mask
        std::vector<cv::Point2f> vnewpts;

        if( pslamstate_->use_shi_tomasi_ ) {
            vnewpts = pfeatextract_->detectGFTT(im, vpts, pframe->pcalib_leftcam_->roi_mask_, nb2detect);
        }
        else if( pslamstate_->use_fast_ ) {
            vnewpts = pfeatextract_->detectGridFAST(im, pslamstate_->nmaxdist_, vpts, pframe->pcalib_leftcam_->roi_rect_);
        }
        else if ( pslamstate_->use_singlescale_detector_ ) {
            vnewpts = pfeatextract_->detectSingleScale(im, pslamstate_->nmaxdist_, vpts, pframe->pcalib_leftcam_->roi_rect_, pframe, isleft);
//            //依次输出检查输入的量im, pslamstate_->nmaxdist_, vpts, pframe->pcalib_leftcam_->roi_rect_
//            std::cout << "Image (im): " << (im.empty() ? "Empty" : "Loaded") << std::endl;
//            if (!im.empty()) {
//                std::cout << "Image size: " << im.size() << std::endl;
//            }
//            std::cout << "Max distance (nmaxdist_): " << pslamstate_->nmaxdist_ << std::endl;
//            std::cout << "Initial vpts size: " << vpts.size() << std::endl;
//
//            const cv::Rect& roi = pframe->pcalib_leftcam_->roi_rect_;
//            std::cout << "ROI Rect (pcalib_leftcam_->roi_rect_): ";
//            std::cout << "x: " << roi.x << ", y: " << roi.y
//                      << ", width: " << roi.width << ", height: " << roi.height << std::endl;
//
//            // 在图像上绘制矩形框（ROI）
//            cv::Mat img_with_roi = im.clone();  // 克隆原图，以免修改原图
//            cv::rectangle(img_with_roi, roi, cv::Scalar(0, 255, 0), 2);  // 绿色矩形框，线宽为2
//
//            // 显示带有 ROI 的图像
//            cv::imshow("Image with ROI", img_with_roi);
//            cv::waitKey(1);  // 等待按键

        } else {
            std::cerr << "\n Choose a detector between : gftt / FAST / SingleScale detector!";
            exit(-1);
        }

        //可视化出特征点
        cv::Mat img = im.clone();
        if (img.channels() == 1)
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        if(isleft){
            for(auto &kp : vnewpts)//绿色：来自新提取
            {
                cv::circle(img, kp, 2, cv::Scalar(0, 255, 0), 2);
            }
            for(auto &kp : vpts)//红色：来自pframe->getKeypoints
            {
                cv::circle(img, kp, 2, cv::Scalar(0, 0, 255), 2);
            }
//            cv::imshow("new_kps 1", img);
//            cv::imwrite("/home/hl/project/ov2_diverg_ws/test/new_kps1.jpg", img);//没有天空点
//            cv::waitKey(1);
        }

        if( !vnewpts.empty() ) {
            if( pslamstate_->use_brief_ ) {
                std::vector<cv::Mat> vdescs;
                vdescs = pfeatextract_->describeBRIEF(im_raw, vnewpts);
                addKeypointsToFrame(im, vnewpts, vdescs, *pframe);
            }
            else if( pslamstate_->use_shi_tomasi_ || pslamstate_->use_fast_
                     || pslamstate_->use_singlescale_detector_ )
            {
                addKeypointsToFrame(im, vnewpts, *pframe);
            }
        }
        //输出vpts和vnewpts点数量
//        if(isleft){
//            std::cout << "----------- left: vpts size: " << vpts.size() <<  ", vnewpts size: " << vnewpts.size() << std::endl;
//        } else {
//            std::cout << "----------- right: vpts size: " << vpts.size() <<  ", vnewpts size: " << vnewpts.size() << std::endl;
//        }
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_CF_extractKeypoints");
}

void MapManager::extractKeypoints_s(const cv::Mat &im, const cv::Mat &im_raw, std::shared_ptr<Frame> &pframe)   //mono_stereo
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_CF_extractKeypoints");

    std::vector<Keypoint> vkps = pframe->getKeypoints();

    std::vector<cv::Point2f> vpts;
    std::vector<int> vscales;
    std::vector<float> vangles;

    for( auto &kp : vkps ) {
        vpts.push_back(kp.px_);
    }

    if( pslamstate_->use_brief_ ) {
//        describeKeypoints(im_raw, vkps, vpts);//todo mono_stereo
        describeKeypoints(im_raw, vkps, vpts, pframe);
    }

    int nb2detect = pslamstate_->nbmaxkps_s_ - pframe->noccupcells_;

    if( nb2detect > 0 ) {
        // Detect kps in the provided images
        // using the cur kps and img roi to set a mask
        std::vector<cv::Point2f> vnewpts;

        if( pslamstate_->use_shi_tomasi_ ) {
            vnewpts = pfeatextract_->detectGFTT_s(im, vpts, pframe->pcalib_leftcam_->roi_mask_, nb2detect);//
        }
        else if( pslamstate_->use_fast_ ) {
            vnewpts = pfeatextract_->detectGridFAST(im, pslamstate_->nmaxdist_, vpts, pframe->pcalib_leftcam_->roi_rect_);
        }
        else if ( pslamstate_->use_singlescale_detector_ ) {
            vnewpts = pfeatextract_->detectSingleScale(im, pslamstate_->nmaxdist_, vpts, pframe->pcalib_leftcam_->roi_rect_);
        } else {
            std::cerr << "\n Choose a detector between : gftt / FAST / SingleScale detector!";
            exit(-1);
        }

        if( !vnewpts.empty() ) {
            if( pslamstate_->use_brief_ ) {
                std::vector<cv::Mat> vdescs;
                vdescs = pfeatextract_->describeBRIEF(im_raw, vnewpts);
                addKeypointsToFrame_s(im, vnewpts, vdescs, *pframe);//
            }
            else if( pslamstate_->use_shi_tomasi_ || pslamstate_->use_fast_
                     || pslamstate_->use_singlescale_detector_ )
            {
                addKeypointsToFrame_s(im, vnewpts, *pframe);//
            }
        }
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_CF_extractKeypoints");
}

// Describe cur frame kps in cur image
void MapManager::describeKeypoints(const cv::Mat &im, const std::vector<Keypoint> &vkps, const std::vector<cv::Point2f> &vpts, const std::vector<int> *pvscales, std::vector<float> *pvangles)
{
    size_t nbkps = vkps.size();
    std::vector<cv::Mat> vdescs;

    if( pslamstate_->use_brief_ ) {
        vdescs = pfeatextract_->describeBRIEF(im, vpts);
    }

    assert( vkps.size() == vdescs.size() );

    for( size_t i = 0 ; i < nbkps ; i++ ) {
        if( !vdescs.at(i).empty() ) {
            pcurframe_->updateKeypointDesc(vkps.at(i).lmid_, vdescs.at(i));
            map_plms_.at(vkps.at(i).lmid_)->addDesc(pcurframe_->kfid_, vdescs.at(i));
        }
    }
}

void MapManager::describeKeypoints(const cv::Mat &im, const std::vector<Keypoint> &vkps, const std::vector<cv::Point2f> &vpts, std::shared_ptr<Frame> &pframe, const std::vector<int> *pvscales, std::vector<float> *pvangles)
{
    size_t nbkps = vkps.size();
    std::vector<cv::Mat> vdescs;

    if( pslamstate_->use_brief_ ) {
        vdescs = pfeatextract_->describeBRIEF(im, vpts);
    }

    assert( vkps.size() == vdescs.size() );

    for( size_t i = 0 ; i < nbkps ; i++ ) {
        if( !vdescs.at(i).empty() ) {
            pframe->updateKeypointDesc(vkps.at(i).lmid_, vdescs.at(i));
            map_plms_.at(vkps.at(i).lmid_)->addDesc(pframe->kfid_, vdescs.at(i));
        }
    }
}


// This function is responsible for performing stereo matching operations
// for the means of triangulation
void MapManager::stereoMatching(Frame &frame, const std::vector<cv::Mat> &vleftpyr, const std::vector<cv::Mat> &vrightpyr) 
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_stereoMatching");

    // Find stereo correspondances with left kps
    auto vleftkps = frame.getKeypoints();
    size_t nbkps = vleftkps.size();

    // ZNCC Parameters
    size_t nmaxpyrlvl = pslamstate_->nklt_pyr_lvl_*2;
    int winsize = 7;

    float uppyrcoef = std::pow(2,pslamstate_->nklt_pyr_lvl_);
    float downpyrcoef = 1. / uppyrcoef;
    
    std::vector<int> v3dkpids, vkpids, voutkpids, vpriorids;
    std::vector<cv::Point2f> v3dkps, v3dpriors, vkps, vpriors;

    // First we're gonna track 3d kps on only 2 levels
    v3dkpids.reserve(frame.nb3dkps_);
    v3dkps.reserve(frame.nb3dkps_);
    v3dpriors.reserve(frame.nb3dkps_);

    // Then we'll track 2d kps on full pyramid levels
    vkpids.reserve(nbkps);
    vkps.reserve(nbkps);
    vpriors.reserve(nbkps);

    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        // Set left kp
        auto &kp = vleftkps.at(i);

        // Set prior right kp
        cv::Point2f priorpt = kp.px_;

        // If 3D, check if we can find a prior in right image
        if( kp.is3d_ ) {
            auto plm = getMapPoint(kp.lmid_);
            if( plm != nullptr ) {
                cv::Point2f projpt = frame.projWorldToRightImageDist(plm->getPoint());
                if( frame.isInRightImage(projpt) ) {
                    v3dkps.push_back(kp.px_);
                    v3dpriors.push_back(projpt);
                    v3dkpids.push_back(kp.lmid_);
                    continue;
                } 
            } else {
                removeMapPointObs(kp.lmid_, frame.kfid_);
                continue;
            }
        } 
        
        // If stereo rect images, prior from SAD
        if( pslamstate_->bdo_stereo_rect_ ) {

            float xprior = -1.;
            float l1err;

            cv::Point2f pyrleftpt = kp.px_ * downpyrcoef;

            ptracker_->getLineMinSAD(vleftpyr.at(nmaxpyrlvl), vrightpyr.at(nmaxpyrlvl), pyrleftpt, winsize, xprior, l1err, true);

            xprior *= uppyrcoef;

            if( xprior >= 0 && xprior <= kp.px_.x ) {
                priorpt.x = xprior;
            }

        }
        else { // Generate prior from 3d neighbors
            const size_t nbmin3dcokps = 1;

            auto vnearkps = frame.getSurroundingKeypoints(kp);
            if( vnearkps.size() >= nbmin3dcokps ) 
            {
                std::vector<Keypoint> vnear3dkps;
                vnear3dkps.reserve(vnearkps.size());
                for( const auto &cokp : vnearkps ) {
                    if( cokp.is3d_ ) {
                        vnear3dkps.push_back(cokp);
                    }
                }

                if( vnear3dkps.size() >= nbmin3dcokps ) {
                
                    size_t nb3dkp = 0;
                    double mean_z = 0.;
                    double weights = 0.;

                    for( const auto &cokp : vnear3dkps ) {
                        auto plm = getMapPoint(cokp.lmid_);
                        if( plm != nullptr ) {
                            nb3dkp++;
                            double coef = 1. / cv::norm(cokp.unpx_ - kp.unpx_);
                            weights += coef;
                            mean_z += coef * frame.projWorldToCam(plm->getPoint()).z();
                        }
                    }

                    if( nb3dkp >= nbmin3dcokps ) {
                        mean_z /= weights;
                        Eigen::Vector3d predcampt = mean_z * ( kp.bv_ / kp.bv_.z() );

                        cv::Point2f projpt = frame.projCamToRightImageDist(predcampt);

                        if( frame.isInRightImage(projpt) ) 
                        {
                            v3dkps.push_back(kp.px_);
                            v3dpriors.push_back(projpt);
                            v3dkpids.push_back(kp.lmid_);
                            continue;
                        }
                    }
                }
            }
        }

        vkpids.push_back(kp.lmid_);
        vkps.push_back(kp.px_);
        vpriors.push_back(priorpt);
    }

    // Storing good tracks   
    std::vector<cv::Point2f> vgoodrkps;
    std::vector<int> vgoodids;
    vgoodrkps.reserve(nbkps);
    vgoodids.reserve(nbkps);

    // 1st track 3d kps if using prior
    if( !v3dpriors.empty() ) 
    {
        size_t nbpyrlvl = 1;
        int nwinsize = pslamstate_->nklt_win_size_; // What about a smaller window here?

        if( vleftpyr.size() < 2*(nbpyrlvl+1) ) {
            nbpyrlvl = vleftpyr.size() / 2 - 1;
        }

        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        ptracker_->fbKltTracking(
                    vleftpyr, 
                    vrightpyr, 
                    nwinsize, 
                    nbpyrlvl, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    v3dkps, 
                    v3dpriors, 
                    vkpstatus);

        size_t nbgood = 0;
        size_t nb3dkps = v3dkps.size();
        
        for(size_t i = 0 ; i < nb3dkps  ; i++ ) 
        {
            if( vkpstatus.at(i) ) {
                vgoodrkps.push_back(v3dpriors.at(i));
                vgoodids.push_back(v3dkpids.at(i));
                nbgood++;
            } else {
                // If tracking failed, gonna try on full pyramid size
                // without prior for 2d kps
                vkpids.push_back(v3dkpids.at(i));
                vkps.push_back(v3dkps.at(i));
                vpriors.push_back(v3dpriors.at(i));
            }
        }

        if( pslamstate_->debug_ ) 
            std::cout << "\n >>> Stereo KLT Tracking on priors : " << nbgood 
                << " out of " << nb3dkps << " kps tracked!\n";
    }

    // 2nd track other kps if any
    if( !vkps.empty() ) 
    {
        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        ptracker_->fbKltTracking(
                    vleftpyr, 
                    vrightpyr, 
                    pslamstate_->nklt_win_size_, 
                    pslamstate_->nklt_pyr_lvl_, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    vkps, 
                    vpriors, 
                    vkpstatus);

        size_t nbgood = 0;
        size_t nb2dkps = vkps.size();

        for(size_t i = 0 ; i < nb2dkps  ; i++ ) 
        {
            if( vkpstatus.at(i) ) {
                vgoodrkps.push_back(vpriors.at(i));
                vgoodids.push_back(vkpids.at(i));
                nbgood++;
            }
        }

        if( pslamstate_->debug_ )
            std::cout << "\n >>> Stereo KLT Tracking w. no priors : " << nbgood
                << " out of " << nb2dkps << " kps tracked!\n";
    }

    nbkps = vgoodids.size();
    size_t nbgood = 0;

    float epi_err = 0.;

    for( size_t i = 0; i < nbkps ; i++ ) 
    {
        cv::Point2f lunpx = frame.getKeypointById(vgoodids.at(i)).unpx_;
        cv::Point2f runpx = frame.pcalib_rightcam_->undistortImagePoint(vgoodrkps.at(i));

        // Check epipolar consistency (same row for rectified images)
        if( pslamstate_->bdo_stereo_rect_ ) {
            epi_err = fabs(lunpx.y - runpx.y);
            // Correct right kp to be on the same row
            vgoodrkps.at(i).y = lunpx.y;
        }
        else {
            epi_err = MultiViewGeometry::computeSampsonDistance(frame.Frl_, lunpx, runpx);
        }
        
        if( epi_err <= 2. ) 
        {
            frame.updateKeypointStereo(vgoodids.at(i), vgoodrkps.at(i));
            nbgood++;
        }
    }

    if( pslamstate_->debug_ )
        std::cout << "\n \t>>> Nb of stereo tracks: " << nbgood
            << " out of " << nbkps << "\n";

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_stereoMatching");
}

void MapManager::stereoMatching_s(Frame &frame, const std::vector<cv::Mat> &vleftpyr, const std::vector<cv::Mat> &vrightpyr)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_stereoMatching");

    // Find stereo correspondances with left kps
    auto vleftkps = frame.getKeypoints();
    //可视化特征点
/*    cv::Mat img = vleftpyr.at(0).clone();
    for(auto &kp : vleftkps)
    {
        cv::circle(img, kp.px_, 2, cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("left_kps", img);
    cv::waitKey(1);*/
    size_t nbkps = vleftkps.size();

    // ZNCC Parameters
    size_t nmaxpyrlvl = pslamstate_->nklt_pyr_lvl_*2;
    int winsize = 7;

    float uppyrcoef = std::pow(2,pslamstate_->nklt_pyr_lvl_);
    float downpyrcoef = 1. / uppyrcoef;

    std::vector<int> v3dkpids, vkpids, voutkpids, vpriorids;
    std::vector<cv::Point2f> v3dkps, v3dpriors, vkps, vpriors;

    // First we're gonna track 3d kps on only 2 levels
    v3dkpids.reserve(frame.nb3dkps_);
    v3dkps.reserve(frame.nb3dkps_);
    v3dpriors.reserve(frame.nb3dkps_);

    // Then we'll track 2d kps on full pyramid levels
    vkpids.reserve(nbkps);
    vkps.reserve(nbkps);
    vpriors.reserve(nbkps);

    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        // Set left kp
        auto &kp = vleftkps.at(i);

        // Set prior right kp
        cv::Point2f priorpt = kp.px_;

        // If 3D, check if we can find a prior in right image
        if( kp.is3d_ ) {
            auto plm = getMapPoint(kp.lmid_);
            if( plm != nullptr ) {
//                cv::Point2f projpt = frame.projWorldToRightImageDist(plm->getPoint());
                cv::Point2f projpt = frame.projWorldToRightImageDist_s(plm->getPoint());// mono_stereo
//                if( frame.isInRightImage(projpt) ) {
                if( frame.isInRightImage_s(projpt) ) {// mono_stereo
                    v3dkps.push_back(kp.px_);
                    v3dpriors.push_back(projpt);
                    v3dkpids.push_back(kp.lmid_);
                    continue;
                }
            } else {
//                removeMapPointObs(kp.lmid_, frame.kfid_);
                removeMapPointObs_s(kp.lmid_, frame.kfid_);// mono_stereo
                continue;
            }
        }

        // If stereo rect images, prior from SAD
        if( pslamstate_->bdo_stereo_rect_ ) {//暂时不改

            float xprior = -1.;
            float l1err;

            cv::Point2f pyrleftpt = kp.px_ * downpyrcoef;

            ptracker_->getLineMinSAD(vleftpyr.at(nmaxpyrlvl), vrightpyr.at(nmaxpyrlvl), pyrleftpt, winsize, xprior, l1err, true);

            xprior *= uppyrcoef;

            if( xprior >= 0 && xprior <= kp.px_.x ) {
                priorpt.x = xprior;
            }

        }
        else { // Generate prior from 3d neighbors
            const size_t nbmin3dcokps = 1;

//            auto vnearkps = frame.getSurroundingKeypoints(kp);
            auto vnearkps = frame.getSurroundingKeypoints_s(kp);//mono_stereo
            if( vnearkps.size() >= nbmin3dcokps )
            {
                std::vector<Keypoint> vnear3dkps;
                vnear3dkps.reserve(vnearkps.size());
                for( const auto &cokp : vnearkps ) {
                    if( cokp.is3d_ ) {
                        vnear3dkps.push_back(cokp);
                    }
                }

                if( vnear3dkps.size() >= nbmin3dcokps ) {

                    size_t nb3dkp = 0;
                    double mean_z = 0.;
                    double weights = 0.;

                    for( const auto &cokp : vnear3dkps ) {
                        auto plm = getMapPoint(cokp.lmid_);
                        if( plm != nullptr ) {
                            nb3dkp++;
                            double coef = 1. / cv::norm(cokp.unpx_ - kp.unpx_);
                            weights += coef;
//                            mean_z += coef * (frame.projWorldToCam(plm->getPoint())).z();
                            mean_z += coef * (frame.projWorldToCam_s(plm->getPoint())).z();//世界到左双目 = 左目到左双目*世界到左目
                        }
                    }

                    if( nb3dkp >= nbmin3dcokps ) {
                        mean_z /= weights;
                        Eigen::Vector3d predcampt = mean_z * ( kp.bv_ / kp.bv_.z() );

//                        cv::Point2f projpt = frame.projCamToRightImageDist(predcampt);
                        cv::Point2f projpt = frame.projCamToRightImageDist_s(predcampt);//左相机坐标系到右双目图像坐标系 = 右双目到右双目图像（右目到右双目*左目到右目）

//                        if( frame.isInRightImage(projpt) )
                        if( frame.isInRightImage_s(projpt) )//mono_stereo
                        {
                            v3dkps.push_back(kp.px_);
                            v3dpriors.push_back(projpt);
                            v3dkpids.push_back(kp.lmid_);
                            continue;
                        }
                    }
                }
            }
        }

        vkpids.push_back(kp.lmid_);
        vkps.push_back(kp.px_);
        vpriors.push_back(priorpt);
    }

    // Storing good tracks
    std::vector<cv::Point2f> vgoodrkps;
    std::vector<int> vgoodids;
    vgoodrkps.reserve(nbkps);
    vgoodids.reserve(nbkps);

    // 1st track 3d kps if using prior
    if( !v3dpriors.empty() )
    {
        size_t nbpyrlvl = 1;
        int nwinsize = pslamstate_->nklt_win_size_; // What about a smaller window here?

        if( vleftpyr.size() < 2*(nbpyrlvl+1) ) {
            nbpyrlvl = vleftpyr.size() / 2 - 1;
        }

        // Good / bad kps vector
        std::vector<bool> vkpstatus;

         ptracker_->fbKltTracking(  //用光流跟踪方法匹配左右目 v3dpriors和vkpstatus更新了，v3dkps没更新
                vleftpyr,
                vrightpyr,
                nwinsize,
                nbpyrlvl,
                pslamstate_->nklt_err_,
                pslamstate_->fmax_fbklt_dist_,
                v3dkps,
                v3dpriors,
                vkpstatus);

        size_t nbgood = 0;
        size_t nb3dkps = v3dkps.size();

////////////////  可视化所有1阶段跟踪到（匹配到）的点
/*        // 在这里初始化图像显示用的副本
        cv::Mat img_left = vleftpyr[0].clone();
        cv::Mat img_right = vrightpyr[0].clone();

        if (img_left.channels() == 1)
            cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
        if (img_right.channels() == 1)
            cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);

        // 在双目图像中绘制匹配点连线
        cv::Mat combined_img;
        cv::hconcat(img_left, img_right, combined_img); // 水平拼接图像

        for (size_t i = 0; i < nb3dkps; i++) {
            if (vkpstatus.at(i)) {
                cv::Point2f left_pt = v3dkps.at(i);
                cv::Point2f right_pt = v3dpriors.at(i);
                right_pt.x += img_left.cols; // 调整右目点的 x 坐标

                // 随机生成颜色
                cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);

                // 在拼接图像上绘制连线和关键点
                cv::circle(combined_img, left_pt, 3, color, -1); // 左目关键点
                cv::circle(combined_img, right_pt, 3, color, -1); // 右目关键点
                cv::line(combined_img, left_pt, right_pt, color, 1); // 连线
            }
        }

        // 显示结果
        cv::imshow("Stereo Matches 1st", combined_img);
        cv::waitKey(1); // 适当的延时*/
/////////

        for(size_t i = 0 ; i < nb3dkps  ; i++ )
        {
            if( vkpstatus.at(i) ) {
                vgoodrkps.push_back(v3dpriors.at(i));
                vgoodids.push_back(v3dkpids.at(i));
                nbgood++;
            } else {
                // If tracking failed, gonna try on full pyramid size
                // without prior for 2d kps
                vkpids.push_back(v3dkpids.at(i));
                vkps.push_back(v3dkps.at(i));
                vpriors.push_back(v3dpriors.at(i));
            }
        }

        if( pslamstate_->debug_ )
            std::cout << "\n >>> Stereo KLT Tracking on priors : " << nbgood
                      << " out of " << nb3dkps << " kps tracked!\n";
    }

    // 2nd track other kps if any
    if( !vkps.empty() )
    {
        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        ptracker_->fbKltTracking(
                vleftpyr,
                vrightpyr,
                pslamstate_->nklt_win_size_,
                pslamstate_->nklt_pyr_lvl_,
                pslamstate_->nklt_err_,
                pslamstate_->fmax_fbklt_dist_,
                vkps,
                vpriors,
                vkpstatus);

        size_t nbgood = 0;
        size_t nb2dkps = vkps.size();

////////////////  可视化所有2阶段跟踪到（匹配到）的点
/*        cv::Mat img_left = vleftpyr[0].clone();
        cv::Mat img_right = vrightpyr[0].clone();
        if (img_left.channels() == 1)
            cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
        if (img_right.channels() == 1)
            cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);
        // 在双目图像中绘制匹配点连线
        cv::Mat combined_img;
        cv::hconcat(img_left, img_right, combined_img); // 水平拼接图像
        for (size_t i = 0; i < nb2dkps; i++) {
            if (vkpstatus.at(i)) {
                cv::Point2f left_pt = vkps.at(i);
                cv::Point2f right_pt = vpriors.at(i);
                right_pt.x += img_left.cols; // 调整右目点的 x 坐标
                // 随机生成颜色
                cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
                // 在拼接图像上绘制连线和关键点
                cv::circle(combined_img, left_pt, 3, color, -1); // 左目关键点
                cv::circle(combined_img, right_pt, 3, color, -1); // 右目关键点
                cv::line(combined_img, left_pt, right_pt, color, 1); // 连线
            }
        }
        // 显示结果
        cv::imshow("Stereo Matches 2nd", combined_img);
        cv::waitKey(1); // 适当的延时*/
//////////////////////////


        for(size_t i = 0 ; i < nb2dkps  ; i++ )
        {
            if( vkpstatus.at(i) ) {
                vgoodrkps.push_back(vpriors.at(i));
                vgoodids.push_back(vkpids.at(i));
                nbgood++;
            }
        }

        if( pslamstate_->debug_ )
            std::cout << "\n >>> Stereo KLT Tracking w. no priors : " << nbgood
                      << " out of " << nb2dkps << " kps tracked!\n";
    }

    nbkps = vgoodids.size();
    size_t nbgood = 0;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> filtered_points;//add

    float epi_err = 0.;

    for( size_t i = 0; i < nbkps ; i++ )
    {
        cv::Point2f lunpx = frame.getKeypointById(vgoodids.at(i)).unpx_;
        cv::Point2f runpx = frame.pcalib_rightcam_s_->undistortImagePoint(vgoodrkps.at(i));//

        // Check epipolar consistency (same row for rectified images)
        if( pslamstate_->bdo_stereo_rect_ ) {   //未修正，但其实仿真环境也可以用。
            epi_err = fabs(lunpx.y - runpx.y);
            // Correct right kp to be on the same row
            vgoodrkps.at(i).y = lunpx.y;//直接修正右目点纵坐标，得到的结果不一定对
        }
        else {
            epi_err = MultiViewGeometry::computeSampsonDistance(frame.Frl_s_, lunpx, runpx);//过滤结果有问题
        }

        if( epi_err <= 2. )
        {
            frame.updateKeypointStereo_s(vgoodids.at(i), vgoodrkps.at(i));//
            nbgood++;
        } else {
            // 保存被过滤掉的点对
            filtered_points.emplace_back(lunpx, runpx);// add
        }
    }

////////////////  可视化所有被过滤的点
   /* if (!filtered_points.empty()) {
        // 创建拼接图像
        cv::Mat img_left = vleftpyr[0].clone();
        cv::Mat img_right = vrightpyr[0].clone();

        if (img_left.channels() == 1)
            cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
        if (img_right.channels() == 1)
            cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);

        cv::Mat combined_img;
        cv::hconcat(img_left, img_right, combined_img);

        for (const auto &pair : filtered_points) {
            cv::Point2f left_pt = pair.first;
            cv::Point2f right_pt = pair.second;
            right_pt.x += img_left.cols; // 调整右目点位置

            // 使用红色绘制被过滤掉的点对
            cv::Scalar color(0, 0, 255); // 红色
            cv::circle(combined_img, left_pt, 3, color, -1);  // 左目点
            cv::circle(combined_img, right_pt, 3, color, -1); // 右目点
            cv::line(combined_img, left_pt, right_pt, color, 1); // 连线
        }

        // 显示被过滤的点
        cv::imshow("Filtered Matches", combined_img);
        cv::waitKey(1); // 延时以刷新显示
    }*/
////////////////  可视化最终匹配点
    cv::Mat img_left = vleftpyr[0].clone();
    cv::Mat img_right = vrightpyr[0].clone();
    if (img_left.channels() == 1)
        cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
    if (img_right.channels() == 1)
        cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);
    cv::Mat combined_img;
    cv::hconcat(img_left, img_right, combined_img);
    // 从 Frame 中提取已更新的立体点
    for (const auto &pair : frame.mapkps_) {
        const Keypoint &kp = pair.second;
        if (kp.is_stereo_) {  // 仅处理通过过滤的点
            cv::Point2f lunpx = kp.unpx_;
            cv::Point2f runpx = kp.runpx_;
            runpx.x += img_left.cols; // 调整右目点的 x 坐标以适应拼接图像
            cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
            cv::circle(combined_img, lunpx, 3, color, -1);  // 左目关键点
            cv::circle(combined_img, runpx, 3, color, -1); // 右目关键点
            cv::line(combined_img, lunpx, runpx, color, 1); // 连线
        }
    }
    cv::imshow("Final Stereo Matches", combined_img);
    cv::waitKey(1);
////////////////

    if( pslamstate_->debug_ )
        std::cout << "\t>>> Nb of stereo tracks: " << nbgood << " out of " << nbkps << std::endl;

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_stereoMatching");
}

void MapManager::stereoMatching_s2(Frame &frame_s, Frame &frame, Frame &frame_r, const std::vector<cv::Mat> &vleftpyr_origin, const std::vector<cv::Mat> &vrightpyr_origin, const std::vector<cv::Mat> &vleftpyr, const std::vector<cv::Mat> &vrightpyr)
{
    //frame_s提供图像和匹配用到的图像信息，frame提供特征点，frame_s只用于提供些信息，最后结果保存到frame。
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_stereoMatching");

    // Find stereo correspondances with left kps
    auto vleftkps = frame.getKeypoints();//原图特征点
    size_t nbkps = vleftkps.size();

    // ZNCC Parameters
    size_t nmaxpyrlvl = pslamstate_->nklt_pyr_lvl_*2;
    int winsize = 7;

    float uppyrcoef = std::pow(2,pslamstate_->nklt_pyr_lvl_);
    float downpyrcoef = 1. / uppyrcoef;

    std::vector<int> v3dkpids, vkpids, voutkpids, vpriorids;
    std::vector<cv::Point2f> v3dkps, v3dpriors, vkps, vpriors;

    // First we're gonna track 3d kps on only 2 levels
    v3dkpids.reserve(frame.nb3dkps_);
    v3dkps.reserve(frame.nb3dkps_);
    v3dpriors.reserve(frame.nb3dkps_);

    // Then we'll track 2d kps on full pyramid levels
    vkpids.reserve(nbkps);
    vkps.reserve(nbkps);
    vpriors.reserve(nbkps);

    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        // Set left kp
        auto &kp = vleftkps.at(i);//左目
        Eigen::Vector3d kp_point3D((kp.px_.x - pslamstate_->cxl_)/pslamstate_->fxl_, (kp.px_.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化3d坐标
        Eigen::Vector3d kp_point3D1 = pslamstate_->R_sl.inverse() * kp_point3D;//不应该用相机坐标系下的左相机到左双目相机变化R_sl.inverse()，应该用世界坐标系下的左相机到左双目相机变化,数值上等于R_sl
//        Eigen::Vector3d kp_point3D2 = pslamstate_->R_sl.inverse() * kp_point3D;
        cv::Point2f kp_s = frame_s.projCamToImage_s(kp_point3D1);//左双目到像素*左目到左双目(纯旋转)*左目
//        cv::Point2f kp_s2 = frame_s.projCamToImage_s(kp_point3D2);//左双目到像素*左目到左双目(纯旋转)*左目
//        std::cout << "kp:" << kp.px_ << std::endl;
//        std::cout<<"kp_point3D:"<<kp_point3D<<std::endl;
//        std::cout << "kp_s:" << kp_s << std::endl;

        //如果关键点在双目区投影超出虚拟图像范围，跳过该点
        if( !frame_s.isInRightImage_s2_l(kp_s) ) {
            //过滤掉该点 感觉没有用
//            removeMapPointObs(kp.lmid_, frame.kfid_);
//            frame.removeStereoKeypointById(kp.lmid_);// add
//            frame_r.removeStereoKeypointById(kp.lmid_);
            continue;
        }

        //画三个图：在原图frame上画出kp位置，在双目区图frame_s上画出kp_s的位置，在双目区图frame_s上画出kp_s2的位置
/*        cv::Mat img_left = vleftpyr_origin[0].clone();
        cv::Mat img_left_s = vleftpyr[0].clone();
        cv::Mat img_left_s2 = vleftpyr[0].clone();
        if (img_left.channels() == 1)
            cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
        if (img_left_s.channels() == 1)
            cv::cvtColor(img_left_s, img_left_s, cv::COLOR_GRAY2BGR);
        if (img_left_s2.channels() == 1)
            cv::cvtColor(img_left_s2, img_left_s2, cv::COLOR_GRAY2BGR);
        cv::circle(img_left, kp.px_, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(img_left_s, kp_s, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(img_left_s2, kp_s2, 3, cv::Scalar(0, 0, 255), -1);
        cv::imwrite("/home/hl/project/ov2_diverg_ws/test2/kp.jpg", img_left);
        cv::imwrite("/home/hl/project/ov2_diverg_ws/test2/kp_s.jpg", img_left_s);
        cv::imwrite("/home/hl/project/ov2_diverg_ws/test2/kp_s2.jpg", img_left_s2);
        //最终证明，kp_s是对的*/

        // Set prior right kp
        cv::Point2f priorpt = kp.px_;
        cv::Point2f priorpt_s = kp_s;

        // If 3D, check if we can find a prior in right image
        if( kp.is3d_ ) {
            auto plm = getMapPoint(kp.lmid_);
            if( plm != nullptr ) {
//                cv::Point2f projpt = frame.projWorldToRightImageDist(plm->getPoint());
                cv::Point2f projpt = frame_s.projWorldToRightImageDist_s(plm->getPoint());// mono_stereo
//                if( frame.isInRightImage(projpt) ) {
                if( frame_s.isInRightImage_s2_l(projpt) ) {// mono_stereo
//                    v3dkps.push_back(kp.px_);//kp换成左双目坐标
                    v3dkps.push_back(kp_s);//kp换成左双目坐标
                    v3dpriors.push_back(projpt);
                    v3dkpids.push_back(kp.lmid_);
//                    std::cout << "find a prior in right image ----------" << std::endl;
                    continue;
                }
            } else {
                removeMapPointObs(kp.lmid_, frame.kfid_);//删除原图对应id的点
//                removeMapPointObs_s(kp.lmid_, frame.kfid_);// mono_stereo
                continue;
            }
        }

        // If stereo rect images, prior from SAD
        if( pslamstate_->bdo_stereo_rect_ ) {//暂时不改

            float xprior = -1.;
            float l1err;

            cv::Point2f pyrleftpt = kp.px_ * downpyrcoef;

            ptracker_->getLineMinSAD(vleftpyr.at(nmaxpyrlvl), vrightpyr.at(nmaxpyrlvl), pyrleftpt, winsize, xprior, l1err, true);

            xprior *= uppyrcoef;

            if( xprior >= 0 && xprior <= kp.px_.x ) {
                priorpt.x = xprior;
            }

        }
        else { // Generate prior from 3d neighbors
            const size_t nbmin3dcokps = 1;

            auto vnearkps = frame.getSurroundingKeypoints(kp);//原图上操作
//            auto vnearkps = frame.getSurroundingKeypoints_s(kp);//mono_stereo
//            std::cout<<"vnearkps.size():"<<vnearkps.size()<<std::endl;
            if( vnearkps.size() >= nbmin3dcokps )
            {
                std::vector<Keypoint> vnear3dkps;
                vnear3dkps.reserve(vnearkps.size());
                for( const auto &cokp : vnearkps ) {
                    if( cokp.is3d_ ) {
                        vnear3dkps.push_back(cokp);
                    }
                }

                if( vnear3dkps.size() >= nbmin3dcokps ) {

                    size_t nb3dkp = 0;
                    double mean_z = 0.;
                    double weights = 0.;

                    for( const auto &cokp : vnear3dkps ) {
                        auto plm = getMapPoint(cokp.lmid_);
                        if( plm != nullptr ) {
                            nb3dkp++;
                            double coef = 1. / cv::norm(cokp.unpx_ - kp.unpx_);//计算邻域每个 3D 特征点与当前特征点之间的距离，并利用这个距离作为权重，计算加权平均深度
                            weights += coef;
                            mean_z += coef * (frame.projWorldToCam(plm->getPoint())).z();
//                            mean_z += coef * (frame.projWorldToCam_s(plm->getPoint())).z();// 所有邻域特征点的平均深度 世界到左双目 = 左目到左双目*世界到左目
                        }
                    }

                    if( nb3dkp >= nbmin3dcokps ) {
                        mean_z /= weights;
                        Eigen::Vector3d predcampt = mean_z * ( kp.bv_ / kp.bv_.z() );//左目 用的参数都是左目的

//                        cv::Point2f projpt = frame.projCamToRightImageDist(predcampt);
                        cv::Point2f projpt = frame_s.projCamToRightImageDist_s(predcampt);//左相机坐标系到右双目图像坐标系 = 右双目到右双目图像（右目到右双目*左目到右目）

//                        if( frame.isInRightImage(projpt) )
                        if( frame_s.isInRightImage_s2_l(projpt) )//mono_stereo
                        {
//                            v3dkps.push_back(kp.px_);//应该存左双目区点坐标，不是kp，是kp的投影
                            v3dkps.push_back(kp_s);//kp换成左双目坐标
                            v3dpriors.push_back(projpt);//右双目区点
                            v3dkpids.push_back(kp.lmid_);
                            continue;
                        }
                    }
                }
            }
        }

        vkpids.push_back(kp.lmid_);
//        vkps.push_back(kp.px_);
        vkps.push_back(kp_s);
//        vpriors.push_back(priorpt);
        vpriors.push_back(priorpt_s);
    }

    // Storing good tracks
    std::vector<cv::Point2f> vgoodrkps;
    std::vector<int> vgoodids;
    vgoodrkps.reserve(nbkps);
    vgoodids.reserve(nbkps);

    // 1st track 3d kps if using prior
    if( !v3dpriors.empty() )
    {
        size_t nbpyrlvl = 1;
        int nwinsize = pslamstate_->nklt_win_size_; // What about a smaller window here?

        if( vleftpyr.size() < 2*(nbpyrlvl+1) ) {
            nbpyrlvl = vleftpyr.size() / 2 - 1;
        }

        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        ptracker_->fbKltTracking(  //用光流跟踪方法匹配左右目 v3dpriors和vkpstatus更新了，v3dkps没更新
                vleftpyr,
                vrightpyr,
                nwinsize,
                nbpyrlvl,
                pslamstate_->nklt_err_,
                pslamstate_->fmax_fbklt_dist_,
                v3dkps,
                v3dpriors,
                vkpstatus);

        size_t nbgood = 0;
        size_t nb3dkps = v3dkps.size();

////////////////  可视化所有1阶段跟踪到（匹配到）的点
/*        // 在这里初始化图像显示用的副本
        cv::Mat img_left = vleftpyr[0].clone();
        cv::Mat img_right = vrightpyr[0].clone();

        if (img_left.channels() == 1)
            cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
        if (img_right.channels() == 1)
            cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);

        // 在双目图像中绘制匹配点连线
        cv::Mat combined_img;
        cv::hconcat(img_left, img_right, combined_img); // 水平拼接图像

        for (size_t i = 0; i < nb3dkps; i++) {
            if (vkpstatus.at(i)) {
                cv::Point2f left_pt = v3dkps.at(i);
                cv::Point2f right_pt = v3dpriors.at(i);
                right_pt.x += img_left.cols; // 调整右目点的 x 坐标

                // 随机生成颜色
                cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);

                // 在拼接图像上绘制连线和关键点
                cv::circle(combined_img, left_pt, 3, color, -1); // 左目关键点
                cv::circle(combined_img, right_pt, 3, color, -1); // 右目关键点
                cv::line(combined_img, left_pt, right_pt, color, 1); // 连线
            }
        }

        // 显示结果
        cv::imshow("Stereo Matches 1st", combined_img);
        cv::waitKey(1); // 适当的延时*/
/////////

        for(size_t i = 0 ; i < nb3dkps  ; i++ )
        {
            if( vkpstatus.at(i) ) {
                vgoodrkps.push_back(v3dpriors.at(i));
                vgoodids.push_back(v3dkpids.at(i));
                nbgood++;
            } else {
                // If tracking failed, gonna try on full pyramid size
                // without prior for 2d kps
                vkpids.push_back(v3dkpids.at(i));
                vkps.push_back(v3dkps.at(i));
                vpriors.push_back(v3dpriors.at(i));
            }
        }

        if( pslamstate_->debug_ )
            std::cout << "\n >>> Stereo KLT Tracking on priors : " << nbgood
                      << " out of " << nb3dkps << " kps tracked!\n";
    }

    // 2nd track other kps if any
    if( !vkps.empty() )
    {
        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        ptracker_->fbKltTracking(
                vleftpyr,
                vrightpyr,
                pslamstate_->nklt_win_size_,
                pslamstate_->nklt_pyr_lvl_,
                pslamstate_->nklt_err_,
                pslamstate_->fmax_fbklt_dist_,
                vkps,
                vpriors,
                vkpstatus);

        size_t nbgood = 0;
        size_t nb2dkps = vkps.size();

////////////////  可视化所有2阶段跟踪到（匹配到）的点
/*        cv::Mat img_left = vleftpyr[0].clone();
        cv::Mat img_right = vrightpyr[0].clone();
        if (img_left.channels() == 1)
            cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
        if (img_right.channels() == 1)
            cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);
        // 在双目图像中绘制匹配点连线
        cv::Mat combined_img;
        cv::hconcat(img_left, img_right, combined_img); // 水平拼接图像
        for (size_t i = 0; i < nb2dkps; i++) {
            if (vkpstatus.at(i)) {
                cv::Point2f left_pt = vkps.at(i);
                cv::Point2f right_pt = vpriors.at(i);
                right_pt.x += img_left.cols; // 调整右目点的 x 坐标
                // 随机生成颜色
                cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
                // 在拼接图像上绘制连线和关键点
                cv::circle(combined_img, left_pt, 3, color, -1); // 左目关键点
                cv::circle(combined_img, right_pt, 3, color, -1); // 右目关键点
                cv::line(combined_img, left_pt, right_pt, color, 1); // 连线
            }
        }
        // 显示结果
        cv::imshow("Stereo Matches 2nd", combined_img);
//        cv::imwrite("/home/hl/project/ov2_diverg_ws/test/2nd-1.png", combined_img);
        cv::waitKey(1); // 适当的延时*/
//////////////////////////


        for(size_t i = 0 ; i < nb2dkps  ; i++ )
        {
            if( vkpstatus.at(i) ) {
                vgoodrkps.push_back(vpriors.at(i));
                vgoodids.push_back(vkpids.at(i));
                nbgood++;
            }
        }

        if( pslamstate_->debug_ )
            std::cout << "\n >>> Stereo KLT Tracking w. no priors : " << nbgood
                      << " out of " << nb2dkps << " kps tracked!\n";
    }

    nbkps = vgoodids.size();
    size_t nbgood = 0;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> filtered_points;//add

    float epi_err = 0.;

    for( size_t i = 0; i < nbkps ; i++ )
    {
//        cv::Point2f lunpx = frame.getKeypointById(vgoodids.at(i)).unpx_;//
//        cv::Point2f runpx = frame.pcalib_rightcam_s_->undistortImagePoint(vgoodrkps.at(i));//
        cv::Point2f good_pt1 = frame.getKeypointById(vgoodids.at(i)).unpx_;
        Eigen::Vector3d good_pt1_point3D((good_pt1.x - pslamstate_->cxl_)/pslamstate_->fxl_, (good_pt1.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化3d坐标
        Eigen::Vector3d good_pt1_point3D1 = pslamstate_->R_sl.inverse() * good_pt1_point3D;
        cv::Point2f lunpx = frame_s.projCamToImage_s(good_pt1_point3D1);//世界到左双目*左目到世界(归一化)
        cv::Point2f runpx = frame_s.pcalib_rightcam_s_->undistortImagePoint(vgoodrkps.at(i));//

        // Check epipolar consistency (same row for rectified images)
        if( pslamstate_->bdo_stereo_rect_ ) {   //未修正，但其实仿真环境也可以用。
            epi_err = fabs(lunpx.y - runpx.y);
            // Correct right kp to be on the same row
            vgoodrkps.at(i).y = lunpx.y;//直接修正右目点纵坐标，得到的结果不一定对
        }
        else {
            epi_err = MultiViewGeometry::computeSampsonDistance(frame_s.Frl_s_, lunpx, runpx);
        }

        if( epi_err <= 2. )
        {//左右目下的地图点都更新
//            frame.updateKeypointStereo_s(vgoodids.at(i), vgoodrkps.at(i));//注：存的右目的匹配点坐标，是双目区的坐标
//            frame_r.updateKeypointStereo_s(vgoodids.at(i), vgoodrkps.at(i));//注：存的右目的匹配点坐标，是双目区的坐标
            //改为：存原图点坐标,从右双目区坐标，变成右目坐标
            cv::Point2f kp_rs = vgoodrkps.at(i);
            double cx = pcurframe_ls_->pcalib_rightcam_s_->cx_;
            double cy = pcurframe_ls_->pcalib_rightcam_s_->cy_;
            double fx = pcurframe_ls_->pcalib_rightcam_s_->fx_;
            double fy = pcurframe_ls_->pcalib_rightcam_s_->fy_;
            Eigen::Vector3d kp_point3D((kp_rs.x - cx)/fx, (kp_rs.y - cy)/fy, 1.0);//归一化3d坐标
            Eigen::Vector3d kp_point3D1 = pslamstate_->R_sr * kp_point3D;
            cv::Point2f kp_r = frame.projCamToImage(kp_point3D1);
            frame.updateKeypointStereo(vgoodids.at(i), kp_r);//存的右目的匹配点坐标，是原图的坐标
//            frame_r.updateKeypointStereo(vgoodids.at(i), kp_r);//存的右目的匹配点坐标，是原图的坐标

            //可视化验证，左目区、左双目区点转换
            /*//画三个图：在原图frame上画出kp位置，在双目区图frame_s上画出kp_s的位置，在双目区图frame_s上画出kp_s2的位置
            cv::Mat img_right = vrightpyr_origin[0].clone();
            cv::Mat img_right_s = vrightpyr[0].clone();
            if (img_right.channels() == 1)
                cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);
            if (img_right_s.channels() == 1)
                cv::cvtColor(img_right_s, img_right_s, cv::COLOR_GRAY2BGR);
            cv::circle(img_right, kp_r, 3, cv::Scalar(0, 0, 255), -1);
            cv::circle(img_right_s, kp_rs, 3, cv::Scalar(0, 0, 255), -1);
            cv::imwrite("/home/hl/project/ov2_diverg_ws/test/kp_rs.jpg", img_right);
            cv::imwrite("/home/hl/project/ov2_diverg_ws/test/kp_r.jpg", img_right_s);
            //最终证明是对的*/

            //可视化验证原图匹配点坐标是对的
/*            cv::Mat img_left = vleftpyr_origin[0].clone();
            cv::Mat img_right = vrightpyr_origin[0].clone();
            if (img_left.channels() == 1)
                cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
            if (img_right.channels() == 1)
                cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);
            cv::circle(img_left, good_pt1, 3, cv::Scalar(0, 0, 255), -1);
            cv::circle(img_right, kp_r, 3, cv::Scalar(0, 0, 255), -1);
            //拼接成一张图，连线
            cv::Mat combined_img;
            cv::hconcat(img_left, img_right, combined_img);
            cv::line(combined_img, good_pt1, cv::Point(kp_r.x + img_left.cols, kp_r.y) , cv::Scalar(0, 0, 255), 1);
            cv::imwrite("/home/hl/project/ov2_diverg_ws/test/match_origin.jpg", combined_img);
            //最终证明是对的*/

            nbgood++;
        } else {
            // 保存被过滤掉的点对
            filtered_points.emplace_back(lunpx, runpx);// add
        }
    }

////////////////  可视化所有被过滤的点
/*     if (!filtered_points.empty()) {
         // 创建拼接图像
         cv::Mat img_left = vleftpyr[0].clone();
         cv::Mat img_right = vrightpyr[0].clone();

         if (img_left.channels() == 1)
             cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
         if (img_right.channels() == 1)
             cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);

         cv::Mat combined_img;
         cv::hconcat(img_left, img_right, combined_img);

         for (const auto &pair : filtered_points) {
             cv::Point2f left_pt = pair.first;
             cv::Point2f right_pt = pair.second;
             right_pt.x += img_left.cols; // 调整右目点位置

             // 使用红色绘制被过滤掉的点对
             cv::Scalar color(0, 0, 255); // 红色
             cv::circle(combined_img, left_pt, 3, color, -1);  // 左目点
             cv::circle(combined_img, right_pt, 3, color, -1); // 右目点
             cv::line(combined_img, left_pt, right_pt, color, 1); // 连线
         }

         // 显示被过滤的点
         cv::imshow("Filtered Matches", combined_img);
//         cv::imwrite("/home/hl/project/ov2_diverg_ws/test/Filtered.png", combined_img);
         cv::waitKey(1); // 延时以刷新显示
     }//*/
////////////////  可视化最终匹配点
    cv::Mat img_left = vleftpyr[0].clone();
    cv::Mat img_right = vrightpyr[0].clone();
    if (img_left.channels() == 1)
        cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
    if (img_right.channels() == 1)
        cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);
    cv::Mat combined_img;
    cv::hconcat(img_left, img_right, combined_img);
    // 从 Frame 中提取已更新的立体点
    for (const auto &pair : frame.mapkps_) {//左目点
        const Keypoint &kp = pair.second;
        if (kp.is_stereo_) {  // 仅处理通过过滤的点
//            cv::Point2f lunpx = kp.unpx_;//改为双目区点坐标
            Eigen::Vector3d kp_point3D((kp.unpx_.x - pslamstate_->cxl_)/pslamstate_->fxl_, (kp.unpx_.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化3d坐标
            Eigen::Vector3d kp_point3D1 = pslamstate_->R_sl.inverse() * kp_point3D;
            cv::Point2f lunpx = frame_s.projCamToImage_s(kp_point3D1);
            Eigen::Vector3d kpr_point3D((kp.runpx_.x - pslamstate_->cxl_)/pslamstate_->fxl_, (kp.runpx_.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化3d坐标
            Eigen::Vector3d kpr_point3D1 = pslamstate_->R_sr.inverse() * kpr_point3D;
            cv::Point2f runpx = frame_s.projCamToImage_s(kpr_point3D1);
            runpx.x += img_left.cols; // 调整右目点的 x 坐标以适应拼接图像
            cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
            cv::circle(combined_img, lunpx, 3, color, -1);  // 左目关键点
            cv::circle(combined_img, runpx, 3, color, -1); // 右目关键点
            cv::line(combined_img, lunpx, runpx, color, 1); // 连线
        }
    }
    cv::imshow("Final Stereo Matches stereo left", combined_img);
//    cv::imwrite("/home/hl/project/ov2_diverg_ws/test2/Final_s.png", combined_img);
    cv::waitKey(1);
////////////////
    cv::Mat img_left2 = vleftpyr_origin[0].clone();
    cv::Mat img_right2 = vrightpyr_origin[0].clone();
    if (img_left2.channels() == 1)
        cv::cvtColor(img_left2, img_left2, cv::COLOR_GRAY2BGR);
    if (img_right2.channels() == 1)
        cv::cvtColor(img_right2, img_right2, cv::COLOR_GRAY2BGR);
    cv::Mat combined_img2;
    cv::hconcat(img_left2, img_right2, combined_img2);
    // 从 Frame 中提取已更新的立体点
    for (const auto &pair : frame.mapkps_) {
        const Keypoint &kp = pair.second;
        if (kp.is_stereo_) {  // 仅处理通过过滤的点
            cv::Point2f lunpx = kp.unpx_;
            cv::Point2f runpx = kp.runpx_;
            runpx.x += img_left2.cols; // 调整右目点的 x 坐标以适应拼接图像
            cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
            cv::circle(combined_img2, lunpx, 3, color, -1);  // 左目关键点
            cv::circle(combined_img2, runpx, 3, color, -1); // 右目关键点
            cv::line(combined_img2, lunpx, runpx, color, 1); // 连线
        }
    }
//    cv::imshow("Final Stereo Matches original left", combined_img2);
//    cv::imwrite("/home/hl/project/ov2_diverg_ws/test2/Final_o.png", combined_img2);
//    cv::waitKey(1);
////////////////

    if( pslamstate_->debug_ )
        std::cout << "\t>>> Nb of stereo tracks: " << nbgood << " out of " << nbkps << std::endl;

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_stereoMatching");
}

void MapManager::stereoMatching_s2_r(Frame &frame_s, Frame &frame, Frame &frame_r, const std::vector<cv::Mat> &vleftpyr_origin, const std::vector<cv::Mat> &vrightpyr_origin, const std::vector<cv::Mat> &vleftpyr, const std::vector<cv::Mat> &vrightpyr)
{
    //frame_s提供图像和匹配用到的图像信息，frame提供特征点，frame_s只用于提供些信息，最后结果保存到frame。
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_stereoMatching");

    // Find stereo correspondances with left kps
//    auto vleftkps = frame.getKeypoints();//原图特征点
//    size_t nbkps = vleftkps.size();
    auto vrightkps = frame_r.getKeypoints();//原图特征点
    size_t nbkps = vrightkps.size();
//    std::cout << "nbkps:" << nbkps << std::endl;

    // ZNCC Parameters
    size_t nmaxpyrlvl = pslamstate_->nklt_pyr_lvl_*2;
    int winsize = 7;

    float uppyrcoef = std::pow(2,pslamstate_->nklt_pyr_lvl_);
    float downpyrcoef = 1. / uppyrcoef;

    std::vector<int> v3dkpids, vkpids, voutkpids, vpriorids;
    std::vector<cv::Point2f> v3dkps, v3dpriors, vkps, vpriors;

    // First we're gonna track 3d kps on only 2 levels
    v3dkpids.reserve(frame_r.nb3dkps_);
    v3dkps.reserve(frame_r.nb3dkps_);
    v3dpriors.reserve(frame_r.nb3dkps_);

    // Then we'll track 2d kps on full pyramid levels
    vkpids.reserve(nbkps);
    vkps.reserve(nbkps);
    vpriors.reserve(nbkps);

    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        // Set right kp
        auto &kp = vrightkps.at(i);//右目
        Eigen::Vector3d kp_point3D((kp.px_.x - pslamstate_->cxl_)/pslamstate_->fxl_, (kp.px_.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化3d坐标
        Eigen::Vector3d kp_point3D1 = pslamstate_->R_sr.inverse() * kp_point3D;//不应该用相机坐标系下的左相机到左双目相机变化R_sl.inverse()，应该用世界坐标系下的左相机到左双目相机变化,数值上等于R_sl
//        Eigen::Vector3d kp_point3D2 = pslamstate_->R_sl.inverse() * kp_point3D;
        cv::Point2f kp_s = frame_s.projCamToImage_s(kp_point3D1);//右双目到像素*右目到右双目(纯旋转)*左目
//        cv::Point2f kp_s2 = frame_s.projCamToImage_s(kp_point3D2);
//        std::cout << "kp:" << kp.px_ << std::endl;
//        std::cout<<"kp_point3D:"<<kp_point3D<<std::endl;
//        std::cout << "kp_s:" << kp_s << std::endl;

        //如果关键点在双目区投影超出虚拟图像范围，跳过该点
        if( !frame_s.isInRightImage_s2_r(kp_s) ) {
            //过滤掉该点 感觉没有用
//            removeMapPointObs(kp.lmid_, frame.kfid_);
//            frame.removeStereoKeypointById(kp.lmid_);// add
//            frame_r.removeStereoKeypointById(kp.lmid_);
            continue;
        }

/*        //画三个图：在原图frame上画出kp位置，在双目区图frame_s上画出kp_s的位置，在双目区图frame_s上画出kp_s2的位置
        cv::Mat img_right = vrightpyr_origin[0].clone();
        cv::Mat img_right_s = vrightpyr[0].clone();
        if (img_right.channels() == 1)
            cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);
        if (img_right_s.channels() == 1)
            cv::cvtColor(img_right_s, img_right_s, cv::COLOR_GRAY2BGR);
        cv::circle(img_right, kp.px_, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(img_right_s, kp_s, 3, cv::Scalar(0, 0, 255), -1);
//        cv::circle(img_left_s2, kp_s2, 3, cv::Scalar(0, 0, 255), -1);
        cv::imshow("kp", img_right);
        cv::imshow("kp_s", img_right_s);
        cv::imwrite("/home/hl/project/ov2_diverg_ws/test2/kp.jpg", img_right);
        cv::imwrite("/home/hl/project/ov2_diverg_ws/test2/kp_s.jpg", img_right_s);
        cv::waitKey(1);*/

        // Set prior right kp
        cv::Point2f priorpt = kp.px_;
        cv::Point2f priorpt_s = kp_s;

        // If 3D, check if we can find a prior in left image
        if( kp.is3d_ ) {
            auto plm = getMapPoint(kp.lmid_);
            if( plm != nullptr ) {
//                cv::Point2f projpt = frame.projWorldToRightImageDist(plm->getPoint());
//                cv::Point2f projpt = frame_s.projWorldToRightImageDist_s(plm->getPoint());// mono_stereo
                cv::Point2f projpt = frame_s.projWorldToImageDist_s(plm->getPoint());// mono_stereo
//                if( frame.isInRightImage(projpt) ) {
                if( frame_s.isInRightImage_s2_r(projpt) ) {// mono_stereo
//                    v3dkps.push_back(kp.px_);//kp换成右双目坐标
                    v3dkps.push_back(kp_s);//kp换成右双目坐标
                    v3dpriors.push_back(projpt);
                    v3dkpids.push_back(kp.lmid_);
//                    std::cout << "find a prior in right image ----------" << std::endl;
                    continue;
                }
            } else {
                removeMapPointObs(kp.lmid_, frame_r.kfid_);//删除原图对应id的点
//                removeMapPointObs_s(kp.lmid_, frame.kfid_);// mono_stereo
                continue;
            }
        }

        // If stereo rect images, prior from SAD
        if( pslamstate_->bdo_stereo_rect_ ) {//暂时不改

            float xprior = -1.;
            float l1err;

//            cv::Point2f pyrleftpt = kp.px_ * downpyrcoef;
            cv::Point2f pyrrightpt = kp.px_ * downpyrcoef;

//            ptracker_->getLineMinSAD(vleftpyr.at(nmaxpyrlvl), vrightpyr.at(nmaxpyrlvl), pyrleftpt, winsize, xprior, l1err, true);
            ptracker_->getLineMinSAD(vleftpyr.at(nmaxpyrlvl), vrightpyr.at(nmaxpyrlvl), pyrrightpt, winsize, xprior, l1err, true);

            xprior *= uppyrcoef;

            if( xprior >= 0 && xprior <= kp.px_.x ) {
                priorpt.x = xprior;
            }

        }
        else { // Generate prior from 3d neighbors
            const size_t nbmin3dcokps = 1;

            auto vnearkps = frame.getSurroundingKeypoints(kp);//原图上操作
//            auto vnearkps = frame.getSurroundingKeypoints_s(kp);//mono_stereo
//            std::cout<<"vnearkps.size():"<<vnearkps.size()<<std::endl;
            if( vnearkps.size() >= nbmin3dcokps )
            {
                std::vector<Keypoint> vnear3dkps;
                vnear3dkps.reserve(vnearkps.size());
                for( const auto &cokp : vnearkps ) {
                    if( cokp.is3d_ ) {
                        vnear3dkps.push_back(cokp);
                    }
                }

                if( vnear3dkps.size() >= nbmin3dcokps ) {

                    size_t nb3dkp = 0;
                    double mean_z = 0.;
                    double weights = 0.;

                    for( const auto &cokp : vnear3dkps ) {
                        auto plm = getMapPoint(cokp.lmid_);
                        if( plm != nullptr ) {
                            nb3dkp++;
                            double coef = 1. / cv::norm(cokp.unpx_ - kp.unpx_);//计算邻域每个 3D 特征点与当前特征点之间的距离，并利用这个距离作为权重，计算加权平均深度
                            weights += coef;
                            mean_z += coef * (frame.projWorldToCam(plm->getPoint())).z();
//                            mean_z += coef * (frame.projWorldToCam_s(plm->getPoint())).z();// 所有邻域特征点的平均深度 世界到左双目 = 左目到左双目*世界到左目
                        }
                    }

                    if( nb3dkp >= nbmin3dcokps ) {
                        mean_z /= weights;
                        Eigen::Vector3d predcampt = mean_z * ( kp.bv_ / kp.bv_.z() );//右目 用的参数都是右目的

//                        cv::Point2f projpt = frame.projCamToRightImageDist(predcampt);
//                        cv::Point2f projpt = frame_s.projCamToRightImageDist_s(predcampt);//左相机坐标系到右双目图像坐标系 = 右双目到右双目图像（右目到右双目*左目到右目）
                        cv::Point2f projpt = frame_s.projCamToImageDist_s(predcampt);//右相机坐标系到左双目图像坐标系 = 左双目到左双目图像（左目到左双目*右目到左目）

//                        if( frame.isInRightImage(projpt) )
                        if( frame_s.isInRightImage_s2_r(projpt) )//mono_stereo
                        {
//                            v3dkps.push_back(kp.px_);//应该存左双目区点坐标，不是kp，是kp的投影
                            v3dkps.push_back(kp_s);//kp换成左双目坐标
                            v3dpriors.push_back(projpt);//右双目区点
                            v3dkpids.push_back(kp.lmid_);
                            continue;
                        }
                    }
                }
            }
        }

        vkpids.push_back(kp.lmid_);
//        vkps.push_back(kp.px_);
        vkps.push_back(kp_s);
//        vpriors.push_back(priorpt);
        vpriors.push_back(priorpt_s);
    }

    // Storing good tracks
//    std::vector<cv::Point2f> vgoodrkps;
    std::vector<cv::Point2f> vgoodlkps;
    std::vector<int> vgoodids;
//    vgoodrkps.reserve(nbkps);
    vgoodlkps.reserve(nbkps);
    vgoodids.reserve(nbkps);

    // 1st track 3d kps if using prior
    if( !v3dpriors.empty() )
    {
        size_t nbpyrlvl = 1;
        int nwinsize = pslamstate_->nklt_win_size_; // What about a smaller window here?

        if( vrightpyr.size() < 2*(nbpyrlvl+1) ) {
            nbpyrlvl = vrightpyr.size() / 2 - 1;
        }

        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        ptracker_->fbKltTracking(  //用光流跟踪方法匹配左右目 v3dpriors和vkpstatus更新了，v3dkps没更新
                vrightpyr,//左右反过来
                vleftpyr,
                nwinsize,
                nbpyrlvl,
                pslamstate_->nklt_err_,
                pslamstate_->fmax_fbklt_dist_,
                v3dkps,
                v3dpriors,
                vkpstatus);

        size_t nbgood = 0;
        size_t nb3dkps = v3dkps.size();

////////////////  可视化所有1阶段跟踪到（匹配到）的点
/*        // 在这里初始化图像显示用的副本
        cv::Mat img_left = vleftpyr[0].clone();
        cv::Mat img_right = vrightpyr[0].clone();

        if (img_left.channels() == 1)
            cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
        if (img_right.channels() == 1)
            cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);

        // 在双目图像中绘制匹配点连线
        cv::Mat combined_img;
        cv::hconcat(img_left, img_right, combined_img); // 水平拼接图像

        for (size_t i = 0; i < nb3dkps; i++) {
            if (vkpstatus.at(i)) {
//                cv::Point2f left_pt = v3dkps.at(i);
//                cv::Point2f right_pt = v3dpriors.at(i);
                cv::Point2f right_pt = v3dkps.at(i);
                cv::Point2f left_pt = v3dpriors.at(i);
                right_pt.x += img_left.cols; // 调整右目点的 x 坐标

                // 随机生成颜色
                cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);

                // 在拼接图像上绘制连线和关键点
                cv::circle(combined_img, left_pt, 3, color, -1); // 左目关键点
                cv::circle(combined_img, right_pt, 3, color, -1); // 右目关键点
                cv::line(combined_img, left_pt, right_pt, color, 1); // 连线
            }
        }

        // 显示结果
        cv::imshow("Stereo Matches 1st", combined_img);
        cv::waitKey(1); // 适当的延时*/
/////////

        for(size_t i = 0 ; i < nb3dkps  ; i++ )
        {
            if( vkpstatus.at(i) ) {
//                vgoodrkps.push_back(v3dpriors.at(i));
                vgoodlkps.push_back(v3dpriors.at(i));
                vgoodids.push_back(v3dkpids.at(i));
                nbgood++;
            } else {
                // If tracking failed, gonna try on full pyramid size
                // without prior for 2d kps
                vkpids.push_back(v3dkpids.at(i));
                vkps.push_back(v3dkps.at(i));
                vpriors.push_back(v3dpriors.at(i));
            }
        }

        if( pslamstate_->debug_ )
            std::cout << "\n >>> Stereo KLT Tracking on priors : " << nbgood
                      << " out of " << nb3dkps << " kps tracked!\n";
    }

    // 2nd track other kps if any
    if( !vkps.empty() )
    {
        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        ptracker_->fbKltTracking(
                vrightpyr,//左右反过来
                vleftpyr,
                pslamstate_->nklt_win_size_,
                pslamstate_->nklt_pyr_lvl_,
                pslamstate_->nklt_err_,
                pslamstate_->fmax_fbklt_dist_,
                vkps,
                vpriors,
                vkpstatus);

        size_t nbgood = 0;
        size_t nb2dkps = vkps.size();

////////////////  可视化所有2阶段跟踪到（匹配到）的点
/*        cv::Mat img_left = vleftpyr[0].clone();
        cv::Mat img_right = vrightpyr[0].clone();
        if (img_left.channels() == 1)
            cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
        if (img_right.channels() == 1)
            cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);
        // 在双目图像中绘制匹配点连线
        cv::Mat combined_img;
        cv::hconcat(img_left, img_right, combined_img); // 水平拼接图像
        for (size_t i = 0; i < nb2dkps; i++) {
            if (vkpstatus.at(i)) {
//                cv::Point2f left_pt = vkps.at(i);
//                cv::Point2f right_pt = vpriors.at(i);
                cv::Point2f right_pt = vkps.at(i);
                cv::Point2f left_pt = vpriors.at(i);
                right_pt.x += img_left.cols; // 调整右目点的 x 坐标
                // 随机生成颜色
                cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
                // 在拼接图像上绘制连线和关键点
                cv::circle(combined_img, left_pt, 3, color, -1); // 左目关键点
                cv::circle(combined_img, right_pt, 3, color, -1); // 右目关键点
                cv::line(combined_img, left_pt, right_pt, color, 1); // 连线
            }
        }
        // 显示结果
        cv::imshow("Stereo Matches 2nd", combined_img);
        cv::imwrite("/home/hl/project/ov2_diverg_ws/test2/2nd-1.png", combined_img);
        cv::waitKey(1);*/
//////////////////////////


        for(size_t i = 0 ; i < nb2dkps  ; i++ )
        {
            if( vkpstatus.at(i) ) {
//                vgoodrkps.push_back(vpriors.at(i));
                vgoodlkps.push_back(vpriors.at(i));//双目区点坐标
                vgoodids.push_back(vkpids.at(i));
                nbgood++;
            }
        }

        if( pslamstate_->debug_ )
            std::cout << "\n >>> Stereo KLT Tracking w. no priors : " << nbgood
                      << " out of " << nb2dkps << " kps tracked!\n";
    }

    nbkps = vgoodids.size();
    size_t nbgood = 0;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> filtered_points;//add

    float epi_err = 0.;

    for( size_t i = 0; i < nbkps ; i++ )
    {
//        //cv::Point2f lunpx = frame.getKeypointById(vgoodids.at(i)).unpx_;//
//        //cv::Point2f runpx = frame.pcalib_rightcam_s_->undistortImagePoint(vgoodrkps.at(i));//
        cv::Point2f good_pt1 = frame_r.getKeypointById(vgoodids.at(i)).unpx_;
        Eigen::Vector3d good_pt1_point3D((good_pt1.x - pslamstate_->cxl_)/pslamstate_->fxl_, (good_pt1.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化3d坐标
        Eigen::Vector3d good_pt1_point3D1 = pslamstate_->R_sr.inverse() * good_pt1_point3D;
        cv::Point2f runpx = frame_s.projCamToImage_s(good_pt1_point3D1);//世界到左双目*左目到世界(归一化)
        cv::Point2f lunpx = frame_s.pcalib_leftcam_s_->undistortImagePoint(vgoodlkps.at(i));//

//        cv::Point2f good_pt1 = frame_r.getKeypointById(vgoodids.at(i)).unpx_;
//        Eigen::Vector3d good_pt1_point3D((good_pt1.x - pslamstate_->cxl_)/pslamstate_->fxl_, (good_pt1.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化3d坐标
//        Eigen::Vector3d good_pt1_point3D1 = pslamstate_->R_sr.inverse() * good_pt1_point3D;
//        cv::Point2f runpx = frame_s.projCamToImage_s(good_pt1_point3D1);//世界到左双目*左目到世界(归一化)
//        cv::Point2f lunpx = frame_s.pcalib_rightcam_s_->undistortImagePoint(vgoodrkps.at(i));//

        // Check epipolar consistency (same row for rectified images)
        if( pslamstate_->bdo_stereo_rect_ ) {   //未修正，但其实仿真环境也可以用。
            epi_err = fabs(lunpx.y - runpx.y);
//            // Correct right kp to be on the same row
//            vgoodrkps.at(i).y = lunpx.y;//直接修正右目点纵坐标，得到的结果不一定对
            // Correct left kp to be on the same row
            vgoodlkps.at(i).y = runpx.y;//直接修正右目点纵坐标，得到的结果不一定对
        }
        else {
            epi_err = MultiViewGeometry::computeSampsonDistance(frame_s.Frl_s_, lunpx, runpx);
        }

        if( epi_err <= 2. )
        {//左右目下的地图点都更新
//            frame.updateKeypointStereo_s(vgoodids.at(i), vgoodrkps.at(i));//注：存的右目的匹配点坐标，是双目区的坐标
//            frame_r.updateKeypointStereo_s(vgoodids.at(i), vgoodrkps.at(i));//注：存的右目的匹配点坐标，是双目区的坐标
            //改为：存原图点坐标,从左双目区坐标，变成左目坐标
            cv::Point2f kp_ls = vgoodlkps.at(i);
//            cv::Point2f kp_ls = lunpx;//same
            double cx = pcurframe_ls_->pcalib_leftcam_s_->cx_;
            double cy = pcurframe_ls_->pcalib_leftcam_s_->cy_;
            double fx = pcurframe_ls_->pcalib_leftcam_s_->fx_;
            double fy = pcurframe_ls_->pcalib_leftcam_s_->fy_;
            Eigen::Vector3d kp_point3D((kp_ls.x - cx)/fx, (kp_ls.y - cy)/fy, 1.0);//归一化3d坐标
            Eigen::Vector3d kp_point3D1 = pslamstate_->R_sl * kp_point3D;
            cv::Point2f kp_l = frame.projCamToImage(kp_point3D1);
//            frame.updateKeypointStereo(vgoodids.at(i), kp_r);//存的右目的匹配点坐标，是原图的坐标
            frame_r.updateKeypointStereo(vgoodids.at(i), kp_l);//runpx...存的左目的匹配点坐标，是原图的坐标

            //可视化验证，左目区、左双目区点转换
/*            //画三个图：在原图frame上画出kp位置，在双目区图frame_s上画出kp_s的位置，在双目区图frame_s上画出kp_s2的位置
            cv::Mat img_left_test = vleftpyr_origin[0].clone();
            cv::Mat img_left_test_s = vleftpyr[0].clone();
            if (img_left_test.channels() == 1)
                cv::cvtColor(img_left_test, img_left_test, cv::COLOR_GRAY2BGR);
            if (img_left_test_s.channels() == 1)
                cv::cvtColor(img_left_test_s, img_left_test_s, cv::COLOR_GRAY2BGR);
            cv::circle(img_left_test, kp_l, 3, cv::Scalar(0, 0, 255), -1);
            cv::circle(img_left_test_s, kp_ls, 3, cv::Scalar(0, 0, 255), -1);
            cv::imwrite("/home/hl/project/ov2_diverg_ws/test2/kp_ls.jpg", img_left_test_s);
            cv::imwrite("/home/hl/project/ov2_diverg_ws/test2/kp_l.jpg", img_left_test);
            //最终证明是对的*/

            //可视化验证原图匹配点坐标是对的
/*            cv::Mat img_left = vleftpyr_origin[0].clone();
            cv::Mat img_right = vrightpyr_origin[0].clone();
            if (img_left.channels() == 1)
                cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
            if (img_right.channels() == 1)
                cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);
            cv::circle(img_left, kp_l, 3, cv::Scalar(0, 0, 255), -1);
            cv::circle(img_right, good_pt1, 3, cv::Scalar(0, 0, 255), -1);
            //拼接成一张图，连线
            cv::Mat combined_img;
            cv::hconcat(img_left, img_right, combined_img);
            cv::line(combined_img, kp_l, cv::Point(good_pt1.x + img_left.cols, good_pt1.y) , cv::Scalar(0, 0, 255), 1);
            cv::imwrite("/home/hl/project/ov2_diverg_ws/test2/match_origin.jpg", combined_img);
            //最终证明是对的*/

            nbgood++;
        } else {
            // 保存被过滤掉的点对
            filtered_points.emplace_back(lunpx, runpx);// add
        }
    }

////////////////  可视化所有被过滤的点
/*     if (!filtered_points.empty()) {
         // 创建拼接图像
         cv::Mat img_left = vleftpyr[0].clone();
         cv::Mat img_right = vrightpyr[0].clone();

         if (img_left.channels() == 1)
             cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
         if (img_right.channels() == 1)
             cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);

         cv::Mat combined_img;
         cv::hconcat(img_left, img_right, combined_img);

         for (const auto &pair : filtered_points) {
             cv::Point2f left_pt = pair.first;
             cv::Point2f right_pt = pair.second;
             right_pt.x += img_left.cols; // 调整右目点位置

             // 使用红色绘制被过滤掉的点对
             cv::Scalar color(0, 0, 255); // 红色
             cv::circle(combined_img, left_pt, 3, color, -1);  // 左目点
             cv::circle(combined_img, right_pt, 3, color, -1); // 右目点
             cv::line(combined_img, left_pt, right_pt, color, 1); // 连线
         }

         // 显示被过滤的点
         cv::imshow("Filtered Matches", combined_img);
         cv::imwrite("/home/hl/project/ov2_diverg_ws/test2/Filtered.png", combined_img);
         cv::waitKey(1); // 延时以刷新显示
     }//*/
////////////////  可视化最终匹配点
    cv::Mat img_left = vleftpyr[0].clone();
    cv::Mat img_right = vrightpyr[0].clone();
    if (img_left.channels() == 1)
        cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
    if (img_right.channels() == 1)
        cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);
    cv::Mat combined_img;
    cv::hconcat(img_left, img_right, combined_img);
    // 从 Frame 中提取已更新的立体点
    for (const auto &pair : frame_r.mapkps_) {//左目点
        const Keypoint &kp = pair.second;
        if (kp.is_stereo_) {  // 仅处理通过过滤的点
//            cv::Point2f lunpx = kp.unpx_;//改为双目区点坐标
            Eigen::Vector3d kp_point3D((kp.unpx_.x - pslamstate_->cxl_)/pslamstate_->fxl_, (kp.unpx_.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化3d坐标
            Eigen::Vector3d kp_point3D1 = pslamstate_->R_sr.inverse() * kp_point3D;
            cv::Point2f runpx = frame_s.projCamToImage_s(kp_point3D1);
            Eigen::Vector3d kpl_point3D((kp.runpx_.x - pslamstate_->cxl_)/pslamstate_->fxl_, (kp.runpx_.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化3d坐标
            Eigen::Vector3d kpl_point3D1 = pslamstate_->R_sl.inverse() * kpl_point3D;
            cv::Point2f lunpx = frame_s.projCamToImage_s(kpl_point3D1);
            runpx.x += img_left.cols; // 调整右目点的 x 坐标以适应拼接图像
            cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
            cv::circle(combined_img, lunpx, 3, color, -1);  // 左目关键点
            cv::circle(combined_img, runpx, 3, color, -1); // 右目关键点
            cv::line(combined_img, lunpx, runpx, color, 1); // 连线
        }
    }
    cv::imshow("Final Stereo Matches stereo right", combined_img);
//    cv::imwrite("/home/hl/project/ov2_diverg_ws/test2/Final_sr.png", combined_img);
    cv::waitKey(1);
////////////////
    cv::Mat img_left2 = vleftpyr_origin[0].clone();
    cv::Mat img_right2 = vrightpyr_origin[0].clone();
    if (img_left2.channels() == 1)
        cv::cvtColor(img_left2, img_left2, cv::COLOR_GRAY2BGR);
    if (img_right2.channels() == 1)
        cv::cvtColor(img_right2, img_right2, cv::COLOR_GRAY2BGR);
    cv::Mat combined_img2;
    cv::hconcat(img_left2, img_right2, combined_img2);
    // 从 Frame 中提取已更新的立体点
    for (const auto &pair : frame_r.mapkps_) {
        const Keypoint &kp = pair.second;
        if (kp.is_stereo_) {  // 仅处理通过过滤的点
            cv::Point2f runpx = kp.unpx_;
            cv::Point2f lunpx = kp.runpx_;
            runpx.x += img_left2.cols; // 调整右目点的 x 坐标以适应拼接图像
            cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
            cv::circle(combined_img2, lunpx, 3, color, -1);  // 左目关键点
            cv::circle(combined_img2, runpx, 3, color, -1); // 右目关键点
            cv::line(combined_img2, lunpx, runpx, color, 1); // 连线
        }
    }
//    cv::imshow("Final Stereo Matches original right", combined_img2);
//    cv::imwrite("/home/hl/project/ov2_diverg_ws/test2/Final_or.png", combined_img2);
    cv::waitKey(1);
////////////////

    if( pslamstate_->debug_ )
        std::cout << "\t>>> Nb of stereo tracks: " << nbgood << " out of " << nbkps << std::endl;

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_stereoMatching");
}


Eigen::Vector3d MapManager::computeTriangulation(const Sophus::SE3d &T, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr)
{
    // OpenGV Triangulate
    return MultiViewGeometry::triangulate(T, bvl, bvr);
}

// This function copies cur. Frame to add it to the KF map
void MapManager::addKeyframe()
{
    // Create a copy of Cur. Frame shared_ptr for creating an 
    // independant KF to add to the map
    std::shared_ptr<Frame> pkf = std::allocate_shared<Frame>(Eigen::aligned_allocator<Frame>(), *pcurframe_);

    std::lock_guard<std::mutex> lock(kf_mutex_);

    // Add KF to the unordered map and update id/nb
    map_pkfs_.emplace(nkfid_, pkf);
    nbkfs_++;
    nkfid_++;
}

// This function adds a new MP to the map
void MapManager::addMapPoint(const cv::Scalar &color)
{
    // Create a new MP with a unique lmid and a KF id obs
    std::shared_ptr<MapPoint> plm = std::allocate_shared<MapPoint>(Eigen::aligned_allocator<MapPoint>(), nlmid_, nkfid_, color);

    // Add new MP to the map and update id/nb
    map_plms_.emplace(nlmid_, plm);
    nlmid_++;
    nblms_++;

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    if( plm->isobs_ ) {
        colored_pt = pcl::PointXYZRGB(255, 0, 0);
    } else {
        colored_pt = pcl::PointXYZRGB(plm->color_[0] 
                                    , plm->color_[0]
                                    , plm->color_[0]
                                    );
    }
    colored_pt.x = 0.;
    colored_pt.y = 0.;
    colored_pt.z = 0.;
    pcloud_->points.push_back(colored_pt);
}


// This function adds a new MP to the map with desc
void MapManager::addMapPoint(const cv::Mat &desc, const cv::Scalar &color)
{
    // Create a new MP with a unique lmid and a KF id obs
    std::shared_ptr<MapPoint> plm = std::allocate_shared<MapPoint>(Eigen::aligned_allocator<MapPoint>(), nlmid_, nkfid_, desc, color);

    // Add new MP to the map and update id/nb
    map_plms_.emplace(nlmid_, plm);
    nlmid_++;
    nblms_++;

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    if( plm->isobs_ ) {
        colored_pt = pcl::PointXYZRGB(255, 0, 0);
    } else {
        colored_pt = pcl::PointXYZRGB(plm->color_[0] 
                                    , plm->color_[0]
                                    , plm->color_[0]
                                    );
    }
    colored_pt.x = 0.;
    colored_pt.y = 0.;
    colored_pt.z = 0.;
    pcloud_->points.push_back(colored_pt);
}

void MapManager::addKeyframe(std::shared_ptr<Frame> &pframe)
{
    // Create a copy of Cur. Frame shared_ptr for creating an
    // independant KF to add to the map
    std::shared_ptr<Frame> pkf = std::allocate_shared<Frame>(Eigen::aligned_allocator<Frame>(), *pframe);

    std::lock_guard<std::mutex> lock(kf_mutex_);

    // Add KF to the unordered map and update id/nb
//    std::cout<<"test: "<<pframe->nbwcells_s_<<std::endl;//没问题
//    std::cout<<"test: "<<pkf->nbwcells_s_<<std::endl;//有问题！！！
    map_pkfs_.emplace(nkfid_, pkf);//map_pkfs_属于pmap_，pmap_已经分左右了，map_pkfs_就不用分左右了
//    std::cout<<"addKeyframe nkfid_:"<<nkfid_<<std::endl;//0
//    std::cout<<"test: "<<map_pkfs_.find(0)->second->nbwcells_s_<<std::endl;//有问题！！！
    nbkfs_++;
    nkfid_++;
}

// Returns a shared_ptr of the req. KF
std::shared_ptr<Frame> MapManager::getKeyframe(const int kfid) const
{
    std::lock_guard<std::mutex> lock(kf_mutex_);

    auto it = map_pkfs_.find(kfid);
    if( it == map_pkfs_.end() ) {
        return nullptr;
    }
    return it->second;
}

// Returns a shared_ptr of the req. MP
std::shared_ptr<MapPoint> MapManager::getMapPoint(const int lmid) const
{
    std::lock_guard<std::mutex> lock(lm_mutex_);

    auto it = map_plms_.find(lmid);
    if( it == map_plms_.end() ) {
        return nullptr;
    }
    return it->second;
}

// Update a MP world pos.
void MapManager::updateMapPoint(const int lmid, const Eigen::Vector3d &wpt, const double kfanch_invdepth)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    auto plmit = map_plms_.find(lmid);

    if( plmit == map_plms_.end() ) {
        return;
    }

    if( plmit->second == nullptr ) {
        return;
    }

    // If MP 2D -> 3D => Notif. KFs 
    if( !plmit->second->is3d_ ) {
        for( const auto &kfid : plmit->second->getKfObsSet() ) {
            auto pkfit = map_pkfs_.find(kfid);
            if( pkfit != map_pkfs_.end() ) {
                pkfit->second->turnKeypoint3d(lmid);
            } else {
                plmit->second->removeKfObs(kfid);
            }
        }
        if( plmit->second->isobs_ ) {
            pcurframe_->turnKeypoint3d(lmid);
        }
    }

    // Update MP world pos.
    if( kfanch_invdepth >= 0. ) {
        plmit->second->setPoint(wpt, kfanch_invdepth);
    } else {
        plmit->second->setPoint(wpt);
    }

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    if(plmit->second->isobs_ ) {
        colored_pt = pcl::PointXYZRGB(255, 0, 0);
    } else {
        colored_pt = pcl::PointXYZRGB(plmit->second->color_[0] 
                                    , plmit->second->color_[0]
                                    , plmit->second->color_[0]
                                    );
    }
    colored_pt.x = wpt.x();
    colored_pt.y = wpt.y();
    colored_pt.z = wpt.z();
    pcloud_->points.at(lmid) = colored_pt;
}

void MapManager::updateMapPoint(const int lmid, const Eigen::Vector3d &wpt, bool isleft, const double kfanch_invdepth)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    auto plmit = map_plms_.find(lmid);

    if( plmit == map_plms_.end() ) {
        return;
    }

    if( plmit->second == nullptr ) {
        return;
    }

    // If MP 2D -> 3D => Notif. KFs
    if( !plmit->second->is3d_ ) {
        for( const auto &kfid : plmit->second->getKfObsSet() ) {
            auto pkfit = map_pkfs_.find(kfid);
            if( pkfit != map_pkfs_.end() ) {
                pkfit->second->turnKeypoint3d(lmid);
            } else {
                plmit->second->removeKfObs(kfid);
            }
        }
        if( plmit->second->isobs_ ) {
//            pcurframe_->turnKeypoint3d(lmid);//
            if(isleft){
                pcurframe_l_->turnKeypoint3d(lmid);// mono_stereo
            } else {
                pcurframe_r_->turnKeypoint3d(lmid);// mono_stereo
            }
        }
    }

    // Update MP world pos.
    if( kfanch_invdepth >= 0. ) {
        plmit->second->setPoint(wpt, kfanch_invdepth);
    } else {
        plmit->second->setPoint(wpt);
    }

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    if(plmit->second->isobs_ ) {
        colored_pt = pcl::PointXYZRGB(255, 0, 0);
    } else {
        colored_pt = pcl::PointXYZRGB(plmit->second->color_[0]
                , plmit->second->color_[0]
                , plmit->second->color_[0]
        );
    }
    colored_pt.x = wpt.x();
    colored_pt.y = wpt.y();
    colored_pt.z = wpt.z();
    pcloud_->points.at(lmid) = colored_pt;
}

void MapManager::updateMapPoint_s(const int lmid, const Eigen::Vector3d &wpt, bool isleft, const double kfanch_invdepth)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    auto plmit = map_plms_.find(lmid);

    if( plmit == map_plms_.end() ) {
        return;
    }

    if( plmit->second == nullptr ) {
        return;
    }

    // If MP 2D -> 3D => Notif. KFs
    if( !plmit->second->is3d_ ) {
        for( const auto &kfid : plmit->second->getKfObsSet() ) {
            auto pkfit = map_pkfs_.find(kfid);
            if( pkfit != map_pkfs_.end() ) {
                pkfit->second->turnKeypoint3d(lmid);
            } else {
                plmit->second->removeKfObs(kfid);
            }
        }
        if( plmit->second->isobs_ ) {
//            pcurframe_->turnKeypoint3d(lmid);//
            if(isleft){
                pcurframe_ls_->turnKeypoint3d(lmid);// mono_stereo
            } else {
                pcurframe_rs_->turnKeypoint3d(lmid);// mono_stereo
            }
        }
    }

    // Update MP world pos.
    if( kfanch_invdepth >= 0. ) {
        plmit->second->setPoint(wpt, kfanch_invdepth);
    } else {
        plmit->second->setPoint(wpt);
    }

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    if(plmit->second->isobs_ ) {
        colored_pt = pcl::PointXYZRGB(255, 0, 0);
    } else {
        colored_pt = pcl::PointXYZRGB(plmit->second->color_[0]
                , plmit->second->color_[0]
                , plmit->second->color_[0]
        );
    }
    colored_pt.x = wpt.x();
    colored_pt.y = wpt.y();
    colored_pt.z = wpt.z();
    pcloud_->points.at(lmid) = colored_pt;
}

// Add a new KF obs to provided MP (lmid)
void MapManager::addMapPointKfObs(const int lmid, const int kfid)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    auto pkfit = map_pkfs_.find(kfid);
    auto plmit = map_plms_.find(lmid);

    if( pkfit == map_pkfs_.end() ) {
        return;
    }

    if( plmit == map_plms_.end() ) {
        return;
    }

    plmit->second->addKfObs(kfid);

    for( const auto &cokfid : plmit->second->getKfObsSet() ) {
        if( cokfid != kfid ) {
            auto pcokfit =  map_pkfs_.find(cokfid);
            if( pcokfit != map_pkfs_.end() ) {
                pcokfit->second->addCovisibleKf(kfid);
                pkfit->second->addCovisibleKf(cokfid);
            } else {
                plmit->second->removeKfObs(cokfid);
            }
        }
    }
}

// Merge two MapPoints
void MapManager::mergeMapPoints(const int prevlmid, const int newlmid)
{
    // 1. Get Kf obs + descs from prev MP
    // 2. Remove prev MP
    // 3. Update new MP and related KF / cur Frame

    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    // Get prev MP to merge into new MP

    auto pprevlmit = map_plms_.find(prevlmid);
    auto pnewlmit = map_plms_.find(newlmid);

    if( pprevlmit == map_plms_.end() ) {
        if( pslamstate_->debug_ )
            std::cout << "\nMergeMapPoints skipping as prevlm is null\n";
        return;
    } else if( pnewlmit == map_plms_.end() ) {
        if( pslamstate_->debug_ )
            std::cout << "\nMergeMapPoints skipping as newlm is null\n";
        return;
    } else if ( !pnewlmit->second->is3d_ ) {
        if( pslamstate_->debug_ )
            std::cout << "\nMergeMapPoints skipping as newlm is not 3d\n";
        return;
    }

    // 1. Get Kf obs + descs from prev MP
    std::set<int> setnewkfids = pnewlmit->second->getKfObsSet();
    std::set<int> setprevkfids = pprevlmit->second->getKfObsSet();
    std::unordered_map<int, cv::Mat> map_prev_kf_desc_ = pprevlmit->second->map_kf_desc_;

    // 3. Update new MP and related KF / cur Frame
    for( const auto &pkfid : setprevkfids ) 
    {
        // Get prev KF and update keypoint
        auto pkfit =  map_pkfs_.find(pkfid);
        if( pkfit != map_pkfs_.end() ) {
            if( pkfit->second->updateKeypointId(prevlmid, newlmid, pnewlmit->second->is3d_) )
            {
                pnewlmit->second->addKfObs(pkfid);
                for( const auto &nkfid : setnewkfids ) {
                    auto pcokfit = map_pkfs_.find(nkfid);
                    if( pcokfit != map_pkfs_.end() ) {
                        pkfit->second->addCovisibleKf(nkfid);
                        pcokfit->second->addCovisibleKf(pkfid);
                    }
                }
            }
        }
    }

    for( const auto &kfid_desc : map_prev_kf_desc_ ) {
        pnewlmit->second->addDesc(kfid_desc.first, kfid_desc.second);
    }

    // Turn new MP observed by cur Frame if prev MP
    // was + update cur Frame's kp ref to new MP
    if( pcurframe_->isObservingKp(prevlmid) ) 
    {
        if( pcurframe_->updateKeypointId(prevlmid, newlmid, pnewlmit->second->is3d_) )
        {
            setMapPointObs(newlmid);
        }
    }

    if( pprevlmit->second->is3d_ ) {
        nblms_--; 
    }

    // Erase MP and update nb MPs
    map_plms_.erase( pprevlmit );
    
    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    colored_pt = pcl::PointXYZRGB(0, 0, 0);
    colored_pt.x = 0.;
    colored_pt.y = 0.;
    colored_pt.z = 0.;
    pcloud_->points[prevlmid] = colored_pt;
}

void MapManager::mergeMapPoints(const int prevlmid, const int newlmid, bool isleft)
{
    // 1. Get Kf obs + descs from prev MP
    // 2. Remove prev MP
    // 3. Update new MP and related KF / cur Frame

    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    // Get prev MP to merge into new MP

    auto pprevlmit = map_plms_.find(prevlmid);
    auto pnewlmit = map_plms_.find(newlmid);

    if( pprevlmit == map_plms_.end() ) {
        if( pslamstate_->debug_ )
            std::cout << "\nMergeMapPoints skipping as prevlm is null\n";
        return;
    } else if( pnewlmit == map_plms_.end() ) {
        if( pslamstate_->debug_ )
            std::cout << "\nMergeMapPoints skipping as newlm is null\n";
        return;
    } else if ( !pnewlmit->second->is3d_ ) {
        if( pslamstate_->debug_ )
            std::cout << "\nMergeMapPoints skipping as newlm is not 3d\n";
        return;
    }

    // 1. Get Kf obs + descs from prev MP
    std::set<int> setnewkfids = pnewlmit->second->getKfObsSet();
    std::set<int> setprevkfids = pprevlmit->second->getKfObsSet();
    std::unordered_map<int, cv::Mat> map_prev_kf_desc_ = pprevlmit->second->map_kf_desc_;

    // 3. Update new MP and related KF / cur Frame
    for( const auto &pkfid : setprevkfids )
    {
        // Get prev KF and update keypoint
        auto pkfit =  map_pkfs_.find(pkfid);
        if( pkfit != map_pkfs_.end() ) {
            if( pkfit->second->updateKeypointId(prevlmid, newlmid, pnewlmit->second->is3d_) )
            {
                pnewlmit->second->addKfObs(pkfid);
                for( const auto &nkfid : setnewkfids ) {
                    auto pcokfit = map_pkfs_.find(nkfid);
                    if( pcokfit != map_pkfs_.end() ) {
                        pkfit->second->addCovisibleKf(nkfid);
                        pcokfit->second->addCovisibleKf(pkfid);
                    }
                }
            }
        }
    }

    for( const auto &kfid_desc : map_prev_kf_desc_ ) {
        pnewlmit->second->addDesc(kfid_desc.first, kfid_desc.second);
    }

    // Turn new MP observed by cur Frame if prev MP
    // was + update cur Frame's kp ref to new MP
    if(isleft){
        if( pcurframe_l_->isObservingKp(prevlmid) )//
        {
            if( pcurframe_l_->updateKeypointId(prevlmid, newlmid, pnewlmit->second->is3d_) )//
            {
                setMapPointObs(newlmid);
            }
        }
    } else {
        if( pcurframe_r_->isObservingKp(prevlmid) )//
        {
            if( pcurframe_r_->updateKeypointId(prevlmid, newlmid, pnewlmit->second->is3d_) )//
            {
                setMapPointObs(newlmid);
            }
        }
    }

    if( pprevlmit->second->is3d_ ) {
        nblms_--;
    }

    // Erase MP and update nb MPs
    map_plms_.erase( pprevlmit );

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    colored_pt = pcl::PointXYZRGB(0, 0, 0);
    colored_pt.x = 0.;
    colored_pt.y = 0.;
    colored_pt.z = 0.;
    pcloud_->points[prevlmid] = colored_pt;
}

// Remove a KF from the map
void MapManager::removeKeyframe(const int kfid)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    // Get KF to remove
    auto pkfit = map_pkfs_.find(kfid);
    // Skip if KF does not exist
    if( pkfit == map_pkfs_.end() ) {
        return;
    }

    // Remove the KF obs from all observed MP
    for( const auto &kp : pkfit->second->getKeypoints() ) {
        // Get MP and remove KF obs
        auto plmit = map_plms_.find(kp.lmid_);
        if( plmit == map_plms_.end() ) {
            continue;
        }
        plmit->second->removeKfObs(kfid);
    }
    for( const auto &kfid_cov : pkfit->second->getCovisibleKfMap() ) {
        auto pcokfit = map_pkfs_.find(kfid_cov.first);
        if( pcokfit != map_pkfs_.end() ) {
            pcokfit->second->removeCovisibleKf(kfid);
        }
    }

    // Remove KF and update nb KFs
    map_pkfs_.erase( pkfit );
    nbkfs_--;

    if( pslamstate_->debug_ )
        std::cout << "\n \t >>> removeKeyframe() --> Removed KF #" << kfid;
}

// Remove a MP from the map
void MapManager::removeMapPoint(const int lmid)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    // Get related MP
    auto plmit = map_plms_.find(lmid);
    // Skip if MP does not exist
    if( plmit != map_plms_.end() ) {
        // Remove all observations from KFs
        for( const auto &kfid : plmit->second->getKfObsSet() ) 
        {
            auto pkfit = map_pkfs_.find(kfid);
            if( pkfit == map_pkfs_.end() ) {
                continue;
            }
            pkfit->second->removeKeypointById(lmid);

            for( const auto &cokfid : plmit->second->getKfObsSet() ) {
                if( cokfid != kfid ) {
                    pkfit->second->decreaseCovisibleKf(cokfid);
                }
            }
        }

        // If obs in cur Frame, remove cur obs
        if( plmit->second->isobs_ ) {
            pcurframe_->removeKeypointById(lmid);
        }

        if( plmit->second->is3d_ ) {
            nblms_--; 
        }

        // Erase MP and update nb MPs
        map_plms_.erase( plmit );
    }

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    colored_pt = pcl::PointXYZRGB(0, 0, 0);
    colored_pt.x = 0.;
    colored_pt.y = 0.;
    colored_pt.z = 0.;
    pcloud_->points.at(lmid) = colored_pt;
}

// Remove a KF obs from a MP
void MapManager::removeMapPointObs(const int lmid, const int kfid)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    // Remove MP obs from KF
    auto pkfit = map_pkfs_.find(kfid);
    if( pkfit != map_pkfs_.end() ) {
        pkfit->second->removeKeypointById(lmid);
    }

    // Remove KF obs from MP
    auto plmit = map_plms_.find(lmid);

    // Skip if MP does not exist
    if( plmit == map_plms_.end() ) {
        return;
    }
    plmit->second->removeKfObs(kfid);

    if( pkfit != map_pkfs_.end() ) {
        for( const auto &cokfid : plmit->second->getKfObsSet() ) {
            auto pcokfit = map_pkfs_.find(cokfid);
            if( pcokfit != map_pkfs_.end() ) {
                pkfit->second->decreaseCovisibleKf(cokfid);
                pcokfit->second->decreaseCovisibleKf(kfid);
            }
        }
    }
}

void MapManager::removeMapPointObs_s(const int lmid, const int kfid)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    // Remove MP obs from KF
    auto pkfit = map_pkfs_.find(kfid);
    if( pkfit != map_pkfs_.end() ) {
        pkfit->second->removeKeypointById_s(lmid);//mono_stereo
    }

    // Remove KF obs from MP
    auto plmit = map_plms_.find(lmid);

    // Skip if MP does not exist
    if( plmit == map_plms_.end() ) {
        return;
    }
    plmit->second->removeKfObs(kfid);

    if( pkfit != map_pkfs_.end() ) {
        for( const auto &cokfid : plmit->second->getKfObsSet() ) {
            auto pcokfit = map_pkfs_.find(cokfid);
            if( pcokfit != map_pkfs_.end() ) {
                pkfit->second->decreaseCovisibleKf(cokfid);//map_covkfs_是frame下面的，所以应该不用分左右
                pcokfit->second->decreaseCovisibleKf(kfid);
            }
        }
    }
}

void MapManager::removeMapPointObs(MapPoint &lm, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    frame.removeKeypointById(lm.lmid_);
    lm.removeKfObs(frame.kfid_);

    for( const auto &cokfid : lm.getKfObsSet() ) {
        if( cokfid != frame.kfid_ ) {
            auto pcokfit = map_pkfs_.find(cokfid);
            if( pcokfit != map_pkfs_.end() ) {
                frame.decreaseCovisibleKf(cokfid);
                pcokfit->second->decreaseCovisibleKf(frame.kfid_);
            }
        }
    }
}

// Remove a MP obs from cur Frame
void MapManager::removeObsFromCurFrameById(const int lmid)
{
    // Remove cur obs
    pcurframe_->removeKeypointById(lmid);
    
    // Set MP as not obs
    auto plmit = map_plms_.find(lmid);

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;

    // Skip if MP does not exist
    if( plmit == map_plms_.end() ) {
        // Set the MP at origin
        pcloud_->points.at(lmid) = colored_pt;
        return;
    }

    plmit->second->isobs_ = false;

    // Update MP color
    colored_pt = pcl::PointXYZRGB(plmit->second->color_[0] 
                                , plmit->second->color_[0]
                                , plmit->second->color_[0]
                                );
                                
    colored_pt.x = pcloud_->points.at(lmid).x;
    colored_pt.y = pcloud_->points.at(lmid).y;
    colored_pt.z = pcloud_->points.at(lmid).z;
    pcloud_->points.at(lmid) = colored_pt;
}

void MapManager::removeObsFromCurFrameById(const int lmid, std::shared_ptr<Frame> &pframe)
{
    // Remove cur obs
//    pcurframe_->removeKeypointById(lmid);
    pframe->removeKeypointById(lmid);

    // Set MP as not obs
    auto plmit = map_plms_.find(lmid);

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;

    // Skip if MP does not exist
    if( plmit == map_plms_.end() ) {
        // Set the MP at origin
        pcloud_->points.at(lmid) = colored_pt;
        return;
    }

    plmit->second->isobs_ = false;

    // Update MP color
    colored_pt = pcl::PointXYZRGB(plmit->second->color_[0]
            , plmit->second->color_[0]
            , plmit->second->color_[0]
    );

    colored_pt.x = pcloud_->points.at(lmid).x;
    colored_pt.y = pcloud_->points.at(lmid).y;
    colored_pt.z = pcloud_->points.at(lmid).z;
    pcloud_->points.at(lmid) = colored_pt;
}

bool MapManager::setMapPointObs(const int lmid) 
{
    if( lmid >= (int)pcloud_->points.size() ) {
        return false;
    }

    auto plmit = map_plms_.find(lmid);

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;

    // Skip if MP does not exist
    if( plmit == map_plms_.end() ) {
        // Set the MP at origin
        pcloud_->points.at(lmid) = colored_pt;
        return false;
    }

    plmit->second->isobs_ = true;

    // Update MP color
    colored_pt = pcl::PointXYZRGB(200, 0, 0);
    colored_pt.x = pcloud_->points.at(lmid).x;
    colored_pt.y = pcloud_->points.at(lmid).y;
    colored_pt.z = pcloud_->points.at(lmid).z;
    pcloud_->points.at(lmid) = colored_pt;

    return true;
}

// Reset MapManager
void MapManager::reset()
{
    nlmid_ = 0;
    nkfid_ = 0;
    nblms_ = 0;
    nbkfs_ = 0;

    map_pkfs_.clear();
    map_plms_.clear();

    pcloud_->points.clear();
}
