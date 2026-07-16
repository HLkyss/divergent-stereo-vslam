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

#include <thread>
#include <opencv2/highgui.hpp>

#include "ov2slam.hpp"
#include "virtual_image_kernel.cuh"

SlamManager::SlamManager(std::shared_ptr<SlamParams> pstate, std::shared_ptr<RosVisualizer> pviz)
    : pslamstate_(pstate)
    , prosviz_(pviz)
{
    std::cout << "\n SLAM Manager is being created...\n";

    #ifdef OPENCV_CONTRIB
        std::cout << "\n OPENCV CONTRIB FOUND!  BRIEF DESCRIPTOR WILL BE USED!\n";
    #else
        std::cout << "\n OPENCV CONTRIB NOT FOUND!  ORB DESCRIPTOR WILL BE USED!\n";
    #endif

    #ifdef USE_OPENGV
        std::cout << "\n OPENGV FOUND!  OPENGV MVG FUNCTIONS WILL BE USED!\n";
    #else
        std::cout << "\n OPENGV NOT FOUND!  OPENCV MVG FUNCTIONS WILL BE USED!\n";
    #endif

    // We first setup the calibration to init everything related
    // to the configuration of the current run
    std::cout << "\n SetupCalibration()\n";
    setupCalibration();//已修改

    if( pslamstate_->stereo_ && pslamstate_->bdo_stereo_rect_ ) {
        std::cout << "\n SetupStereoCalibration()\n";
        setupStereoCalibration();//只对双目区图像进行对齐？可以先不修改，不启用对齐功能
    }
    else if( pslamstate_->mono_ && pslamstate_->bdo_stereo_rect_ ) {
        pslamstate_->bdo_stereo_rect_ = false;
    }

    // If no stereo rectification required (i.e. mono config or 
    // stereo w/o rectification) and image undistortion required
    if( !pslamstate_->bdo_stereo_rect_ && pslamstate_->bdo_undist_ ) {
        std::cout << "\n Setup Image Undistortion\n";
        pcalib_model_left_->setUndistMap(pslamstate_->alpha_);//先不修改
        if( pslamstate_->stereo_ )
            pcalib_model_right_->setUndistMap(pslamstate_->alpha_);
    }

    if( pslamstate_->mono_ ) {
        pcurframe_.reset( new Frame(pcalib_model_left_, pslamstate_->nmaxdist_) );
    } else if( pslamstate_->stereo_ ) {
        pcurframe_.reset( new Frame(pcalib_model_left_, pcalib_model_right_, pslamstate_->nmaxdist_) );
    } else if ( pslamstate_->mono_stereo_){
        pcurframe_.reset( new Frame(pcalib_model_left_, pcalib_model_right_, pcalib_model_left_mono_, pcalib_model_right_mono_, pcalib_model_left_stereo_, pcalib_model_right_stereo_, pslamstate_->nmaxdist_, pslamstate_->theta) );
        pcurframe_l_.reset( new Frame(pcalib_model_left_, pcalib_model_right_, pcalib_model_left_mono_, pcalib_model_right_mono_, pcalib_model_left_stereo_, pcalib_model_right_stereo_, pslamstate_->nmaxdist_, pslamstate_->theta) );
        pcurframe_r_.reset( new Frame(pcalib_model_left_, pcalib_model_right_, pcalib_model_left_mono_, pcalib_model_right_mono_, pcalib_model_left_stereo_, pcalib_model_right_stereo_, pslamstate_->nmaxdist_, pslamstate_->theta) );
        pcurframe_lm_.reset( new Frame(pcalib_model_left_, pcalib_model_right_, pcalib_model_left_mono_, pcalib_model_right_mono_, pcalib_model_left_stereo_, pcalib_model_right_stereo_, pslamstate_->nmaxdist_, pslamstate_->theta) );
        pcurframe_ls_.reset( new Frame(pcalib_model_left_, pcalib_model_right_, pcalib_model_left_mono_, pcalib_model_right_mono_, pcalib_model_left_stereo_, pcalib_model_right_stereo_, pslamstate_->nmaxdist_, pslamstate_->theta) );
        pcurframe_rm_.reset( new Frame(pcalib_model_left_, pcalib_model_right_, pcalib_model_left_mono_, pcalib_model_right_mono_, pcalib_model_left_stereo_, pcalib_model_right_stereo_, pslamstate_->nmaxdist_, pslamstate_->theta) );
        pcurframe_rs_.reset( new Frame(pcalib_model_left_, pcalib_model_right_, pcalib_model_left_mono_, pcalib_model_right_mono_, pcalib_model_left_stereo_, pcalib_model_right_stereo_, pslamstate_->nmaxdist_, pslamstate_->theta) );
//        std::cout<<"pmap_ nbwcells_s_ -----"<<pmap_->getKeyframe(0)->nbwcells_s_<<std::endl;//从这里开始就有问题了
//        std::cout<<"pmap_ nbwcells_s_ -----"<<pcurframe_->nbwcells_s_<<std::endl;//在pcurframe_里通过frame的构造函数，正确赋值。没问题

    } else {
        std::cerr << "\n\n=====================================================\n\n";
        std::cerr << "\t\t YOU MUST CHOOSE BETWEEN MONO / STEREO (and soon RGBD) / MONO_STEREO\n";
        std::cerr << "\n\n=====================================================\n\n";
    }

    // Create all objects to be used within OV²SLAM
    // =============================================
    int tilesize = 50;
    cv::Size clahe_tiles(pcalib_model_left_->img_w_ / tilesize
                        , pcalib_model_left_->img_h_ / tilesize);
    cv::Size clahe_tiles_m(pcalib_model_left_mono_->img_w_ / tilesize
            , pcalib_model_left_mono_->img_h_ / tilesize);
    cv::Size clahe_tiles_s(pcalib_model_left_stereo_->img_w_ / tilesize
            , pcalib_model_left_stereo_->img_h_ / tilesize);
                        
    cv::Ptr<cv::CLAHE> pclahe = cv::createCLAHE(pslamstate_->fclahe_val_, clahe_tiles);
    cv::Ptr<cv::CLAHE> pclahe_m = cv::createCLAHE(pslamstate_->fclahe_val_, clahe_tiles_m);
    cv::Ptr<cv::CLAHE> pclahe_s = cv::createCLAHE(pslamstate_->fclahe_val_, clahe_tiles_s);

    if(pslamstate_->mono_stereo_){
        pfeatextract_.reset( new FeatureExtractor(
                                     pslamstate_->nbmaxkps_, pslamstate_->nbmaxkps_m_, pslamstate_->nbmaxkps_s_, pslamstate_->nmaxdist_,
                                     pslamstate_->dmaxquality_, pslamstate_->nfast_th_
                             )
        );

        ptracker_.reset( new FeatureTracker(pslamstate_->nmax_iter_,
                                            pslamstate_->fmax_px_precision_, pclahe, pclahe_m, pclahe_s
                         )
        );

        // Map Manager will handle Keyframes / MapPoints
        pmap_.reset( new MapManager(pslamstate_, pcurframe_, pcurframe_l_, pcurframe_r_, pcurframe_lm_, pcurframe_ls_, pcurframe_rm_, pcurframe_rs_, pfeatextract_, ptracker_) );
        pmap_l_.reset( new MapManager(pslamstate_, pcurframe_, pcurframe_l_, pcurframe_r_, pcurframe_lm_, pcurframe_ls_, pcurframe_rm_, pcurframe_rs_, pfeatextract_, ptracker_) );
        pmap_r_.reset( new MapManager(pslamstate_, pcurframe_, pcurframe_l_, pcurframe_r_, pcurframe_lm_, pcurframe_ls_, pcurframe_rm_, pcurframe_rs_, pfeatextract_, ptracker_) );
        pmap_lm_.reset( new MapManager(pslamstate_, pcurframe_, pcurframe_l_, pcurframe_r_, pcurframe_lm_, pcurframe_ls_, pcurframe_rm_, pcurframe_rs_, pfeatextract_, ptracker_) );
        pmap_ls_.reset( new MapManager(pslamstate_, pcurframe_, pcurframe_l_, pcurframe_r_, pcurframe_lm_, pcurframe_ls_, pcurframe_rm_, pcurframe_rs_, pfeatextract_, ptracker_) );
        pmap_rm_.reset( new MapManager(pslamstate_, pcurframe_, pcurframe_l_, pcurframe_r_, pcurframe_lm_, pcurframe_ls_, pcurframe_rm_, pcurframe_rs_, pfeatextract_, ptracker_) );
        pmap_rs_.reset( new MapManager(pslamstate_, pcurframe_, pcurframe_l_, pcurframe_r_, pcurframe_lm_, pcurframe_ls_, pcurframe_rm_, pcurframe_rs_, pfeatextract_, ptracker_) );
        pmap_l_to_r_.reset( new MapManager(pslamstate_, pcurframe_, pcurframe_l_, pcurframe_r_, pcurframe_lm_, pcurframe_ls_, pcurframe_rm_, pcurframe_rs_, pfeatextract_, ptracker_) );
        pmap_r_to_l_.reset( new MapManager(pslamstate_, pcurframe_, pcurframe_l_, pcurframe_r_, pcurframe_lm_, pcurframe_ls_, pcurframe_rm_, pcurframe_rs_, pfeatextract_, ptracker_) );

        // Visual Front-End processes every incoming frames
        pvisualfrontend_.reset( new VisualFrontEnd(pslamstate_, pcurframe_, pcurframe_l_, pcurframe_r_, pcurframe_lm_, pcurframe_ls_, pcurframe_rm_, pcurframe_rs_,
                                                   pmap_, pmap_l_, pmap_r_, pmap_lm_, pmap_ls_, pmap_rm_, pmap_rs_, pmap_l_to_r_, pmap_r_to_l_, ptracker_
                                )
        );

        // Mapper thread handles Keyframes' processing
        // (i.e. triangulation, local map tracking, BA, LC)
        pmapper_.reset( new Mapper(pslamstate_, pmap_, pmap_l_, pmap_r_, pmap_lm_, pmap_ls_, pmap_rm_, pmap_rs_, pmap_l_to_r_, pmap_r_to_l_, pcurframe_, pcurframe_l_, pcurframe_r_, pcurframe_lm_, pcurframe_ls_, pcurframe_rm_, pcurframe_rs_) );
    } else {
        pfeatextract_.reset( new FeatureExtractor(
                                    pslamstate_->nbmaxkps_, pslamstate_->nmaxdist_,
                                    pslamstate_->dmaxquality_, pslamstate_->nfast_th_
                                )
                            );

        ptracker_.reset( new FeatureTracker(pslamstate_->nmax_iter_,
                                pslamstate_->fmax_px_precision_, pclahe
                            )
                        );

        // Map Manager will handle Keyframes / MapPoints
        pmap_.reset( new MapManager(pslamstate_, pcurframe_, pfeatextract_, ptracker_) );

        // Visual Front-End processes every incoming frames
        pvisualfrontend_.reset( new VisualFrontEnd(pslamstate_, pcurframe_,
                                                   pmap_, ptracker_
                                )
        );

        // Mapper thread handles Keyframes' processing
        // (i.e. triangulation, local map tracking, BA, LC)
        pmapper_.reset( new Mapper(pslamstate_, pmap_, pcurframe_) );
    }
}

void SlamManager::run()
{
    std::cout << "\nOV²SLAM is ready to process incoming images!\n";

    std::ofstream kf_log("kf_count.txt");  // Keyframe count log for paper revision

    bis_on_ = true;

    cv::Mat img_left, img_right;
    cv::Mat img_left_m, img_left_s, img_right_m, img_right_s;

    double time = -1.; // Current image timestamp
    double cam_delay = -1.; // Delay between two successive images
    double last_img_time = -1.; // Last received image time

    // Main SLAM loop
    while( !bexit_required_ ) {

        // 0. Get New Images
        // =============================================
//        if( getNewImage(img_left, img_right, time) )
        if( getNewImage(img_left, img_right, img_left_m, img_left_s, img_right_m, img_right_s, time) )//mono_stereo 生成虚拟图像
        {
//            std::cout<<"time = "<<time<<std::endl;//有问题，一直是0
            // Update current frame
            frame_id_++;
            pcurframe_->updateFrame(frame_id_, time);
            if(pslamstate_->mono_stereo_){
                pcurframe_l_->updateFrame(frame_id_, time);
                pcurframe_r_->updateFrame(frame_id_, time);
                pcurframe_lm_->updateFrame(frame_id_, time);
                pcurframe_ls_->updateFrame(frame_id_, time);
                pcurframe_rm_->updateFrame(frame_id_, time);
                pcurframe_rs_->updateFrame(frame_id_, time);
            }

            // Update cam delay for automatic exit
            if( frame_id_ > 0 ) {
                cam_delay = ros::Time::now().toSec() - last_img_time;
                last_img_time += cam_delay;
            } else {
                last_img_time = ros::Time::now().toSec();
            }

            // Display info on current frame state
            if( pslamstate_->debug_ ){
                pcurframe_->displayFrameInfo();
                if(pslamstate_->mono_stereo_){
                    pcurframe_l_->displayFrameInfo();
                    pcurframe_r_->displayFrameInfo();
                    pcurframe_lm_->displayFrameInfo();
                    pcurframe_ls_->displayFrameInfo();
                    pcurframe_rm_->displayFrameInfo();
                    pcurframe_rs_->displayFrameInfo();
                }
            }

            // 1. Send images to the FrontEnd
            // =============================================
            if( pslamstate_->debug_ )
                std::cout << "\n \t >>> [SLAM Node] New image send to Front-End\n";

            bool is_kf_req;
            if(pslamstate_->mono_stereo_){
//                is_kf_req = pvisualfrontend_->visualTracking(img_left, img_right, img_left_m, img_left_s, img_right_m, img_right_s, time);
                is_kf_req = pvisualfrontend_->visualTracking(img_left, img_right, img_left_m, img_left_s, img_right_m, img_right_s, time);
            } else {
                is_kf_req = pvisualfrontend_->visualTracking(img_left, time);
            }

            //论文大修 统计关键帧数量
//我们这个代码里，pmap_l_和pmap_r_同时创建，然后各自会有一个nbkfs_成员变量，左地图和右地图的nbkfs_应该是一样的，统计的时候只要统计一个就行，左右同时创建一个关键帧 就当作 创建一个关键帧
            int kf_count = pmap_l_->nbkfs_;  // 左图 KF 数即可
            kf_log << time << " " << kf_count << std::endl;

            // Save current pose
            Logger::addSE3Pose(time, pcurframe_l_->getTwc(), is_kf_req);// todo 保存为左目位姿

            if( pslamstate_->breset_req_ ) {
                reset();// mono_stereo
                continue;
            }

            // 2. Create new KF if req. / Send new KF to Mapper
            // ================================================
            if( is_kf_req )
            {
                if( pslamstate_->debug_ )
                    std::cout << "\n \t >>> [SLAM Node] New Keyframe send to Back-End\n";

                if( pslamstate_->stereo_ )
                {
                    Keyframe kf(
                        pcurframe_->kfid_,
                        img_left,
                        img_right,
                        pvisualfrontend_->cur_pyr_
                        );

                    pmapper_->addNewKf(kf);
                }
                else if( pslamstate_->mono_ )
                {
                    Keyframe kf(pcurframe_->kfid_, img_left);
                    pmapper_->addNewKf(kf);
                }
                else if( pslamstate_->mono_stereo_)// mono_stereo
                {
                    Keyframe kf(
                            pcurframe_->kfid_,
                            img_left,
                            img_right,
                            img_left_m,
                            img_left_s,
                            img_right_m,
                            img_right_s,
                            pvisualfrontend_->cur_pyr_,
                            pvisualfrontend_->cur_pyrl_,
                            pvisualfrontend_->cur_pyrr_,
                            pvisualfrontend_->cur_pyrlm_,
                            pvisualfrontend_->cur_pyrls_,
                            pvisualfrontend_->cur_pyrrm_,
                            pvisualfrontend_->cur_pyrrs_
                    );

//                    Keyframe kf_l(    // todo       可以创建多个keyframe kf，但是现在选择只用一个，所有东西都塞到里面。
//                            pcurframe_->kfid_,
//                            img_left,
//                            img_right,
//                            img_left_m,
//                            img_left_s,
//                            img_right_m,
//                            img_right_s,
//                            pvisualfrontend_->cur_pyr_,
//                            pvisualfrontend_->cur_pyrl_,
//                            pvisualfrontend_->cur_pyrr_,
//                            pvisualfrontend_->cur_pyrlm_,
//                            pvisualfrontend_->cur_pyrls_,
//                            pvisualfrontend_->cur_pyrrm_,
//                            pvisualfrontend_->cur_pyrrs_
//                    );

                    pmapper_->addNewKf(kf);
                }

                if( !bkf_viz_ison_ ) {
                    std::thread kf_viz_thread(&SlamManager::visualizeAtKFsRate, this, time);// todo mono_stereo 共视关键帧可视化、关键帧轨迹结果可视化（只发布与kf相关的）
                    kf_viz_thread.detach();
                }
            }

            if( pslamstate_->debug_ || pslamstate_->log_timings_ )
                std::cout << Profiler::getInstance().displayTimeLogs() << std::endl;

            // Frame rate visualization (limit the visualization processing)
            if( !bframe_viz_ison_ ) {
                std::thread viz_thread(&SlamManager::visualizeAtFrameRate, this, time);// mono_stereo 图像帧可视化、VO轨迹可视化、点云可视化。和上面的可视化更新频率不一样。
                viz_thread.detach();
            }
        }
        else {

            // 3. Check if we are done with a sequence!
            // ========================================
            bool c1 = cam_delay > 0;
            bool c2 = ( ros::Time::now().toSec() - last_img_time ) > 100. * cam_delay;
            bool c3 = !bnew_img_available_;

            if( c1 && c2 && c3 )
            {
                bexit_required_ = true;

                // Warn threads to stop and then save the results only in this case of
                // automatic stop because end of sequence reached
                // (avoid wasting time when forcing stop by CTRL+C)
                pmapper_->bexit_required_ = true;

                writeResults();// todo mono_stereo

                // Notify exit to ROS
                ros::requestShutdown();
            }
            else {
                std::chrono::milliseconds dura(1);
                std::this_thread::sleep_for(dura);
            }
        }
    }

    // 在退出循环后，确保保存结果
    if (bexit_required_) {
        pmapper_->bexit_required_ = true;
        writeResults();
    }

    std::cout << "\nOV²SLAM is stopping!\n";

    bis_on_ = false;
}

void SlamManager::run2(bool isleft)
{
    std::cout << "\nOV²SLAM is ready to process incoming images!\n";

    bis_on_ = true;

    cv::Mat img_left, img_right;
    cv::Mat img_left_m, img_left_s, img_right_m, img_right_s;

    double time = -1.; // Current image timestamp
    double cam_delay = -1.; // Delay between two successive images
    double last_img_time = -1.; // Last received image time

    // Main SLAM loop
    while( !bexit_required_ ) {

        // 0. Get New Images
        // =============================================
//        if( getNewImage(img_left, img_right, time) )
        if( getNewImage(img_left, img_right, img_left_m, img_left_s, img_right_m, img_right_s, time, isleft) )
        {
            // Update current frame
            frame_id_++;
            pcurframe_->updateFrame(frame_id_, time);
            if(pslamstate_->mono_stereo_){
                if(isleft){
                    pcurframe_l_->updateFrame(frame_id_, time);
                    pcurframe_lm_->updateFrame(frame_id_, time);
                    pcurframe_ls_->updateFrame(frame_id_, time);
                } else {
                    pcurframe_r_->updateFrame(frame_id_, time);
                    pcurframe_rm_->updateFrame(frame_id_, time);
                    pcurframe_rs_->updateFrame(frame_id_, time);
                }
            }

            // Update cam delay for automatic exit
            if( frame_id_ > 0 ) {
                cam_delay = ros::Time::now().toSec() - last_img_time;
                last_img_time += cam_delay;
            } else {
                last_img_time = ros::Time::now().toSec();
            }

            // Display info on current frame state
            if( pslamstate_->debug_ ){
                pcurframe_->displayFrameInfo();
                if(pslamstate_->mono_stereo_){
                    if(isleft){
                        pcurframe_l_->displayFrameInfo();
                        pcurframe_lm_->displayFrameInfo();
                        pcurframe_ls_->displayFrameInfo();
                    } else {
                        pcurframe_r_->displayFrameInfo();
                        pcurframe_rm_->displayFrameInfo();
                        pcurframe_rs_->displayFrameInfo();
                    }
                }
            }

            // 1. Send images to the FrontEnd
            // =============================================
            if( pslamstate_->debug_ )
                std::cout << "\n \t >>> [SLAM Node] New image send to Front-End\n";

            bool is_kf_req;
            if(pslamstate_->mono_stereo_){
                if(isleft){
                    is_kf_req = pvisualfrontend_->visualTracking(img_left, img_right, img_left_m, img_left_s, img_right_m, img_right_s, time, true);
                } else {
                    is_kf_req = pvisualfrontend_->visualTracking(img_left, img_right, img_left_m, img_left_s, img_right_m, img_right_s, time, false);
                }
            } else {
                is_kf_req = pvisualfrontend_->visualTracking(img_left, time);
            }

            // Save current pose
            Logger::addSE3Pose(time, pcurframe_->getTwc(), is_kf_req);//这个先不修改了，就保存左目位姿

            if( pslamstate_->breset_req_ ) {
                reset(isleft);// mono_stereo
                continue;
            }

            // 2. Create new KF if req. / Send new KF to Mapper
            // ================================================
            if( is_kf_req )
            {
                if( pslamstate_->debug_ )
                    std::cout << "\n \t >>> [SLAM Node] New Keyframe send to Back-End\n";

                if( pslamstate_->stereo_ )
                {
                    Keyframe kf(
                            pcurframe_->kfid_,
                            img_left,
                            img_right,
                            pvisualfrontend_->cur_pyr_
                    );

                    pmapper_->addNewKf(kf);
                }
                else if( pslamstate_->mono_ )
                {
                    Keyframe kf(pcurframe_->kfid_, img_left);
                    pmapper_->addNewKf(kf);
                }
                else if( pslamstate_->mono_stereo_)// mono_stereo
                {
                    Keyframe kf(
                            pcurframe_->kfid_,
                            img_left,
                            img_right,
                            img_left_m,
                            img_left_s,
                            img_right_m,
                            img_right_s,
                            pvisualfrontend_->cur_pyr_,
                            pvisualfrontend_->cur_pyrl_,
                            pvisualfrontend_->cur_pyrr_,
                            pvisualfrontend_->cur_pyrlm_,
                            pvisualfrontend_->cur_pyrls_,
                            pvisualfrontend_->cur_pyrrm_,
                            pvisualfrontend_->cur_pyrrs_
                    );

                    pmapper_->addNewKf(kf);
                }

                if( !bkf_viz_ison_ ) {
                    std::thread kf_viz_thread(&SlamManager::visualizeAtKFsRate, this, time);// todo mono_stereo 共视关键帧可视化、关键帧轨迹结果可视化
                    kf_viz_thread.detach();
                }
            }

            if( pslamstate_->debug_ || pslamstate_->log_timings_ )
                std::cout << Profiler::getInstance().displayTimeLogs() << std::endl;

            // Frame rate visualization (limit the visualization processing)
            if( !bframe_viz_ison_ ) {
                std::thread viz_thread(&SlamManager::visualizeAtFrameRate, this, time);// mono_stereo 图像帧可视化、VO轨迹可视化、点云可视化。和上面的可视化更新频率不一样。
                viz_thread.detach();
            }
        }
        else {

            // 3. Check if we are done with a sequence!
            // ========================================
            bool c1 = cam_delay > 0;
            bool c2 = ( ros::Time::now().toSec() - last_img_time ) > 100. * cam_delay;
            bool c3 = !bnew_img_available_;

            if( c1 && c2 && c3 )
            {
                bexit_required_ = true;

                // Warn threads to stop and then save the results only in this case of
                // automatic stop because end of sequence reached
                // (avoid wasting time when forcing stop by CTRL+C)
                pmapper_->bexit_required_ = true;

                writeResults();// todo mono_stereo

                // Notify exit to ROS
                ros::requestShutdown();
            }
            else {
                std::chrono::milliseconds dura(1);
                std::this_thread::sleep_for(dura);
            }
        }
    }

    std::cout << "\nOV²SLAM is stopping!\n";

    bis_on_ = false;
}

void SlamManager::addNewMonoImage(const double time, cv::Mat &im0)
{
    if( pslamstate_->bdo_undist_ ) {
        pcalib_model_left_->rectifyImage(im0, im0);
    }

    std::lock_guard<std::mutex> lock(img_mutex_);
    qimg_left_.push(im0);
    qimg_time_.push(time);

    bnew_img_available_ = true;
}

void SlamManager::addNewStereoImages(const double time, cv::Mat &im0, cv::Mat &im1) 
{
    if( pslamstate_->bdo_stereo_rect_ || pslamstate_->bdo_undist_ ) {
        pcalib_model_left_->rectifyImage(im0, im0);
        pcalib_model_right_->rectifyImage(im1, im1);
    }

    std::lock_guard<std::mutex> lock(img_mutex_);
    qimg_left_.push(im0);
    qimg_right_.push(im1);
    qimg_time_.push(time);

    bnew_img_available_ = true;
}

Eigen::Vector2d SlamManager::projectToRealCamera(const Eigen::Vector3d& point3D, const Eigen::Matrix3d& realCameraK)
{
    Eigen::Vector3d pixel = realCameraK * point3D;
    return Eigen::Vector2d(pixel(0) / pixel(2), pixel(1) / pixel(2));
}

cv::Mat SlamManager::generateVirtualImage(const cv::Mat& realImage, const int& virtualImageWidth, const int& virtualImageHeight, const Eigen::Matrix3d& R, const Eigen::Matrix3d& virtualCameraK, const Eigen::Matrix3d& realCameraK, const Eigen::Vector3d& T, const bool& cuda)
{
    cv::Mat virtualImage = cv::Mat::zeros(virtualImageHeight, virtualImageWidth, realImage.type());

//    std::cout<<"cuda = "<<cuda<<std::endl;
    if (cuda) {
        // 将 Eigen 矩阵转换为数组以传递给 CUDA
        double R_array[9], virtualCameraK_array[9], realCameraK_array[9], T_array[3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R_array[i * 3 + j] = R(i, j);
                virtualCameraK_array[i * 3 + j] = virtualCameraK(i, j);
                realCameraK_array[i * 3 + j] = realCameraK(i, j);
            }
            T_array[i] = T(i);
        }

        // 调用 CUDA 加速的虚拟图像生成函数
        generateVirtualImageCUDA(realImage.ptr<unsigned char>(), virtualImage.ptr<unsigned char>(),
                                 virtualImageWidth, virtualImageHeight, realImage.cols, realImage.rows,
                                 R_array, virtualCameraK_array, realCameraK_array, T_array);
    } else {//不用cuda，遍历像素很慢
        //    int count_=0;
//    cv::namedWindow("virtualImage", cv::WINDOW_NORMAL);//设置一个特定窗口，一直保持打开状态，每次循环迭代更新这个窗口的图像
        for (int v_y = 0; v_y < virtualImageHeight; ++v_y) {
            for (int v_x = 0; v_x < virtualImageWidth; ++v_x) {
//            if(count_<148390)
//            {
                // 虚拟相机的像素坐标归一化
                Eigen::Vector3d point3D;
                point3D << (v_x - virtualCameraK(0, 2)) / virtualCameraK(0, 0),
                        (v_y - virtualCameraK(1, 2)) / virtualCameraK(1, 1),
                        1.0;
                // 转换到真实相机的坐标系
//                Eigen::Vector3d point3DReal = R.transpose() * (point3D - T);// error
                Eigen::Vector3d point3DReal = R * (point3D - T);

                // 投影到真实相机的像素坐标
                Eigen::Vector2d realPixel = projectToRealCamera(point3DReal, realCameraK);
                // 检查坐标是否在真实图像范围内
                int pixelX = static_cast<int>(realPixel.x());
                int pixelY = static_cast<int>(realPixel.y());
                if (pixelX >= 0 && pixelX < realImage.cols && pixelY >= 0 && pixelY < realImage.rows) {
//                    virtualImage.at<cv::Vec3b>(v_y, v_x) = realImage.at<cv::Vec3b>(pixelY, pixelX);
                    virtualImage.at<uchar>(v_y, v_x) = realImage.at<uchar>(pixelY, pixelX);//灰度图，用uchar
                }
//            count_++;
//            std::cout << "count = " << count_ << std::endl;

//            if (count_ % 1000 == 0) {// 每1000次更新一次窗口中的图像
//                cv::imshow("virtualImage", virtualImage);
//                cv::waitKey(1); // 0 等待用户按键关闭窗口 1 短暂等待以刷新窗口
//            }
//        }
            }
        }
    }
    return virtualImage;
}

//bool SlamManager::getNewImage(cv::Mat &iml, cv::Mat &imr, double &time)
bool SlamManager::getNewImage(cv::Mat &iml, cv::Mat &imr, cv::Mat &iml_m, cv::Mat &iml_s, cv::Mat &imr_m, cv::Mat &imr_s, double &time)
{
    std::lock_guard<std::mutex> lock(img_mutex_);

    if( !bnew_img_available_ ) {
        return false;
    }

    int k = 0;

    do {
        k++;

        iml = qimg_left_.front();
        qimg_left_.pop();

        time = qimg_time_.front();
        qimg_time_.pop();
        
        if( pslamstate_->stereo_ ) {
            imr = qimg_right_.front();
            qimg_right_.pop();
        }

        if( !pslamstate_->bforce_realtime_ )
            break;

    } while( !qimg_left_.empty() );

    if( k > 1 ) {    
        if( pslamstate_->debug_ )
            std::cout << "\n SLAM is late!  Skipped " << k-1 << " frames...\n";
    }
    
    if( qimg_left_.empty() ) {
        bnew_img_available_ = false;
    }

    if( pslamstate_->mono_stereo_ ) {
//    if( 0 ) {
        imr = qimg_right_.front();
        qimg_right_.pop();
        //mono_stereo
        Eigen::Vector3d T(0, 0, 0);
        Eigen::Matrix3d realCameraK;
        realCameraK << pslamstate_->fxl_, 0, pslamstate_->img_left_w_ / 2,
                0, pslamstate_->fyl_, pslamstate_->img_left_h_ / 2,
                0, 0, 1;
        Eigen::Matrix3d virtualCameraK_m;
        virtualCameraK_m << pslamstate_->fxl_, 0, pslamstate_->img_leftm_w_ / 2,
                0, pslamstate_->fyl_, pslamstate_->img_leftm_h_ / 2,
                0, 0, 1;
        Eigen::Matrix3d virtualCameraK_s;
        virtualCameraK_s << pslamstate_->fxl_, 0, pslamstate_->img_lefts_w_ / 2,
                0, pslamstate_->fyl_, pslamstate_->img_lefts_h_ / 2,
                0, 0, 1;

        iml_m = generateVirtualImage(iml, pslamstate_->img_leftm_w_, pslamstate_->img_leftm_h_, pslamstate_->R_ml, virtualCameraK_m, realCameraK, T, pslamstate_->use_cuda_);
        iml_s = generateVirtualImage(iml, pslamstate_->img_lefts_w_, pslamstate_->img_lefts_h_, pslamstate_->R_sl, virtualCameraK_s, realCameraK, T, pslamstate_->use_cuda_);
        imr_s = generateVirtualImage(imr, pslamstate_->img_lefts_w_, pslamstate_->img_lefts_h_, pslamstate_->R_sr, virtualCameraK_s, realCameraK, T, pslamstate_->use_cuda_);
        imr_m = generateVirtualImage(imr, pslamstate_->img_leftm_w_, pslamstate_->img_leftm_h_, pslamstate_->R_mr, virtualCameraK_m, realCameraK, T, pslamstate_->use_cuda_);

//        cv::imshow("iml", iml);
//        cv::imshow("imr", imr);
//        cv::imshow("iml_m", iml_m);
//        cv::imshow("iml_s", iml_s);
//        cv::imshow("imr_m", imr_m);
//        cv::imshow("imr_s", imr_s);
        cv::Mat leftRightConcat;
        cv::hconcat(iml, imr, leftRightConcat);// 左右拼接 iml 和 imr

        std::vector<cv::Mat> imagesToConcat = {iml_m, iml_s, imr_s, imr_m};
        cv::Mat multiConcat;
        cv::hconcat(imagesToConcat, multiConcat);// 依次拼接 iml_m, iml_s, imr_s, imr_m

        // 可视化拼接结果
//        cv::imshow("Left-Right Concat", leftRightConcat);
//        cv::imshow("Multi Concat", multiConcat);
//        cv::waitKey(1);
    }

    return true;
}

bool SlamManager::getNewImage(cv::Mat &iml, cv::Mat &imr, cv::Mat &iml_m, cv::Mat &iml_s, cv::Mat &imr_m, cv::Mat &imr_s, double &time, bool isleft)
{
    std::lock_guard<std::mutex> lock(img_mutex_);

    if( !bnew_img_available_ ) {
        return false;
    }

    int k = 0;

    do {
        k++;

        iml = qimg_left_.front();
        qimg_left_.pop();

        time = qimg_time_.front();
        qimg_time_.pop();

        if( pslamstate_->stereo_ ) {
            imr = qimg_right_.front();
            qimg_right_.pop();
        }

        if( !pslamstate_->bforce_realtime_ )
            break;

    } while( !qimg_left_.empty() );

    if( k > 1 ) {
        if( pslamstate_->debug_ )
            std::cout << "\n SLAM is late!  Skipped " << k-1 << " frames...\n";
    }

    if( qimg_left_.empty() ) {
        bnew_img_available_ = false;
    }

    if( pslamstate_->mono_stereo_ ) {
//    if( 0 ) {
        imr = qimg_right_.front();
        qimg_right_.pop();
        //mono_stereo
        Eigen::Vector3d T(0, 0, 0);
        Eigen::Matrix3d realCameraK;
        realCameraK << pslamstate_->fxl_, 0, pslamstate_->img_left_w_ / 2,
                0, pslamstate_->fyl_, pslamstate_->img_left_h_ / 2,
                0, 0, 1;
        Eigen::Matrix3d virtualCameraK_m;
        virtualCameraK_m << pslamstate_->fxl_, 0, pslamstate_->img_leftm_w_ / 2,
                0, pslamstate_->fyl_, pslamstate_->img_leftm_h_ / 2,
                0, 0, 1;
        Eigen::Matrix3d virtualCameraK_s;
        virtualCameraK_s << pslamstate_->fxl_, 0, pslamstate_->img_lefts_w_ / 2,
                0, pslamstate_->fyl_, pslamstate_->img_lefts_h_ / 2,
                0, 0, 1;

        if(isleft){
            iml_m = generateVirtualImage(iml, pslamstate_->img_leftm_w_, pslamstate_->img_leftm_h_, pslamstate_->R_ml, virtualCameraK_m, realCameraK, T, pslamstate_->use_cuda_);
            iml_s = generateVirtualImage(iml, pslamstate_->img_lefts_w_, pslamstate_->img_lefts_h_, pslamstate_->R_sl, virtualCameraK_s, realCameraK, T, pslamstate_->use_cuda_);

        } else {
            imr_s = generateVirtualImage(imr, pslamstate_->img_lefts_w_, pslamstate_->img_lefts_h_, pslamstate_->R_sr, virtualCameraK_s, realCameraK, T, pslamstate_->use_cuda_);
            imr_m = generateVirtualImage(imr, pslamstate_->img_leftm_w_, pslamstate_->img_leftm_h_, pslamstate_->R_mr, virtualCameraK_m, realCameraK, T, pslamstate_->use_cuda_);
        }

        // 可视化拼接结果
//        if(isleft){
//            cv::imshow("iml", iml);
//            cv::imshow("iml_m", iml_m);
//            cv::imshow("iml_s", iml_s);
//            cv::waitKey(1);
//        } else {
//            cv::imshow("imr", imr);
//            cv::imshow("imr_m", imr_m);
//            cv::imshow("imr_s", imr_s);
//            cv::waitKey(1);
//        }
//        cv::Mat leftRightConcat;
//        cv::hconcat(iml, imr, leftRightConcat);// 左右拼接 iml 和 imr
//        std::vector<cv::Mat> imagesToConcat = {iml_m, iml_s, imr_s, imr_m};
//        cv::Mat multiConcat;
//        cv::hconcat(imagesToConcat, multiConcat);// 依次拼接 iml_m, iml_s, imr_s, imr_m
//        cv::imshow("Left-Right Concat", leftRightConcat);
//        cv::imshow("Multi Concat", multiConcat);
//        cv::waitKey(1);
    }

    return true;
}

void SlamManager::setupCalibration()
{
    pcalib_model_left_.reset( 
                new CameraCalibration(
                        pslamstate_->cam_left_model_, 
                        pslamstate_->fxl_, pslamstate_->fyl_, 
                        pslamstate_->cxl_, pslamstate_->cyl_,
                        pslamstate_->k1l_, pslamstate_->k2l_, 
                        pslamstate_->p1l_, pslamstate_->p2l_,
                        pslamstate_->img_left_w_, 
                        pslamstate_->img_left_h_,
                        pslamstate_->img_w_crop_
                        ) 
                    );

    if( pslamstate_->stereo_ )
    {
        pcalib_model_right_.reset( 
                    new CameraCalibration(
                            pslamstate_->cam_right_model_, 
                            pslamstate_->fxr_, pslamstate_->fyr_, 
                            pslamstate_->cxr_, pslamstate_->cyr_,
                            pslamstate_->k1r_, pslamstate_->k2r_, 
                            pslamstate_->p1r_, pslamstate_->p2r_,
                            pslamstate_->img_right_w_, 
                            pslamstate_->img_right_h_,
                            pslamstate_->img_w_crop_
                            ) 
                        );
        
        // TODO: Change this and directly add the extrinsic parameters within the 
        // constructor (maybe set default parameters on extrinsic with identity / zero)
        pcalib_model_right_->setupExtrinsic(pslamstate_->T_left_right_);
    }
    if( pslamstate_->mono_stereo_ )
    {
        pcalib_model_right_.reset(
                new CameraCalibration(
                        pslamstate_->cam_right_model_,
                        pslamstate_->fxr_, pslamstate_->fyr_,
                        pslamstate_->cxr_, pslamstate_->cyr_,
                        pslamstate_->k1r_, pslamstate_->k2r_,
                        pslamstate_->p1r_, pslamstate_->p2r_,
                        pslamstate_->img_right_w_,
                        pslamstate_->img_right_h_,
                        pslamstate_->img_w_crop_
                )
        );
        pcalib_model_right_->setupExtrinsic(pslamstate_->T_left_right_);//左目 右目 变换

        //111-虚拟相机内外参，先大致给一个。验证test/test_virtual_cam2.cpp：将真实图像像素点投影到3d空间，然后投影到虚拟相机上，对比图像看是否一致，虚拟相机图像和真实相机一半的图像应该一样
        pcalib_model_left_mono_.reset(
                new CameraCalibration(
                        pslamstate_->cam_left_model_,
                        pslamstate_->fxr_, pslamstate_->fyr_,
                        pslamstate_->img_leftm_w_/2, pslamstate_->img_leftm_h_/2,
                        pslamstate_->k1r_, pslamstate_->k2r_,
                        pslamstate_->p1r_, pslamstate_->p2r_,
                        pslamstate_->img_leftm_w_,
                        pslamstate_->img_leftm_h_,
                        pslamstate_->img_w_crop_
                )
        );
        pcalib_model_left_mono_->setupExtrinsic(pslamstate_->T_left_leftm_);//左目 左单目区虚拟相机 变换

        pcalib_model_right_mono_.reset(
                new CameraCalibration(
                        pslamstate_->cam_right_model_,
                        pslamstate_->fxr_, pslamstate_->fyr_,
                        pslamstate_->img_rightm_w_/2, pslamstate_->img_rightm_h_/2,
                        pslamstate_->k1r_, pslamstate_->k2r_,
                        pslamstate_->p1r_, pslamstate_->p2r_,
                        pslamstate_->img_rightm_w_,
                        pslamstate_->img_rightm_h_,
                        pslamstate_->img_w_crop_
                )
        );
        pcalib_model_right_mono_->setupExtrinsic(pslamstate_->T_right_rightm_);//右目 右单目区虚拟相机 变换

        pcalib_model_left_stereo_.reset(
                new CameraCalibration(
                        pslamstate_->cam_left_model_,
                        pslamstate_->fxr_, pslamstate_->fyr_,
                        pslamstate_->img_lefts_w_/2, pslamstate_->img_lefts_h_/2,
                        pslamstate_->k1r_, pslamstate_->k2r_,
                        pslamstate_->p1r_, pslamstate_->p2r_,
                        pslamstate_->img_lefts_w_,
                        pslamstate_->img_lefts_h_,
                        pslamstate_->img_w_crop_
                )
        );
        pcalib_model_left_stereo_->setupExtrinsic(pslamstate_->T_left_lefts_);//左目 左双目区虚拟相机 变换

        pcalib_model_right_stereo_.reset(
                new CameraCalibration(
                        pslamstate_->cam_right_model_,
                        pslamstate_->fxr_, pslamstate_->fyr_,
                        pslamstate_->img_rights_w_/2, pslamstate_->img_rights_h_/2,
                        pslamstate_->k1r_, pslamstate_->k2r_,
                        pslamstate_->p1r_, pslamstate_->p2r_,
                        pslamstate_->img_rights_w_,
                        pslamstate_->img_rights_h_,
                        pslamstate_->img_w_crop_
                )
        );
        pcalib_model_right_stereo_->setupExtrinsic(pslamstate_->T_right_rights_);//右目 右双目区虚拟相机 变换
    }
}

void SlamManager::setupStereoCalibration()
{
    // Apply stereorectify and setup the calibration models
    cv::Mat Rl, Rr, Pl, Pr, Q;

    cv::Rect rectleft, rectright;

    if( pcalib_model_left_->model_ != pcalib_model_right_->model_ )
    {
        std::cerr << "\n Left and Right cam have different distortion model.  Cannot use stereo rectifcation!\n";
        return;
    }

    if( cv::countNonZero(pcalib_model_left_->Dcv_) == 0 && 
        cv::countNonZero(pcalib_model_right_->Dcv_) == 0 &&
        pcalib_model_right_->Tc0ci_.rotationMatrix().isIdentity(1.e-5) )
    {
        std::cout << "\n No distorsion and R_left_right = I3x3 / NO rectif to apply!";
        return;
    }

    if( pcalib_model_left_->model_ == CameraCalibration::Pinhole )
    {
        cv::stereoRectify(
                pcalib_model_left_->Kcv_, pcalib_model_left_->Dcv_,
                pcalib_model_right_->Kcv_, pcalib_model_right_->Dcv_,
                pcalib_model_left_->img_size_, 
                pcalib_model_right_->Rcv_cic0_, 
                pcalib_model_right_->tcv_cic0_,
                Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY, 
                pslamstate_->alpha_,
                pcalib_model_left_->img_size_, 
                &rectleft, &rectright
                );
    }
    else 
    {
        cv::fisheye::stereoRectify(
                pcalib_model_left_->Kcv_, pcalib_model_left_->Dcv_,
                pcalib_model_right_->Kcv_, pcalib_model_right_->Dcv_,
                pcalib_model_left_->img_size_, 
                pcalib_model_right_->Rcv_cic0_, 
                pcalib_model_right_->tcv_cic0_,
                Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY, 
                pcalib_model_left_->img_size_, 
                pslamstate_->alpha_
                );
        
        rectleft = cv::Rect(0, 0, pcalib_model_left_->img_w_, pcalib_model_left_->img_h_);
        rectright = cv::Rect(0, 0, pcalib_model_right_->img_w_, pcalib_model_right_->img_h_);
    }

    std::cout << "\n Alpha : " << pslamstate_->alpha_;

    std::cout << "\n Kl : \n" << pcalib_model_left_->Kcv_;
    std::cout << "\n Kr : \n" << pcalib_model_right_->Kcv_;

    std::cout << "\n Dl : \n" << pcalib_model_left_->Dcv_;
    std::cout << "\n Dr : \n" << pcalib_model_right_->Dcv_;

    std::cout << "\n Rl : \n" << Rl;
    std::cout << "\n Rr : \n" << Rr;
    
    std::cout << "\n Pl : \n" << Pl;
    std::cout << "\n Pr : \n" << Pr;

    // % OpenCV can handle left-right or up-down camera arrangements
    // isVerticalStereo = abs(RCT.P2(2,4)) > abs(RCT.P2(1,4));

    pcalib_model_left_->setUndistStereoMap(Rl, Pl, rectleft);
    pcalib_model_right_->setUndistStereoMap(Rr, Pr, rectright);

    // SLAM state keeps track of the initial intrinsic
    // parameters (perhaps to be used for optim...)
    pslamstate_->fxl_ = pcalib_model_left_->fx_;
    pslamstate_->fyl_ = pcalib_model_left_->fy_;
    pslamstate_->cxl_ = pcalib_model_left_->cx_;
    pslamstate_->cyl_ = pcalib_model_left_->cy_;

    pslamstate_->fxr_ = pcalib_model_right_->fx_;
    pslamstate_->fyr_ = pcalib_model_right_->fy_;
    pslamstate_->cxr_ = pcalib_model_right_->cx_;
    pslamstate_->cyr_ = pcalib_model_right_->cy_;
}

void SlamManager::reset()
{
    std::cout << "\n=======================================\n";
    std::cout << "\t RESET REQUIRED!";
    std::cout << "\n=======================================\n";

    pcurframe_->reset();
    pcurframe_l_->reset();
    pcurframe_r_->reset();
    pcurframe_lm_->reset();
    pcurframe_ls_->reset();
    pcurframe_rm_->reset();
    pcurframe_rs_->reset();
    pvisualfrontend_->reset();
    pmap_->reset();
    pmap_l_->reset();
    pmap_r_->reset();
    pmap_lm_->reset();
    pmap_ls_->reset();
    pmap_rm_->reset();
    pmap_rs_->reset();
    pmapper_->reset();

    pslamstate_->reset();
    Logger::reset();

    frame_id_ = -1;

    std::lock_guard<std::mutex> lock(img_mutex_);
    
    qimg_left_ = std::queue<cv::Mat>(); 
    qimg_right_ = std::queue<cv::Mat>();
    qimg_time_ = std::queue<double>();

    bnew_img_available_ = false;
    
    std::cout << "\n=======================================\n";
    std::cout << "\t RESET APPLIED!";
    std::cout << "\n=======================================\n";
}

void SlamManager::reset(bool isleft)
{
    std::cout << "\n=======================================\n";
    std::cout << "\t RESET REQUIRED!";
    std::cout << "\n=======================================\n";

    pcurframe_->reset();
    if(isleft){
        pcurframe_l_->reset();
        pcurframe_lm_->reset();
        pcurframe_ls_->reset();
        pmap_l_->reset();
        pmap_lm_->reset();
        pmap_ls_->reset();
    } else {
        pcurframe_r_->reset();
        pcurframe_rm_->reset();
        pcurframe_rs_->reset();
        pmap_r_->reset();
        pmap_rm_->reset();
        pmap_rs_->reset();
    }

    pvisualfrontend_->reset();
//    pmap_->reset();

    pmapper_->reset();

    pslamstate_->reset();
    Logger::reset();

    frame_id_ = -1;

    std::lock_guard<std::mutex> lock(img_mutex_);

    qimg_left_ = std::queue<cv::Mat>();
    qimg_right_ = std::queue<cv::Mat>();
    qimg_time_ = std::queue<double>();

    bnew_img_available_ = false;

    std::cout << "\n=======================================\n";
    std::cout << "\t RESET APPLIED!";
    std::cout << "\n=======================================\n";
}


// ==========================
//   Visualization functions
// ==========================

void SlamManager::visualizeAtFrameRate(const double time) 
{
    bframe_viz_ison_ = true;

    if( pslamstate_->mono_stereo_){
        visualizeFrame(pvisualfrontend_->cur_img_, time);//原算法
        visualizeFrame_left(pvisualfrontend_->cur_imgl_, time);
        visualizeFrame_right(pvisualfrontend_->cur_imgr_, time);
    } else {
        visualizeFrame(pvisualfrontend_->cur_img_, time);
    }
    visualizeVOTraj(time);
    prosviz_->pubPointCloud(pmap_->pcloud_, time);
    prosviz_->pubPointCloud_left(pmap_l_->pcloud_, time);
    prosviz_->pubPointCloud_right(pmap_r_->pcloud_, time);
    prosviz_->pubPointCloud_l_to_r(pmap_l_to_r_->pcloud_, time);
    prosviz_->pubPointCloud_r_to_l(pmap_r_to_l_->pcloud_, time);

    bframe_viz_ison_ = false;
}

void SlamManager::visualizeAtKFsRate(const double time)
{
    bkf_viz_ison_ = true;

    visualizeCovisibleKFs(time);// todo mono_stereo
    visualizeFullKFsTraj(time);// todo mono_stereo

    bkf_viz_ison_ = false;
}


void SlamManager::visualizeFrame(const cv::Mat &imleft, const double time)
{
    if( prosviz_->pub_image_track_.getNumSubscribers() == 0 ) {
        return;
    }

    // Display keypoints
    cv::Mat img_2_pub;
    cv::cvtColor(imleft, img_2_pub, CV_GRAY2RGB);

    for( const auto &kp : pcurframe_->getKeypoints() ) {
        cv::Scalar col;

        if(kp.is_retracked_) {
            if(kp.is3d_) {
                col = cv::Scalar(0,255,0);
            } else {
                col = cv::Scalar(235, 235, 52);
            } 
        } else if(kp.is3d_) {
            col = cv::Scalar(255,0,0);
        } else {
            col = cv::Scalar(0,0,255);
        }

        cv::circle(img_2_pub, kp.px_, 4, col, -1);
    }

    prosviz_->pubTrackImage(img_2_pub, time);
}

void SlamManager::visualizeFrame_left(const cv::Mat &imleft, const double time)
{
    if( prosviz_->pub_image_track_l_.getNumSubscribers() == 0 ) {
        return;
    }

    // Display keypoints
    cv::Mat img_2_pub;
    if (! imleft.empty()) {
        cv::cvtColor(imleft, img_2_pub, CV_GRAY2RGB);
    }

    //输出左目的特征点数量
//    std::cout << "left keypoints size = " << pcurframe_l_->getKeypoints().size() << std::endl;
    for( const auto &kp : pcurframe_l_->getKeypoints() ) {
        cv::Scalar col;

        if(kp.is_retracked_) { // matchingToLocalMap——mergeMatches2——mergeMapPoints——updateKeypointId
            if(kp.is3d_) {
                col = cv::Scalar(0,255,0);//绿色,正在被跟踪的3D点
            } else {
                col = cv::Scalar(235, 235, 52);//黄色,正在被跟踪的2D点
            }
        } else if(kp.is3d_) { // triangulateStereo、triangulateTemporal——MapManager::updateMapPoint——Frame::turnKeypoint3d & MapPoint::setPoint
            col = cv::Scalar(255,0,0);//红色,3D点 （rviz中顺序rgb）
        } else {
            col = cv::Scalar(0,0,255);//蓝色,2D点
        }

        cv::circle(img_2_pub, kp.px_, 4, col, -1);
    }

    prosviz_->pubTrackImage(img_2_pub, prosviz_->pub_image_track_l_, time);
}

void SlamManager::visualizeFrame_right(const cv::Mat &imright, const double time)
{
    if( prosviz_->pub_image_track_r_.getNumSubscribers() == 0 ) {
        return;
    }

    // Display keypoints
    cv::Mat img_2_pub;
    if (! imright.empty()) {
        cv::cvtColor(imright, img_2_pub, CV_GRAY2RGB);
    }

    for( const auto &kp : pcurframe_r_->getKeypoints() ) {
        cv::Scalar col;

        if(kp.is_retracked_) {
            if(kp.is3d_) {
                col = cv::Scalar(0,255,0);
            } else {
                col = cv::Scalar(235, 235, 52);
            }
        } else if(kp.is3d_) {
            col = cv::Scalar(255,0,0);
        } else {
            col = cv::Scalar(0,0,255);
        }

        cv::circle(img_2_pub, kp.px_, 4, col, -1);
    }

    prosviz_->pubTrackImage(img_2_pub, prosviz_->pub_image_track_r_, time);
}


void SlamManager::visualizeVOTraj(const double time)
{
    prosviz_->pubVO(pcurframe_l_->getTwc(), time, pslamstate_->T_right_left_);// todo stage2 只用左目位姿结果
    prosviz_->pubVO_r(pcurframe_r_->getTwc(), time, pslamstate_->T_right_left_);// todo stage2 只用右目位姿结果（现在每个frame存的位姿都是转换到左目的，就算使用的右目图像，右目得出的右目位姿也被转换到左相机了）
    //改成这俩都用吧
}


void SlamManager::visualizeCovisibleKFs(const double time)
{
    if( prosviz_->pub_kfs_pose_.getNumSubscribers() == 0 ) {
        return;
    }

//    for( const auto &covkf : pcurframe_->getCovisibleKfMap() ) {
//      auto pkf = pmap_->getKeyframe(covkf.first);
    for( const auto &covkf : pcurframe_l_->getCovisibleKfMap() ) { //
        auto pkf = pmap_l_->getKeyframe(covkf.first); //
        if( pkf != nullptr ) {
            prosviz_->addVisualKF(pkf->getTwc());
        }
    }

    prosviz_->pubVisualKFs(time);
}


void SlamManager::visualizeFullKFsTraj(const double time)
{
    if( prosviz_->pub_kfs_traj_.getNumSubscribers() == 0 ) {
            return;
    }

    prosviz_->clearKFsTraj();
//    for( int i = 0 ; i <= pcurframe_->kfid_ ; i++ ) {
//        auto pkf = pmap_->getKeyframe(i);
    for( int i = 0 ; i <= pcurframe_l_->kfid_ ; i++ ) {   //
        auto pkf = pmap_l_->getKeyframe(i); //
        if( pkf != nullptr ) {
            prosviz_->addKFsTraj(pkf->getTwc());
        }
    }
    prosviz_->pubKFsTraj(time);
}


void SlamManager::visualizeFinalKFsTraj()
{
    if( prosviz_->pub_final_kfs_traj_.getNumSubscribers() == 0 ) {
        return;
    }

    for( int i = 0 ; i <= pcurframe_->kfid_ ; i++ ) {
        auto pkf = pmap_->getKeyframe(i);
        if( pkf != nullptr ) {
            prosviz_->pubFinalKFsTraj(pkf->getTwc(), pkf->img_time_);
        }
    }
}

// ==========================
// Write Results functions
// ==========================


void SlamManager::writeResults()
{
    // Make sure that nothing is running in the background
    while( pslamstate_->blocalba_is_on_ || pslamstate_->blc_is_on_ ) {
        std::chrono::seconds dura(1);
        std::this_thread::sleep_for(dura);
    }
    
    visualizeFullKFsTraj(pcurframe_->img_time_);

    // Write Trajectories files
    Logger::writeTrajectory("ov2slam_traj.txt");
    Logger::writeTrajectoryKITTI("ov2slam_traj_kitti.txt");

    for( const auto & kfid_pkf : pmap_->map_pkfs_ )
    {
        auto pkf = kfid_pkf.second;
        if( pkf != nullptr ) {
            Logger::addKfSE3Pose(pkf->img_time_, pkf->getTwc());
        }
    }
    Logger::writeKfsTrajectory("ov2slam_kfs_traj.txt");

    // Apply full BA on KFs + 3D MPs if required + save
    if( pslamstate_->do_full_ba_ ) 
    {
        std::lock_guard<std::mutex> lock(pmap_->map_mutex_);
        pmapper_->runFullBA();

        prosviz_->pubPointCloud(pmap_->pcloud_, ros::Time::now().toSec());
        visualizeFinalKFsTraj();

        for( const auto & kfid_pkf : pmap_->map_pkfs_ ) {
            auto pkf = kfid_pkf.second;
            if( pkf != nullptr ) {
                Logger::addKfSE3Pose(pkf->img_time_, pkf->getTwc());
            }
        }

        Logger::writeKfsTrajectory("ov2slam_fullba_kfs_traj.txt");
    }

    // Write full trajectories taking into account LC
    if( pslamstate_->buse_loop_closer_ )
    {
        writeFullTrajectoryLC();
    }
}


void SlamManager::writeFullTrajectoryLC()
{
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vTwc;
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vTpc;
    std::vector<bool> viskf;

    vTwc.reserve(Logger::vframepose_.size());
    vTpc.reserve(Logger::vframepose_.size());
    viskf.reserve(Logger::vframepose_.size());

    size_t kfid = 0;

    Sophus::SE3d Twc, Twkf;

    std::ofstream f;
    std::string filename = "ov2slam_full_traj_wlc.txt";

    std::cout << "\n Going to write the full trajectory w. LC into : " << filename << "\n";

    f.open(filename.c_str());
    f << std::fixed;

    float fid = 0.;

    for( auto & fr : Logger::vframepose_ )
    {
        if( !fr.iskf_ || (fr.iskf_ && !pmap_->map_pkfs_.count(kfid)) ) 
        {
            // Get frame's pose from relative pose w.r.t. prev frame
            Eigen::Map<Eigen::Vector3d> t(fr.tprev_cur_);
            Eigen::Map<Eigen::Quaterniond> q(fr.qprev_cur_);

            vTpc.push_back(Sophus::SE3d(q,t));

            Sophus::SE3d Tprevcur(q,t);

            Twc = Twc * Tprevcur;

            viskf.push_back(false);

        } else {

            // Get keyframe's pose from map manager
            auto pkf = pmap_->map_pkfs_.at(kfid);

            Twc = pkf->getTwc();

            Twkf = Twc;

            kfid++;

            viskf.push_back(true);

            Eigen::Map<Eigen::Vector3d> t(fr.tprev_cur_);
            Eigen::Map<Eigen::Quaterniond> q(fr.qprev_cur_);
            vTpc.push_back(Sophus::SE3d(q,t));
        }

        vTwc.push_back(Twc);

        Eigen::Vector3d twc = Twc.translation();
        Eigen::Quaterniond qwc = Twc.unit_quaternion();

        f << std::setprecision(9) << fid << " " << twc.x() << " " << twc.y() << " " << twc.z()
            << " " << qwc.x() << " " << qwc.y() << " " << qwc.z() << " " << qwc.w() << std::endl;

        f.flush();

        fid += 1.;
    }

    f.close();

    std::cout << "\nFull Trajectory w. LC file written!\n";
    
    // Apply full pose graph for optimal full trajectory w. LC
    pmapper_->pestimator_->poptimizer_->fullPoseGraph(vTwc, vTpc, viskf);
}