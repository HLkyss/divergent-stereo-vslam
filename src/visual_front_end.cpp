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

#include <opencv2/video/tracking.hpp>

#include "visual_front_end.hpp"
#include "multi_view_geometry.hpp"

#include <opencv2/highgui.hpp>

#include "switchMapKeypoint.cuh"


VisualFrontEnd::VisualFrontEnd(std::shared_ptr<SlamParams> pstate, std::shared_ptr<Frame> pframe, 
        std::shared_ptr<MapManager> pmap, std::shared_ptr<FeatureTracker> ptracker)
    : pslamstate_(pstate), pcurframe_(pframe), pmap_(pmap), ptracker_(ptracker)
{}

VisualFrontEnd::VisualFrontEnd(std::shared_ptr<SlamParams> pstate, std::shared_ptr<Frame> pframe, std::shared_ptr<Frame> pframe_l, std::shared_ptr<Frame> pframe_r, std::shared_ptr<Frame> pframe_lm, std::shared_ptr<Frame> pframe_ls, std::shared_ptr<Frame> pframe_rm, std::shared_ptr<Frame> pframe_rs,
                               std::shared_ptr<MapManager> pmap, std::shared_ptr<MapManager> pmap_l, std::shared_ptr<MapManager> pmap_r, std::shared_ptr<MapManager> pmap_lm, std::shared_ptr<MapManager> pmap_ls, std::shared_ptr<MapManager> pmap_rm, std::shared_ptr<MapManager> pmap_rs, std::shared_ptr<MapManager> pmap_l_to_r, std::shared_ptr<MapManager> pmap_r_to_l, std::shared_ptr<FeatureTracker> ptracker)
        : pslamstate_(pstate), pcurframe_(pframe), pcurframe_l_(pframe_l), pcurframe_r_(pframe_r), pcurframe_lm_(pframe_lm), pcurframe_ls_(pframe_ls), pcurframe_rm_(pframe_rm), pcurframe_rs_(pframe_rs), pmap_(pmap), pmap_l_(pmap_l), pmap_r_(pmap_r), pmap_lm_(pmap_lm), pmap_ls_(pmap_ls), pmap_rm_(pmap_rm), pmap_rs_(pmap_rs), pmap_l_to_r_(pmap_l_to_r), pmap_r_to_l_(pmap_r_to_l), ptracker_(ptracker) //mono_stereo
{}

bool VisualFrontEnd::visualTracking(cv::Mat &iml, double time)
{
    std::lock_guard<std::mutex> lock(pmap_->map_mutex_);
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("0.Full-Front_End");

    bool iskfreq = trackMono(iml, time);

    if( iskfreq ) {
        pmap_->createKeyframe(cur_img_, iml);

        if( pslamstate_->btrack_keyframetoframe_ ) {
            cv::buildOpticalFlowPyramid(cur_img_, kf_pyr_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
        }
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "0.Full-Front_End");

    return iskfreq;
}

bool VisualFrontEnd::visualTracking(cv::Mat &iml, cv::Mat &imr, cv::Mat &imlm, cv::Mat &imls, cv::Mat &imrm, cv::Mat &imrs, double time)//mono_stereo
{
    std::lock_guard<std::mutex> lock(pmap_->map_mutex_);// todo mono_stereo 锁
    std::lock_guard<std::mutex> lock_l(pmap_l_->map_mutex_);
    std::lock_guard<std::mutex> lock_r(pmap_r_->map_mutex_);
    std::lock_guard<std::mutex> lock_ls(pmap_ls_ ->map_mutex_);
    std::lock_guard<std::mutex> lock_rs(pmap_rs_->map_mutex_);
    std::lock_guard<std::mutex> lock_l_to_r(pmap_l_to_r_->map_mutex_);
    std::lock_guard<std::mutex> lock_r_to_l(pmap_r_to_l_->map_mutex_);

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("0.Full-Front_End");

    bool iskfreq = trackMonoStereo(iml, imr, imlm, imls, imrm, imrs, time);// todo mono_stereo

    //交换地图点，更新帧kp
//    std::cout<<"//--------------------------"<<std::endl;
//    std::cout<<"before switch: pmap_l has "<<pmap_l_->nblms_<<" lms, pmap_r has "<<pmap_r_->nblms_<<"lms, "<<pmap_l_->nblms_+pmap_r_->nblms_<<" in total"<<std::endl;
    switchMapKeypoint(pcurframe_l_, pcurframe_r_, pmap_l_, pmap_r_, pmap_l_to_r_, pmap_r_to_l_, iml, imr);
////    switchMapKeypoint_cuda(pcurframe_l_, pcurframe_r_, pmap_l_, pmap_r_, pmap_l_to_r_, pmap_r_to_l_, iml, imr);
//    std::cout<<"after switch: pmap_l has "<<pmap_l_->nblms_<<" lms, pmap_r has "<<pmap_r_->nblms_<<"lms, "<<pmap_l_->nblms_+pmap_r_->nblms_<<" in total"<<std::endl;
//    std::cout<<"--------------------------//"<<std::endl;

    if( iskfreq ) {
//        pmap_->createKeyframe(cur_img_, iml);// 原代码 todo mono_stereo 每个相机关键帧是一起创建的，不存在左目创建了但右目没创建的情况
//        std::cout<<"isleft = true"<<std::endl;
        pmap_l_->createKeyframe(cur_imgl_, iml, pcurframe_l_, true);
        pmap_r_->createKeyframe(cur_imgr_, imr, pcurframe_r_, false);
        pmap_->createKeyframe(cur_img_, iml, pcurframe_, false);//是否可以删掉？
//        pmap_ls_->createKeyframe_s(cur_imgls_, imls, pcurframe_ls_);
//        pmap_rs_->createKeyframe_s(cur_imgrs_, imrs, pcurframe_rs_);
        pmap_ls_->createKeyframe_s2(cur_imgls_, imls, pcurframe_ls_);
        pmap_rs_->createKeyframe_s2(cur_imgrs_, imrs, pcurframe_rs_);

//        std::cout<<"pcurframe_l_ nbwcells_s_"<<pcurframe_l_->nbwcells_s_<<std::endl;
//        std::cout<<"pmap_ nbwcells_ -----"<<pmap_->getKeyframe(0)->nbwcells_<<std::endl;
//        std::cout<<"pmap_l_ nbwcells_ -----"<<pmap_l_->getKeyframe(0)->nbwcells_<<std::endl;
//        std::cout<<"pmap_ nbwcells_s_ -----"<<pmap_->getKeyframe(0)->nbwcells_s_<<std::endl;//从这里开始就有问题了
//        std::cout<<"pmap_l_ nbwcells_s_ -----"<<pmap_l_->getKeyframe(0)->nbwcells_s_<<std::endl;//从这里开始就有问题了

        if( pslamstate_->btrack_keyframetoframe_ ) {
            cv::buildOpticalFlowPyramid(cur_img_, kf_pyr_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
            cv::buildOpticalFlowPyramid(cur_imgl_, kf_pyrl_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
            cv::buildOpticalFlowPyramid(cur_imgr_, kf_pyrr_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
            cv::buildOpticalFlowPyramid(cur_imglm_, kf_pyrlm_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
            cv::buildOpticalFlowPyramid(cur_imgls_, kf_pyrls_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
            cv::buildOpticalFlowPyramid(cur_imgrm_, kf_pyrrm_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
            cv::buildOpticalFlowPyramid(cur_imgrs_, kf_pyrrs_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
        }
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "0.Full-Front_End");

    return iskfreq;
}

bool VisualFrontEnd::visualTracking(cv::Mat &iml, cv::Mat &imr, cv::Mat &imlm, cv::Mat &imls, cv::Mat &imrm, cv::Mat &imrs, double time, bool isleft)//mono_stereo todo 2.0
{
//    std::lock_guard<std::mutex> lock(pmap_->map_mutex_);// todo mono_stereo 锁 一起锁了，应不应该分开？
    std::unique_ptr<std::lock_guard<std::mutex>> lock;
    if (isleft) {
        lock = std::make_unique<std::lock_guard<std::mutex>>(pmap_l_->map_mutex_);
    } else {
        lock = std::make_unique<std::lock_guard<std::mutex>>(pmap_r_->map_mutex_);
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("0.Full-Front_End");

    bool iskfreq = trackMonoStereo(iml, imr, imlm, imls, imrm, imrs, time, isleft);// todo mono_stereo

    if( iskfreq ) {
//        pmap_->createKeyframe(cur_img_, iml);// 原代码 todo mono_stereo
        if(isleft){
            pmap_l_->createKeyframe(cur_imgl_, iml, pcurframe_l_, true);
        } else {
            pmap_r_->createKeyframe(cur_imgr_, imr, pcurframe_r_, false);
        }
//        pmap_->createKeyframe(cur_img_, iml, pcurframe_);//和源代码效果一样

        if( pslamstate_->btrack_keyframetoframe_ ) {
            if(isleft){
                cv::buildOpticalFlowPyramid(cur_imgl_, kf_pyrl_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
                cv::buildOpticalFlowPyramid(cur_imglm_, kf_pyrlm_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
                cv::buildOpticalFlowPyramid(cur_imgls_, kf_pyrls_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
            } else {
                cv::buildOpticalFlowPyramid(cur_imgr_, kf_pyrr_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
                cv::buildOpticalFlowPyramid(cur_imgrm_, kf_pyrrm_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
                cv::buildOpticalFlowPyramid(cur_imgrs_, kf_pyrrs_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
            }
//            cv::buildOpticalFlowPyramid(cur_img_, kf_pyr_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
        }
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "0.Full-Front_End");

    return iskfreq;
}

// Perform tracking in one image, update kps and MP obs, return true if a new KF is req.
bool VisualFrontEnd::trackMono(cv::Mat &im, double time)
{
    if( pslamstate_->debug_ )
        std::cout << "\n\n - [Visual-Front-End]: Track Mono Image\n";
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.FE_Track-Mono");

    // Preprocess the new image
    preprocessImage(im);

    // Create KF if 1st frame processed
    if( pcurframe_->id_ == 0 ) {
        return true;
    }
    
    // Apply Motion model to predict cur Frame pose
    Sophus::SE3d Twc = pcurframe_->getTwc();
    motion_model_.applyMotionModel(Twc, time);
    pcurframe_->setTwc(Twc);
    
    // Track the new image
    if( pslamstate_->btrack_keyframetoframe_ ) {
        kltTrackingFromKF();
    } else {
        kltTracking();
    }

    if( pslamstate_->doepipolar_ ) {
        // Check2d2dOutliers
        epipolar2d2dFiltering();
    }

    if( pslamstate_->mono_ && !pslamstate_->bvision_init_ ) 
    {
        if( pcurframe_->nb2dkps_ < 50 ) {
            pslamstate_->breset_req_ = true;
            return false;
        } 
        else if( checkReadyForInit() ) {
            std::cout << "\n\n - [Visual-Front-End]: Mono Visual SLAM ready for initialization!";
            pslamstate_->bvision_init_ = true;
            return true;
        } 
        else {
            std::cout << "\n\n - [Visual-Front-End]: Not ready to init yet!";
            return false;
        }
    }

    // Compute Pose (2D-3D)
    computePose();

    // Update Motion model from estimated pose
    motion_model_.updateMotionModel(pcurframe_->Twc_, time);

    // Check if New KF req.
    bool is_kf_req = checkNewKfReq();

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.FE_Track-Mono");

    return is_kf_req;
}

void VisualFrontEnd::switchMapKeypoint(std::shared_ptr<Frame>& pframe_l,  std::shared_ptr<Frame>& pframe_r, std::shared_ptr<MapManager>& pmap_l, std::shared_ptr<MapManager>& pmap_r, std::shared_ptr<MapManager>& pmap_l_to_r, std::shared_ptr<MapManager>& pmap_r_to_l, cv::Mat &iml, cv::Mat &imr)
{
    //这个方法要遍历所有地图点，会很慢吧？
    //又想到一个点，如果地图点太多的话，可以删掉一部分,或者在投影的时候只选最近的一段时间内的地图点进行投影

    //用左地图更新右地图和右帧 一般在左转时触发
    for (const auto& it : pmap_l->map_plms_) {
        const std::shared_ptr<MapPoint>& map_point_l = it.second;
        // 获取当前MapPoint的3D坐标
        Eigen::Vector3d point_3d_l = map_point_l->getPoint();
        Sophus::SE3d Twc_r;
        //当用左目地图作为填充信息，是给右目填充，因此应该找在右目视野范围内的点
        Twc_r = pframe_r->getTwc() * pslamstate_->T_left_right_;// 右目到世界=左目到世界*右目到左目

        //比较地图点与相机的连线 和 相机朝向 的角度，如果大于fov/2，就是看不到的点，不用投影
        Eigen::Vector3d point_to_camera = point_3d_l - Twc_r.translation();// 计算相机到地图点的向量
        Eigen::Vector3d camera_direction = Twc_r.rotationMatrix().col(2); // 相机的朝向向量（相机坐标系的Z轴）
        // 计算地图点连线和相机朝向之间的夹角
        double dot_product = camera_direction.dot(point_to_camera);
        double camera_distance = point_to_camera.norm();
        double point_distance = point_to_camera.norm();
//        double cos_angle = dot_product / (camera_distance * point_distance);
        // 如果点积大于cos(FOV/2) * 距离乘积，则点在相机视野内
        std::vector<cv::Point2f> vnewpts;
        if (dot_product >= pslamstate_->cos_half_fov * camera_distance * point_distance) {// 如果在视野内，进行投影等操作
            //用左目地图补充时，将地图点投影到右目
            Eigen::Vector3d point_in_camera = Twc_r.inverse() * point_3d_l;//左目得到的地图点在右相机坐标系下的坐标
            cv::Point2f pt = pframe_r->projCamToImage(point_in_camera);//左地图点的右目投影

            // todotodo 数值不一样，不应该是一半
            if(pt.x > pslamstate_->img_w_crop_ && pt.y > 0 && pt.x < pslamstate_->img_left_w_ && pt.y < pslamstate_->img_left_h_){//如果投影点坐标在roi区域（不同于fov，差半个双目区）
//            if(pt.x > pslamstate_->img_lefts_w_/2. && pt.y > 0 && pt.x < pslamstate_->img_left_w_ && pt.y < pslamstate_->img_left_h_){//如果投影点坐标在roi区域（不同于fov，差半个双目区）
//                std::cout<<"oooooooooooooooooooooooooooooooooo"<<point_3d_l<<std::endl;
                vnewpts.push_back(pt);
                ///参考addKeypointsToFrame（addKeypointsToFrame会同时添加kp并新建地图点到map_plms_，但是保存的地图点(map_plms_中id为nlmid_-1的点)没有坐标信息，后面通过updateMapPoint等方法更新）
                pframe_r->addKeypoint(pt, pmap_r->nlmid_);//右目帧添加关键点
                cv::Scalar col = iml.at<uchar>(pt.y,pt.x);
                pmap_r->addMapPoint(col);//右目地图添加新地图点(无坐标信息)，并且nlmid_++：每次使用完，nlmid_都会+1，下次再使用时已经是+1之后的新值，再想访问当前点应该是nlmid_-1（前提是过程中没有别的地方再修改nlmid_）
                pmap_l_to_r->addMapPoint(col);//只用于可视化
                ///参考triangulateStereo——updateMapPoint，保存坐标
//                Eigen::Vector3d wpt = pmap_l_->getMapPoint(pmap_r_->nlmid_)->getPoint();
//                pmap_r_->updateMapPoint(pmap_r_->nlmid_, wpt, false, 1./left_pt.z());
                pmap_r->updateMapPoint(pmap_r->nlmid_-1, point_3d_l, false, 1./point_in_camera.z());//存右目逆深度
                pmap_l_to_r->updateMapPoint(pmap_l_to_r->nlmid_-1, point_3d_l, false, 1./point_in_camera.z());//只用于可视化
                ///参考Estimator::mapFiltering——removeMapPointObs，左目地图删除该点
                pmap_l->removeMapPointObs(map_point_l->lmid_, pframe_l->id_);//SlamManager::run——pcurframe_l_->updateFrame(frame_id_, time);——id_= id;
/*                auto covkf_map = pframe_l->getCovisibleKfMap();
                for( auto it = covkf_map.rbegin() ; it != covkf_map.rend() ; it++ ){
                    int kfid = it->first;
                    //原mapFiltering函数还有很多内容，这里需不需要？
                    auto pkf = pmap_l->getKeyframe(kfid);
                    if( pkf == nullptr ) {
                        pframe_l->removeCovisibleKf(kfid);
                        continue;
                    }
                    else if( (int)pkf->nb3dkps_ < pslamstate_->nmin_covscore_ / 2 ) {
                        std::lock_guard<std::mutex> lock(pmap_l->map_mutex_);
                        pmap_l->removeKeyframe(kfid);
                        continue;
                    }
                    pmap_l->removeMapPointObs(map_point_l->lmid_, kfid);
                }*/

            }
        } else {// 点在相机视野外，跳过
            continue;
        }

        //因为这里只涉及到地图点的转换，不用后端其他对地图点的操作（updateFrameCovisibility matchingToLocalMap），是不是可以单独在这里实现？
        //不对，共视图也要更新？
//        pmap_l_->updateFrameCovisibility(*pframe_l, true);// mono_stereo todo stage-new 不再单独创建右目地图
//        pmap_r_->updateFrameCovisibility(*pframe_r, false);

    }

    //用右地图更新左地图和左帧 一般在右转时触发
    for (const auto& it : pmap_r->map_plms_) {
        const std::shared_ptr<MapPoint>& map_point_r = it.second;
        // 获取当前MapPoint的3D坐标
        Eigen::Vector3d point_3d_r = map_point_r->getPoint();
        Sophus::SE3d Twc_l;
        //当用右目地图作为填充信息，是给左目填充，因此应该找在左目视野范围内的点
        Twc_l = pframe_l->getTwc();//左目到世界

        //比较地图点与相机的连线 和 相机朝向 的角度，如果大于fov/2，就是看不到的点，不用投影
        Eigen::Vector3d point_to_camera = point_3d_r - Twc_l.translation();// 计算相机到地图点的向量
        Eigen::Vector3d camera_direction = Twc_l.rotationMatrix().col(2); // 相机的朝向向量（相机坐标系的Z轴）
        // 计算地图点连线和相机朝向之间的夹角
        double dot_product = camera_direction.dot(point_to_camera);
        double camera_distance = point_to_camera.norm();
        double point_distance = point_to_camera.norm();
//        double cos_angle = dot_product / (camera_distance * point_distance);
        // 如果点积大于cos(FOV/2) * 距离乘积，则点在相机视野内
        std::vector<cv::Point2f> vnewpts;
        if (dot_product >= pslamstate_->cos_half_fov * camera_distance * point_distance) {// 如果在视野内，进行投影等操作
            //用右目地图补充时，将地图点投影到左目
            Eigen::Vector3d point_in_camera = Twc_l.inverse() * point_3d_r;//右目得到的地图点在左相机坐标系下的坐标
            cv::Point2f pt = pframe_l->projCamToImage(point_in_camera);//右地图点的左目投影

            // todotodo 数值不一样，不应该是一半
            if(pt.x > 0 && pt.y > 0 && pt.x < pslamstate_->img_left_w_ - pslamstate_->img_w_crop_ && pt.y < pslamstate_->img_left_h_){//如果投影点坐标在roi区域（不同于fov，差半个双目区）
//            if(pt.x > 0 && pt.y > 0 && pt.x < pslamstate_->img_left_w_ -pslamstate_->img_lefts_w_/2. && pt.y < pslamstate_->img_left_h_) {//如果投影点坐标在roi区域（不同于fov，差半个双目区）
//                std::cout<<"oooooooooooooooooooooooooooooooooo"<<point_3d_r<<std::endl;
                vnewpts.push_back(pt);
                ///参考addKeypointsToFrame（addKeypointsToFrame会同时添加kp并新建地图点到map_plms_，但是保存的地图点(map_plms_中id为nlmid_-1的点)没有坐标信息，后面通过updateMapPoint等方法更新）
                pframe_l->addKeypoint(pt, pmap_l->nlmid_);//右目帧添加关键点
                cv::Scalar col = imr.at<uchar>(pt.y, pt.x);
                pmap_l->addMapPoint(col);//右目地图添加新地图点(无坐标信息)，并且nlmid_++：每次使用完，nlmid_都会+1，下次再使用时已经是+1之后的新值，再想访问当前点应该是nlmid_-1（前提是过程中没有别的地方再修改nlmid_）
                pmap_r_to_l->addMapPoint(col);//只用于可视化
                ///参考triangulateStereo——updateMapPoint，保存坐标
                pmap_l->updateMapPoint(pmap_l->nlmid_ - 1, point_3d_r, true, 1. / point_in_camera.z());//存左目逆深度
                pmap_r_to_l->updateMapPoint(pmap_r_to_l->nlmid_ - 1, point_3d_r, true, 1. / point_in_camera.z());//只用于可视化
                ///参考Estimator::mapFiltering——removeMapPointObs，左目地图删除该点
                pmap_r->removeMapPointObs(map_point_r->lmid_, pframe_r->id_);//SlamManager::run——pcurframe_l_->updateFrame(frame_id_, time);——id_= id;
            }
        } else {// 点在相机视野外，跳过
            continue;
        }
    }

}

void VisualFrontEnd::switchMapKeypoint_cuda(std::shared_ptr<Frame>& pframe_l,  std::shared_ptr<Frame>& pframe_r, std::shared_ptr<MapManager>& pmap_l, std::shared_ptr<MapManager>& pmap_r, std::shared_ptr<MapManager>& pmap_l_to_r, std::shared_ptr<MapManager>& pmap_r_to_l, cv::Mat &iml, cv::Mat &imr)
{
    std::unordered_map<int, std::shared_ptr<MapPoint>> map_plms_l = pmap_l->map_plms_;

    // 将MapPoint转换为MapPointData数组
    std::vector<MapPointData> mapPoints;
    mapPoints.reserve(map_plms_l.size());
    for (const auto& it : map_plms_l) {
        const std::shared_ptr<MapPoint>& map_point_l = it.second;
        MapPointData data;
        data.lmid = map_point_l->lmid_;
        Eigen::Vector3d pt = map_point_l->getPoint();
        data.x = pt.x();
        data.y = pt.y();
        data.z = pt.z();
        mapPoints.push_back(data);
    }

    FrameData frame_r_data;
    Sophus::SE3d Twc_r = pframe_r->getTwc() * pslamstate_->T_left_right_;

    // 填充Twc和T_left_right
    for(int i = 0; i < 16; ++i){
        frame_r_data.Twc[i] = Twc_r.matrix()(i / 4, i % 4);
        frame_r_data.T_left_right[i] = pslamstate_->T_left_right_.matrix()(i / 4, i % 4);
    }

    frame_r_data.cos_half_fov = pslamstate_->cos_half_fov;
    frame_r_data.img_left_w = pslamstate_->img_left_w_;
    frame_r_data.img_left_h = pslamstate_->img_left_h_;
    frame_r_data.img_lefts_w = pslamstate_->img_lefts_w_;

    // 上传图像到设备内存
    unsigned char* d_iml;
    cudaMalloc(&d_iml, iml.rows * iml.cols * sizeof(unsigned char));
    cudaMemcpy(d_iml, iml.data, iml.rows * iml.cols * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // 输出的新关键点
    std::vector<float> newPoints; // 每个关键点占用两个float (x, y)
    int maxNewPoints = MAX_NEW_POINTS; // 预分配的最大关键点数量

    // 调用CUDA加速的函数
    switchMapKeypointCUDA(
            mapPoints,
            frame_r_data,
            d_iml,
            newPoints,
            maxNewPoints
    );

    // 释放设备图像内存
    cudaFree(d_iml);

    // 处理newPoints，例如添加到pframe_r
    int numNewPoints = newPoints.size() / 2;
    std::cout<<"numNewPoints = "<<numNewPoints<<std::endl;
    for (int i = 0; i < numNewPoints; ++i) {
        float x = newPoints[2 * i];
        float y = newPoints[2 * i + 1];
        cv::Point2f pt(x, y);

        // 添加关键点到Frame
        pframe_r->addKeypoint(pt, pmap_r->nlmid_);

        // 获取颜色（假设图像为灰度图）
        uchar col = iml.at<uchar>(static_cast<int>(y), static_cast<int>(x));

        // 添加MapPoint到pmap_r和pmap_l_to_r
        pmap_r->addMapPoint(cv::Scalar(col));
        pmap_l_to_r->addMapPoint(cv::Scalar(col));

        // 更新MapPoint，存右目逆深度
        Eigen::Vector3d point_in_camera = Twc_r.inverse() * map_plms_l[mapPoints[i].lmid]->getPoint();
        double inv_depth = 1.0 / point_in_camera.z();
        pmap_r->updateMapPoint(pmap_r->nlmid_ - 1, map_plms_l[mapPoints[i].lmid]->getPoint(), false, inv_depth);
        pmap_l_to_r->updateMapPoint(pmap_l_to_r->nlmid_ - 1, map_plms_l[mapPoints[i].lmid]->getPoint(), false, inv_depth);

        // 移除MapPoint观察
        pmap_l->removeMapPointObs(mapPoints[i].lmid, pframe_l->id_);
    }


//    for (const auto& it : pmap_l->map_plms_) {
//        const std::shared_ptr<MapPoint>& map_point_l = it.second;
//
//        Eigen::Vector3d point_3d_l = map_point_l->getPoint();
//        Sophus::SE3d Twc_r;
//
//        Twc_r = pframe_r->getTwc() * pslamstate_->T_left_right_;
//
//        Eigen::Vector3d point_to_camera = point_3d_l - Twc_r.translation();
//        Eigen::Vector3d camera_direction = Twc_r.rotationMatrix().col(2);
//
//        double dot_product = camera_direction.dot(point_to_camera);
//        double camera_distance = point_to_camera.norm();
//        double point_distance = point_to_camera.norm();
//
//        std::vector<cv::Point2f> vnewpts;
//        if (dot_product >= pslamstate_->cos_half_fov * camera_distance * point_distance) {
//
//            Eigen::Vector3d point_in_camera = Twc_r.inverse() * point_3d_l;
//            cv::Point2f pt = pframe_r->projCamToImage(point_in_camera);
//
//            if(pt.x > pslamstate_->img_lefts_w_/2. && pt.y > 0 && pt.x < pslamstate_->img_left_w_ && pt.y < pslamstate_->img_left_h_){
//                vnewpts.push_back(pt);
//                pframe_r->addKeypoint(pt, pmap_r->nlmid_);
//                cv::Scalar col = iml.at<uchar>(pt.y,pt.x);
//                pmap_r->addMapPoint(col);
//                pmap_l_to_r->addMapPoint(col);
//
//                pmap_r->updateMapPoint(pmap_r->nlmid_-1, point_3d_l, false, 1./point_in_camera.z());//存右目逆深度
//                pmap_l_to_r->updateMapPoint(pmap_l_to_r->nlmid_-1, point_3d_l, false, 1./point_in_camera.z());//只用于可视化
//
//                pmap_l->removeMapPointObs(map_point_l->lmid_, pframe_l->id_);
//
//            }
//        } else {
//            continue;
//        }
//
//    }

//    //用右地图更新左地图和左帧 一般在右转时触发
//    for (const auto& it : pmap_r->map_plms_) {
//        const std::shared_ptr<MapPoint>& map_point_r = it.second;
//        // 获取当前MapPoint的3D坐标
//        Eigen::Vector3d point_3d_r = map_point_r->getPoint();
//        Sophus::SE3d Twc_l;
//        //当用右目地图作为填充信息，是给左目填充，因此应该找在左目视野范围内的点
//        Twc_l = pframe_l->getTwc();//左目到世界
//
//        //比较地图点与相机的连线 和 相机朝向 的角度，如果大于fov/2，就是看不到的点，不用投影
//        Eigen::Vector3d point_to_camera = point_3d_r - Twc_l.translation();// 计算相机到地图点的向量
//        Eigen::Vector3d camera_direction = Twc_l.rotationMatrix().col(2); // 相机的朝向向量（相机坐标系的Z轴）
//        // 计算地图点连线和相机朝向之间的夹角
//        double dot_product = camera_direction.dot(point_to_camera);
//        double camera_distance = point_to_camera.norm();
//        double point_distance = point_to_camera.norm();
////        double cos_angle = dot_product / (camera_distance * point_distance);
//        // 如果点积大于cos(FOV/2) * 距离乘积，则点在相机视野内
//        std::vector<cv::Point2f> vnewpts;
//        if (dot_product >= pslamstate_->cos_half_fov * camera_distance * point_distance) {// 如果在视野内，进行投影等操作
//            //用右目地图补充时，将地图点投影到左目
//            Eigen::Vector3d point_in_camera = Twc_l.inverse() * point_3d_r;//右目得到的地图点在左相机坐标系下的坐标
//            cv::Point2f pt = pframe_l->projCamToImage(point_in_camera);//右地图点的左目投影
//
//            if(pt.x > 0 && pt.y > 0 && pt.x < pslamstate_->img_left_w_ -pslamstate_->img_w_crop_ && pt.y < pslamstate_->img_left_h_) {//如果投影点坐标在roi区域（不同于fov，差半个双目区）
////                std::cout<<"oooooooooooooooooooooooooooooooooo"<<point_3d_r<<std::endl;
//                vnewpts.push_back(pt);
//                ///参考addKeypointsToFrame（addKeypointsToFrame会同时添加kp并新建地图点到map_plms_，但是保存的地图点(map_plms_中id为nlmid_-1的点)没有坐标信息，后面通过updateMapPoint等方法更新）
//                pframe_l->addKeypoint(pt, pmap_l->nlmid_);//右目帧添加关键点
//                cv::Scalar col = imr.at<uchar>(pt.y, pt.x);
//                pmap_l->addMapPoint(col);//右目地图添加新地图点(无坐标信息)，并且nlmid_++：每次使用完，nlmid_都会+1，下次再使用时已经是+1之后的新值，再想访问当前点应该是nlmid_-1（前提是过程中没有别的地方再修改nlmid_）
//                pmap_r_to_l->addMapPoint(col);//只用于可视化
//                ///参考triangulateStereo——updateMapPoint，保存坐标
//                pmap_l->updateMapPoint(pmap_l->nlmid_ - 1, point_3d_r, true, 1. / point_in_camera.z());//存左目逆深度
//                pmap_r_to_l->updateMapPoint(pmap_r_to_l->nlmid_ - 1, point_3d_r, true, 1. / point_in_camera.z());//只用于可视化
//                ///参考Estimator::mapFiltering——removeMapPointObs，左目地图删除该点
//                pmap_r->removeMapPointObs(map_point_r->lmid_, pframe_r->id_);//SlamManager::run——pcurframe_l_->updateFrame(frame_id_, time);——id_= id;
//            }
//        } else {// 点在相机视野外，跳过
//            continue;
//        }
//    }

}


bool VisualFrontEnd::trackMonoStereo(cv::Mat &iml, cv::Mat &imr, cv::Mat &imlm, cv::Mat &imls, cv::Mat &imrm, cv::Mat &imrs, double time) {//mono_stereo 更新的trackMonoStereo相当于仍保留了之前算法的部分（用左目），只是添加了左右目各自的跟踪以及获取左右目各自位姿
    if( pslamstate_->debug_ )
        std::cout << "\n\n - [Visual-Front-End]: Track Mono-Stereo Image\n";

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.FE_Track-Mono-Stereo");

    // Preprocess the new image
    preprocessImage(iml, imr, imlm, imls, imrm, imrs);//mono_stereo

    // Create KF if 1st frame processed
    if( pcurframe_l_->id_ == 0 ) {//
        return true;
    }

    // Apply Motion model to predict cur Frame pose
    Sophus::SE3d Twc = pcurframe_l_->getTwc();//存的Twc都一样，用谁都一样
    Sophus::SE3d Twc_r = pcurframe_r_->getTwc();//存的Twc都一样，用谁都一样  用右目信息计算得到的左目位姿结果
    motion_model_.applyMotionModel(Twc, time);// 用的还是原本的pcurframe_的位姿
    motion_model_r_.applyMotionModel(Twc_r, time);// 用的还是原本的pcurframe_的位姿
    pcurframe_->setTwc(Twc);//
    pcurframe_l_->setTwc(Twc);//每个frame的Twc都保存左目的位姿 // todo mono_stereo
//    pcurframe_r_->setTwc(Twc);// todo stage2
    pcurframe_r_->setTwc(Twc_r);// todo stage2

    // Track the new image
    if( pslamstate_->btrack_keyframetoframe_ ) {
        kltTrackingFromKF();
    } else {
//        kltTracking();//官方配置文件用的基本都是这个 // todo mono_stereo
        kltTracking(pcurframe_l_, prev_pyrl_, cur_pyrl_, pmap_l_, true);//左目原图跟踪
        kltTracking(pcurframe_r_, prev_pyrr_, cur_pyrr_, pmap_r_, false);//右目原图跟踪
//        kltTracking(pcurframe_, prev_pyr_, cur_pyr_, pmap_, true);//和原算法效果一致
    }

    if( pslamstate_->doepipolar_ ) {
        // Check2d2dOutliers
//        epipolar2d2dFiltering(); // todo mono_stereo
        epipolar2d2dFiltering(pcurframe_l_, pmap_l_, true);//左目筛选
        epipolar2d2dFiltering(pcurframe_r_, pmap_r_, false);//右目筛选
//        epipolar2d2dFiltering(pcurframe_, pmap_, true);//和原算法效果一致
    }

    if( pslamstate_->mono_ && !pslamstate_->bvision_init_ )
    {
        if( pcurframe_->nb2dkps_ < 50 ) {
            pslamstate_->breset_req_ = true;
            return false;
        }
        else if( checkReadyForInit() ) {
            std::cout << "\n\n - [Visual-Front-End]: Mono Visual SLAM ready for initialization!";
            pslamstate_->bvision_init_ = true;
            return true;
        }
        else {
            std::cout << "\n\n - [Visual-Front-End]: Not ready to init yet!";
            return false;
        }
    }

    // Compute Pose (2D-3D)
//    computePose();//之前只用一张图跟踪。现在分别在两张图上进行跟踪，如果分开算则每个图都会算一个位姿。能不能合并成一个呢？在优化里合并成一个
    computePose(pcurframe_l_, pmap_l_, cur_pyrl_, cur_pyrr_, true);
    computePose(pcurframe_r_, pmap_r_, cur_pyrl_, cur_pyrr_, false);// todo stage2
//    computePose(pcurframe_, pmap_, true);//和源程序一样
    // 可能在这里分别额外求出左右目的新位姿，后面在优化函数里合并得到优化后的pcurframe_->Twc_。最终每一帧仍然只有一个位姿结果pcurframe_->Twc_

    // Update Motion model from estimated pose //  mono_stereo stage2 前面计算了三个位姿，这里更新是否要使用？
    motion_model_.updateMotionModel(pcurframe_l_->Twc_, time);//只用左目位姿
    motion_model_r_.updateMotionModel(pcurframe_r_->Twc_, time);//只用右目位姿
//    std::cout<<"Twc_l = "<<pcurframe_l_->Twc_.matrix()<<std::endl;
//    std::cout<<"Twc_r = "<<pcurframe_r_->Twc_.matrix()<<std::endl;

    // Check if New KF req.
//    bool is_kf_req = checkNewKfReq();
    bool is_kf_req_l = checkNewKfReq(pcurframe_l_, pmap_l_, true);
    bool is_kf_req_r = checkNewKfReq(pcurframe_r_, pmap_r_, false);
//    bool is_kf_req_diverg = checkNewKfReq_diverg(pcurframe_l_, pcurframe_r_, pmap_l_, pmap_r_);
//    bool is_kf_req = checkNewKfReq(pcurframe_, pmap_, true);//和源程序一样

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.FE_Track-Mono");

//    return is_kf_req;
//    std::cout<<"require??? = "<<(is_kf_req && is_kf_req_l && is_kf_req_r)<<std::endl;
    return (is_kf_req_l || is_kf_req_r);// todo mono_stereo 如果左右目有一个需要新的关键帧，则返回true 正常论文实验用的是这个，会导致关键帧比较多
//    return (is_kf_req_l && is_kf_req_r);// todo mono_stereo 如果左右目都需要新的关键帧，则返回true 这个关键帧少一些，但是效果不好
//    return is_kf_req_l;
}

bool VisualFrontEnd::trackMonoStereo(cv::Mat &iml, cv::Mat &imr, cv::Mat &imlm, cv::Mat &imls, cv::Mat &imrm, cv::Mat &imrs, double time, bool isleft) {//mono_stereo todo 2.0 更新的trackMonoStereo相当于仍保留了之前算法的部分（用左目），只是添加了左右目各自的跟踪以及获取左右目各自位姿
    if( pslamstate_->debug_ )
        std::cout << "\n\n - [Visual-Front-End]: Track Mono-Stereo Image\n";

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.FE_Track-Mono-Stereo");

    // Preprocess the new image
    preprocessImage(iml, imr, imlm, imls, imrm, imrs);//mono_stereo

    // Create KF if 1st frame processed
    if( pcurframe_->id_ == 0 ) {
        return true;
    }

    // Apply Motion model to predict cur Frame pose
    Sophus::SE3d Twc = pcurframe_->getTwc();
//    motion_model_.applyMotionModel(Twc, time);// 用的还是原本的pcurframe_的位姿
    pcurframe_->setTwc(Twc);
    pcurframe_l_->setTwc(Twc);//每个frame的Twc都保存左目的位姿 // todo mono_stereo
    pcurframe_r_->setTwc(Twc);

    // Track the new image
    if( pslamstate_->btrack_keyframetoframe_ ) {
        kltTrackingFromKF();
    } else {
//        kltTracking();//官方配置文件用的基本都是这个 // todo mono_stereo
        kltTracking(pcurframe_l_, prev_pyrl_, cur_pyrl_, pmap_l_, isleft);//
    }

    if( pslamstate_->doepipolar_ ) {
        // Check2d2dOutliers
//        epipolar2d2dFiltering(); // todo mono_stereo
        epipolar2d2dFiltering(pcurframe_l_, pmap_l_, isleft);//
    }

    if( pslamstate_->mono_ && !pslamstate_->bvision_init_ )
    {
        if( pcurframe_->nb2dkps_ < 50 ) {
            pslamstate_->breset_req_ = true;
            return false;
        }
        else if( checkReadyForInit() ) {
            std::cout << "\n\n - [Visual-Front-End]: Mono Visual SLAM ready for initialization!";
            pslamstate_->bvision_init_ = true;
            return true;
        }
        else {
            std::cout << "\n\n - [Visual-Front-End]: Not ready to init yet!";
            return false;
        }
    }

    // Compute Pose (2D-3D) // todo mono_stereo 在这之前，左右帧的位姿都和pcurframe_一样，这里分别更新
//    computePose();//之前只用一张图跟踪。现在分别在两张图上进行跟踪，如果分开算则每个图都会算一个位姿。能不能合并成一个呢？在优化里合并成一个
    computePose(pcurframe_l_, pmap_l_, cur_pyrl_, cur_pyrr_, isleft);//
    // 可能在这里分别额外求出左右目的新位姿，后面在优化函数里合并得到优化后的pcurframe_->Twc_。最终每一帧仍然只有一个位姿结果pcurframe_->Twc_

    // Update Motion model from estimated pose
    motion_model_.updateMotionModel(pcurframe_l_->Twc_, time);// todo mono_stereo 前面计算了三个位姿，这里更新是否要使用？

    // Check if New KF req.
//    bool is_kf_req = checkNewKfReq();
    bool is_kf_req;
    if(isleft){
        is_kf_req = checkNewKfReq(pcurframe_l_, pmap_l_, true);
    } else {
        is_kf_req = checkNewKfReq(pcurframe_r_, pmap_r_, false);
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.FE_Track-Mono");

    return is_kf_req;
//    std::cout<<"require??? = "<<(is_kf_req && is_kf_req_l && is_kf_req_r)<<std::endl;
//    return (is_kf_req && is_kf_req_l && is_kf_req_r);// todo mono_stereo 如果左右目都需要新的关键帧，则返回true
}

// KLT Tracking with motion prior
void VisualFrontEnd::kltTracking()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_KLT-Tracking");

    // Get current kps and init priors for tracking
    std::vector<int> v3dkpids, vkpids;
    std::vector<cv::Point2f> v3dkps, v3dpriors, vkps, vpriors;
    std::vector<bool> vkpis3d;

    // First we're gonna track 3d kps on only 2 levels
    v3dkpids.reserve(pcurframe_->nb3dkps_);
    v3dkps.reserve(pcurframe_->nb3dkps_);
    v3dpriors.reserve(pcurframe_->nb3dkps_);

    // Then we'll track 2d kps on full pyramid levels
    vkpids.reserve(pcurframe_->nbkps_);
    vkps.reserve(pcurframe_->nbkps_);
    vpriors.reserve(pcurframe_->nbkps_);

    vkpis3d.reserve(pcurframe_->nbkps_);


    // Front-End is thread-safe so we can direclty access curframe's kps
    for( const auto &it : pcurframe_->mapkps_ ) 
    {
        auto &kp = it.second;

        // Init prior px pos. from motion model
        if( pslamstate_->klt_use_prior_ )
        {
            if( kp.is3d_ ) 
            {
                cv::Point2f projpx = pcurframe_->projWorldToImageDist(pmap_->map_plms_.at(kp.lmid_)->getPoint());

                // Add prior if projected into image
                if( pcurframe_->isInImage(projpx) ) 
                {
                    v3dkps.push_back(kp.px_);
                    v3dpriors.push_back(projpx);
                    v3dkpids.push_back(kp.lmid_);

                    vkpis3d.push_back(true);
                    continue;
                }
            }
        }

        // For other kps init prior with prev px pos.
        vkpids.push_back(kp.lmid_);
        vkps.push_back(kp.px_);
        vpriors.push_back(kp.px_);
    }

    // 1st track 3d kps if using prior
    if( pslamstate_->klt_use_prior_ && !v3dpriors.empty() ) 
    {
        int nbpyrlvl = 1;

        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        auto vprior = v3dpriors;

        ptracker_->fbKltTracking(
                    prev_pyr_, 
                    cur_pyr_, 
                    pslamstate_->nklt_win_size_, 
                    nbpyrlvl, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    v3dkps, 
                    v3dpriors, 
                    vkpstatus);

        size_t nbgood = 0;
        size_t nbkps = v3dkps.size();

        for(size_t i = 0 ; i < nbkps  ; i++ ) 
        {
            if( vkpstatus.at(i) ) {
                pcurframe_->updateKeypoint(v3dkpids.at(i), v3dpriors.at(i));
                nbgood++;
            } else {
                // If tracking failed, gonna try on full pyramid size
                vkpids.push_back(v3dkpids.at(i));
                vkps.push_back(v3dkps.at(i));
                vpriors.push_back(v3dpriors.at(i));
            }
        }

        if( pslamstate_->debug_ ) {
            std::cout << "\n >>> KLT Tracking w. priors : " << nbgood;
            std::cout << " out of " << nbkps << " kps tracked!\n";
        }

        if( nbgood < 0.33 * nbkps ) {
            // Motion model might be quite wrong, P3P is recommended next
            // and not using any prior
            bp3preq_ = true;
            vpriors = vkps;
        }
    }

    // 2nd track other kps if any
    if( !vkps.empty() ) 
    {
        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        ptracker_->fbKltTracking(
                    prev_pyr_, 
                    cur_pyr_, 
                    pslamstate_->nklt_win_size_, 
                    pslamstate_->nklt_pyr_lvl_, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    vkps, 
                    vpriors, 
                    vkpstatus);
        
        size_t nbgood = 0;
        size_t nbkps = vkps.size();

        for(size_t i = 0 ; i < nbkps  ; i++ ) 
        {
            if( vkpstatus.at(i) ) {
                pcurframe_->updateKeypoint(vkpids.at(i), vpriors.at(i));
                nbgood++;
            } else {
                // MapManager is responsible for all the removing operations
                pmap_->removeObsFromCurFrameById(vkpids.at(i));
            }
        }

        if( pslamstate_->debug_ ) {
            std::cout << "\n >>> KLT Tracking no prior : " << nbgood;
            std::cout << " out of " << nbkps << " kps tracked!\n";
        }
    } 
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_KLT-Tracking");
}

// KLT Tracking with motion prior
void VisualFrontEnd::kltTracking(std::shared_ptr<Frame> &pframe, const std::vector<cv::Mat> &prevpyr, const std::vector<cv::Mat> &curpyr, std::shared_ptr<MapManager>& pmap, bool isleft)  //mono_stereo
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_KLT-Tracking");

    // Get current kps and init priors for tracking
    std::vector<int> v3dkpids, vkpids;
    std::vector<cv::Point2f> v3dkps, v3dpriors, vkps, vpriors;
    std::vector<bool> vkpis3d;

    // 保存跟踪前后点位置对，用于绘制光流
//    std::vector<std::pair<cv::Point2f, cv::Point2f>> optical_flow_lines;

    // First we're gonna track 3d kps on only 2 levels
//    std::cout << "pframe->nb3dkps_:"<<pframe->nb3dkps_<<std::endl;
    v3dkpids.reserve(pframe->nb3dkps_);
    v3dkps.reserve(pframe->nb3dkps_);
    v3dpriors.reserve(pframe->nb3dkps_);
//    if(isleft){
//        std::cout<<"left pframe->nb3dkps_ = "<<pframe->nb3dkps_<<std::endl;
//        std::cout<<"left pmap->nbkfs_ = "<<pmap->nbkfs_<<std::endl;
//    } else{
//        std::cout<<"right pframe->nb3dkps_ = "<<pframe->nb3dkps_<<std::endl;
//        std::cout<<"right pmap->nbkfs_ = "<<pmap->nbkfs_<<std::endl;
//    }

    // Then we'll track 2d kps on full pyramid levels
    //输出pframe->nbkps_大小
//    std::cout<<"pframe->nbkps_:"<<pframe->nbkps_<<std::endl;
    vkpids.reserve(pframe->nbkps_);
    vkps.reserve(pframe->nbkps_);
    vpriors.reserve(pframe->nbkps_);

    vkpis3d.reserve(pframe->nbkps_);

    //输出点数量
//    if(isleft){
//        std::cout<<"  ------- left pframe->mapkps_.size():"<<pframe->mapkps_.size()<<std::endl;
//    } else{
//        std::cout<<"  ------- right pframe->mapkps_.size():"<<pframe->mapkps_.size()<<std::endl;
//    }

    // Front-End is thread-safe so we can direclty access curframe's kps
    for( const auto &it : pframe->mapkps_ )
    {
        auto &kp = it.second;
        //防止重复范围建立地图点 一个点在一个时刻只应该归一个相机处理，如果两个相机同时跟踪一个点，那后面都会对这个点三角化，同一个位置就有了两个地图点
        // todotodo 数值不一样，不应该是一半
        if(isleft){
            if( kp.px_.x > pslamstate_->img_left_w_ - pslamstate_->img_w_crop_) {
//            if( kp.px_.x > pslamstate_->img_left_w_ - pslamstate_->img_lefts_w_/2.) {
                continue;
            }
        } else {
            if( kp.px_.x < pslamstate_->img_w_crop_) {
//            if( kp.px_.x < pslamstate_->img_lefts_w_/2.) {
                continue;
            }
        }

        // Init prior px pos. from motion model
        if( pslamstate_->klt_use_prior_ )
        {
            if( kp.is3d_ )
            {
                cv::Point2f projpx;

                // 检查索引是否存在
                auto map_it = pmap->map_plms_.find(kp.lmid_);
                if (map_it == pmap->map_plms_.end()) {
                    std::cerr << "Warning1: Keypoint with lmid " << kp.lmid_ << " not found in map_plms_.\n";
                    continue; // 跳过当前关键点
                }

                if(isleft){
                    projpx = pframe->projWorldToImageDist(pmap->map_plms_.at(kp.lmid_)->getPoint());//注意原工程projWorldToImageDist用的是左目
                } else {
                    projpx = pframe->projWorldToImageDist_right(pmap->map_plms_.at(kp.lmid_)->getPoint());//添加右目投影
                }

                // Add prior if projected into image
                if( pframe->isInImage(projpx) ) //注意原工程isInImage用的是左目，右目可以用一样的
                {
                    v3dkps.push_back(kp.px_);
                    v3dpriors.push_back(projpx);
                    v3dkpids.push_back(kp.lmid_);

                    vkpis3d.push_back(true);
                    continue;
                }
            }
        }

        // For other kps init prior with prev px pos.
        vkpids.push_back(kp.lmid_);
        vkps.push_back(kp.px_);
        vpriors.push_back(kp.px_);
    }

    //可视化特征点v3dkps  vkps
/*    cv::Mat draw_img1 = curpyr[0].clone(); // 获取当前图像的顶层
    for(auto &kp : v3dkps)
    {
        cv::circle(draw_img1, kp, 2, cv::Scalar(0, 255, 0), 2);
    }
    for(auto &kp : vkps)
    {
        cv::circle(draw_img1, kp, 2, cv::Scalar(0, 0, 255), 2);
    }
    cv::imshow("new_kps 1", draw_img1);
    cv::imwrite("/home/hl/project/ov2_diverg_ws/test/new_kps2.jpg", draw_img1);//没有天空点
    cv::waitKey(1);//没有天空点*/

    //用于可视化
    cv::Mat draw_img = curpyr[0].clone(); // 获取当前图像的顶层
    if (draw_img.channels() == 1)
        cv::cvtColor(draw_img, draw_img, cv::COLOR_GRAY2BGR);
    static int image_counter_l = 0; // 静态计数器，函数每次调用时会保留计数
    static int image_counter_r = 0; // 静态计数器，函数每次调用时会保留计数
    /// 1st track 3d kps if using prior
    if( pslamstate_->klt_use_prior_ && !v3dpriors.empty() )
    {
        int nbpyrlvl = 1;

        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        auto vprior = v3dpriors;

//        ptracker_->fbKltTracking(   // 光流法跟踪3d点 经测试，v3dpriors和vkpstatus更新了，v3dkps没更新
//                prevpyr,
//                curpyr,
//                pslamstate_->nklt_win_size_,
//                nbpyrlvl,
//                pslamstate_->nklt_err_,
//                pslamstate_->fmax_fbklt_dist_,
//                v3dkps,
//                v3dpriors,
//                vkpstatus);
        ptracker_->fbKltTracking2(prevpyr,curpyr,pslamstate_->nklt_win_size_,nbpyrlvl,pslamstate_->nklt_err_,pslamstate_->fmax_fbklt_dist_,
                v3dkps,v3dpriors,vkpstatus, isleft);

        size_t nbgood = 0;
        size_t nbkps = v3dkps.size();

        for(size_t i = 0 ; i < nbkps  ; i++ )
        {
            if( vkpstatus.at(i) ) {

                pframe->updateKeypoint(v3dkpids.at(i), v3dpriors.at(i));
//                optical_flow_lines.emplace_back(v3dkps[i], v3dpriors[i]); // 保存光流连线
                //可视化光流 有3d先验为红色
//            if (cv::norm(v3dkps[i] - v3dpriors[i]) < 1.0) continue; //如果移动小于 1 像素，跳过该点
                cv::line(draw_img, v3dpriors[i], v3dkps[i], cv::Scalar(0, 0, 255), 2); // 红色线条
                cv::circle(draw_img, v3dkps[i], 2, cv::Scalar(0, 0, 255), -1);      // 红色点
                nbgood++;
            } else {
                // If tracking failed, gonna try on full pyramid size
                vkpids.push_back(v3dkpids.at(i));
                vkps.push_back(v3dkps.at(i));
                vpriors.push_back(v3dpriors.at(i));
            }
        }

        if( pslamstate_->debug_ ) {
            std::cout << "\n >>> KLT Tracking w. priors : " << nbgood;
            std::cout << " out of " << nbkps << " kps tracked!\n";
        }

        if( nbgood < 0.33 * nbkps ) {
            // Motion model might be quite wrong, P3P is recommended next
            // and not using any prior
            if(isleft){
                bp3preq_ = true;
            } else {
                bp3preq_r_ = true;
            }

            vpriors = vkps;
        }
    }

    /// 2nd track other kps if any
    if( !vkps.empty() )
    {
        // Good / bad kps vector
        std::vector<bool> vkpstatus;

//        ptracker_->fbKltTracking(
//                prevpyr,
//                curpyr,
//                pslamstate_->nklt_win_size_,
//                pslamstate_->nklt_pyr_lvl_,
//                pslamstate_->nklt_err_,
//                pslamstate_->fmax_fbklt_dist_,
//                vkps,
//                vpriors,
//                vkpstatus);
        ptracker_->fbKltTracking2(prevpyr,curpyr,pslamstate_->nklt_win_size_,pslamstate_->nklt_pyr_lvl_,pslamstate_->nklt_err_,pslamstate_->fmax_fbklt_dist_,
                vkps,vpriors,vkpstatus, isleft);

        size_t nbgood = 0;
        size_t nbkps = vkps.size();

        for(size_t i = 0 ; i < nbkps  ; i++ )
        {
            if( vkpstatus.at(i) ) {
                pframe->updateKeypoint(vkpids.at(i), vpriors.at(i));
//                optical_flow_lines.emplace_back(vkps[i], vpriors[i]); // 保存光流连线
                //可视化光流 无3d先验为蓝色
//            if (cv::norm(vkps[i] - vpriors[i]) < 1.0) continue; //如果移动小于 1 像素，跳过该点
                cv::line(draw_img, vpriors[i], vkps[i], cv::Scalar(255, 0, 0), 2); // 蓝色线条
                cv::circle(draw_img, vkps[i], 2, cv::Scalar(255, 0, 0), -1);      // 蓝色点
                nbgood++;
            } else {
                // MapManager is responsible for all the removing operations
                pmap->removeObsFromCurFrameById(vkpids.at(i), pframe);//原函数用的是左目currentframe和，修改，传入pframe
            }
        }

        if( pslamstate_->debug_ ) {
            std::cout << "\n >>> KLT Tracking no prior : " << nbgood;
            std::cout << " out of " << nbkps << " kps tracked!\n";
        }
    }

    // 绘制光流连线
    if(isleft){
//        cv::imshow("Optical Flow Tracking left", draw_img);
        std::string filename = "/home/hl/project/ov2_diverg_ws/test2/optical_left/" + std::to_string(image_counter_l++) + ".png";
//        cv::imwrite(filename, draw_img);
//        cv::waitKey(1);
    } else {
//        cv::imshow("Optical Flow Tracking right", draw_img);
        std::string filename = "/home/hl/project/ov2_diverg_ws/test2/optical_right/" + std::to_string(image_counter_r++) + ".png";
//        cv::imwrite(filename, draw_img);
//        cv::waitKey(1);
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_KLT-Tracking");
}


void VisualFrontEnd::kltTrackingFromKF()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_KLT-Tracking-from-KF");

    // Get current kps and init priors for tracking
    std::vector<int> v3dkpids, vkpids;
    std::vector<cv::Point2f> v3dkps, v3dpriors, vkps, vpriors;
    std::vector<bool> vkpis3d;

    // First we're gonna track 3d kps on only 2 levels
    v3dkpids.reserve(pcurframe_->nb3dkps_);
    v3dkps.reserve(pcurframe_->nb3dkps_);
    v3dpriors.reserve(pcurframe_->nb3dkps_);

    // Then we'll track 2d kps on full pyramid levels
    vkpids.reserve(pcurframe_->nbkps_);
    vkps.reserve(pcurframe_->nbkps_);
    vpriors.reserve(pcurframe_->nbkps_);

    vkpis3d.reserve(pcurframe_->nbkps_);

    // Get prev KF
    auto pkf = pmap_->map_pkfs_.at(pcurframe_->kfid_);

    if( pkf == nullptr ) {
        return;
    }

    std::vector<int> vbadids;
    vbadids.reserve(pcurframe_->nbkps_ * 0.2);


    // Front-End is thread-safe so we can direclty access curframe's kps
    for( const auto &it : pcurframe_->mapkps_ ) 
    {
        auto &kp = it.second;

        auto kfkpit = pkf->mapkps_.find(kp.lmid_);
        if( kfkpit == pkf->mapkps_.end() ) {
            vbadids.push_back(kp.lmid_);
            continue;
        }

        // Init prior px pos. from motion model
        if( pslamstate_->klt_use_prior_ )
        {
            if( kp.is3d_ ) 
            {
                cv::Point2f projpx = pcurframe_->projWorldToImageDist(pmap_->map_plms_.at(kp.lmid_)->getPoint());

                // Add prior if projected into image
                if( pcurframe_->isInImage(projpx) ) 
                {
                    v3dkps.push_back(kfkpit->second.px_);
                    v3dpriors.push_back(projpx);
                    v3dkpids.push_back(kp.lmid_);

                    vkpis3d.push_back(true);
                    continue;
                }
            }
        }

        // For other kps init prior with prev px pos.
        vkpids.push_back(kp.lmid_);
        vkps.push_back(kfkpit->second.px_);
        vpriors.push_back(kp.px_);
    }

    for( const auto &badid : vbadids ) {
        // MapManager is responsible for all the removing operations
        pmap_->removeObsFromCurFrameById(badid);
    }

    // 1st track 3d kps if using prior
    if( pslamstate_->klt_use_prior_ && !v3dpriors.empty() ) 
    {
        int nbpyrlvl = 1;

        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        auto vprior = v3dpriors;

        ptracker_->fbKltTracking(
                    kf_pyr_, 
                    cur_pyr_, 
                    pslamstate_->nklt_win_size_, 
                    nbpyrlvl, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    v3dkps, 
                    v3dpriors, 
                    vkpstatus);

        size_t nbgood = 0;
        size_t nbkps = v3dkps.size();

        for(size_t i = 0 ; i < nbkps  ; i++ ) 
        {
            if( vkpstatus.at(i) ) {
                pcurframe_->updateKeypoint(v3dkpids.at(i), v3dpriors.at(i));
                nbgood++;
            } else {
                // If tracking failed, gonna try on full pyramid size
                vkpids.push_back(v3dkpids.at(i));
                vkps.push_back(v3dkps.at(i));
                vpriors.push_back(pcurframe_->mapkps_.at(v3dkpids.at(i)).px_);
            }
        }

        if( pslamstate_->debug_ ) {
            std::cout << "\n >>> KLT Tracking w. priors : " << nbgood;
            std::cout << " out of " << nbkps << " kps tracked!\n";
        }

        if( nbgood < 0.33 * nbkps ) {
            // Motion model might be quite wrong, P3P is recommended next
            // and not using any prior
            bp3preq_ = true;
            vpriors = vkps;
        }
    }

    // 2nd track other kps if any
    if( !vkps.empty() ) 
    {
        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        ptracker_->fbKltTracking(
                    kf_pyr_, 
                    cur_pyr_, 
                    pslamstate_->nklt_win_size_, 
                    pslamstate_->nklt_pyr_lvl_, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    vkps, 
                    vpriors, 
                    vkpstatus);
        
        size_t nbgood = 0;
        size_t nbkps = vkps.size();

        for(size_t i = 0 ; i < nbkps  ; i++ ) 
        {
            if( vkpstatus.at(i) ) {
                pcurframe_->updateKeypoint(vkpids.at(i), vpriors.at(i));
                nbgood++;
            } else {
                // MapManager is responsible for all the removing operations
                pmap_->removeObsFromCurFrameById(vkpids.at(i));
            }
        }

        if( pslamstate_->debug_ ) {
            std::cout << "\n >>> KLT Tracking no prior : " << nbgood;
            std::cout << " out of " << nbkps << " kps tracked!\n";
        }
    } 
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_KLT-Tracking");
}


// This function apply a 2d-2d based outliers filtering
void VisualFrontEnd::epipolar2d2dFiltering()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_EpipolarFiltering");
    
    // Get prev. KF (direct access as Front-End is thread safe)
    auto pkf = pmap_->map_pkfs_.at(pcurframe_->kfid_);

    if( pkf == nullptr ) {
        std::cerr << "\nERROR! Previous Kf does not exist yet (epipolar2d2d()).\n";
        exit(-1);
    }

    // Get cur. Frame nb kps
    size_t nbkps = pcurframe_->nbkps_;

    if( nbkps < 8 ) {
        if( pslamstate_->debug_ )
            std::cout << "\nNot enough kps to compute Essential Matrix\n";
        return;
    }

    // Setup Essential Matrix computation for OpenGV-based filtering
    std::vector<int> vkpsids, voutliersidx;
    vkpsids.reserve(nbkps);
    voutliersidx.reserve(nbkps);

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vkfbvs, vcurbvs;
    vkfbvs.reserve(nbkps);
    vcurbvs.reserve(nbkps);
    
    size_t nbparallax = 0;
    float avg_parallax = 0.;

    // In stereo mode, we consider 3d kps as better tracks and therefore
    // use only them for computing E with RANSAC, 2d kps are then removed based
    // on the resulting Fundamental Mat.
    bool epifrom3dkps = false;
    if( pslamstate_->stereo_ && pcurframe_->nb3dkps_ > 30 ) {
        epifrom3dkps = true;
    }

    // Compute rotation compensated parallax
    Eigen::Matrix3d Rkfcur = pkf->getRcw() * pcurframe_->getRwc();

    // Init bearing vectors and check parallax
    for( const auto &it : pcurframe_->mapkps_ ) {

        if( epifrom3dkps ) {
            if( !it.second.is3d_ ) {
                continue;
            }
        }

        auto &kp = it.second;

        // Get the prev. KF related kp if it exists
        auto kfkp = pkf->getKeypointById(kp.lmid_);

        if( kfkp.lmid_ != kp.lmid_ ) {
            continue;
        }

        // Store the bvs and their ids
        vkfbvs.push_back(kfkp.bv_);
        vcurbvs.push_back(kp.bv_);
        vkpsids.push_back(kp.lmid_);

        cv::Point2f rotpx = pkf->projCamToImage(Rkfcur * kp.bv_);

        // Compute parallax
        avg_parallax += cv::norm(rotpx - kfkp.unpx_);
        nbparallax++;
    }

    if( nbkps < 8 ) {
        if( pslamstate_->debug_ )
            std::cout << "\nNot enough kps to compute Essential Matrix\n";
        return;
    }

    // Average parallax
    avg_parallax /= nbparallax;

    if( avg_parallax < 2. * pslamstate_->fransac_err_ ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> Not enough parallax (" << avg_parallax 
                << " px) to compute 5-pt Essential Matrix\n";
        return;
    }

    bool do_optimize = false;

    // In monocular case, we'll use the resulting motion if tracking is poor
    if( pslamstate_->mono_ && pmap_->nbkfs_ > 2 
        && pcurframe_->nb3dkps_ < 30 ) 
    {
        do_optimize = true;
    }

    Eigen::Matrix3d Rkfc;
    Eigen::Vector3d tkfc;

    if( pslamstate_->debug_ ) {
        std::cout << "\n \t>>> 5-pt EssentialMatrix Ransac :";
        std::cout << "\n \t>>> only on 3d kps : " << epifrom3dkps;
        std::cout << "\n \t>>> nb pts : " << nbkps;
        std::cout << " / avg. parallax : " << avg_parallax;
        std::cout << " / nransac_iter_ : " << pslamstate_->nransac_iter_;
        std::cout << " / fransac_err_ : " << pslamstate_->fransac_err_;
        std::cout << "\n\n";
    }
    
    bool success = 
        MultiViewGeometry::compute5ptEssentialMatrix(
                    vkfbvs, vcurbvs, 
                    pslamstate_->nransac_iter_, 
                    pslamstate_->fransac_err_, 
                    do_optimize, 
                    pslamstate_->bdo_random, 
                    pcurframe_->pcalib_leftcam_->fx_, 
                    pcurframe_->pcalib_leftcam_->fy_, 
                    Rkfc, tkfc, 
                    voutliersidx);

    if( pslamstate_->debug_ )
        std::cout << "\n \t>>> Epipolar nb outliers : " << voutliersidx.size();

    if( !success) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> No pose could be computed from 5-pt EssentialMatrix\n";
        return;
    }

    if( voutliersidx.size() > 0.5 * vkfbvs.size() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> Too many outliers, skipping as might be degenerate case\n";
        return;
    }

    // Remove outliers
    for( const auto & idx : voutliersidx ) {
        // MapManager is responsible for all the removing operations.
        pmap_->removeObsFromCurFrameById(vkpsids.at(idx));
    }

    // In case we wanted to use the resulting motion 
    // (mono mode - can help when tracking is poor)
    if( do_optimize && pmap_->nbkfs_ > 2 ) 
    {
        // Get motion model translation scale from last KF
        Sophus::SE3d Tkfw = pkf->getTcw();
        Sophus::SE3d Tkfcur = Tkfw * pcurframe_->getTwc();

        double scale = Tkfcur.translation().norm();
        tkfc.normalize();

        // Update current pose with Essential Mat. relative motion
        // and current trans. scale
        Sophus::SE3d Tkfc(Rkfc, scale * tkfc);

        pcurframe_->setTwc(pkf->getTwc() * Tkfc);
    }

    // In case we only used 3d kps for computing E (stereo mode)
    if( epifrom3dkps ) {

        if( pslamstate_->debug_ )
            std::cout << "\n Applying found Essential Mat to 2D kps!\n";

        Sophus::SE3d Tidentity;
        Sophus::SE3d Tkfcur(Rkfc, tkfc);

        Eigen::Matrix3d Fkfcur = MultiViewGeometry::computeFundamentalMat12(Tidentity, Tkfcur, pcurframe_->pcalib_leftcam_->K_);

        std::vector<int> vbadkpids;
        vbadkpids.reserve(pcurframe_->nb2dkps_);

        for( const auto &it : pcurframe_->mapkps_ ) 
        {
            if( it.second.is3d_ ) {
                continue;
            }

            auto &kp = it.second;

            // Get the prev. KF related kp if it exists
            auto kfkp = pkf->getKeypointById(kp.lmid_);

            // Normalized coord.
            Eigen::Vector3d curpt(kp.unpx_.x, kp.unpx_.y, 1.);
            Eigen::Vector3d kfpt(kfkp.unpx_.x, kfkp.unpx_.y, 1.);

            float epi_err = MultiViewGeometry::computeSampsonDistance(Fkfcur, curpt, kfpt);

            if( epi_err > pslamstate_->fransac_err_ ) {
                vbadkpids.push_back(kp.lmid_);
            }
        }

        for( const auto & kpid : vbadkpids ) {
            pmap_->removeObsFromCurFrameById(kpid);
        }

        if( pslamstate_->debug_ )
            std::cout << "\n Nb of 2d kps removed : " << vbadkpids.size() << " \n";
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_EpipolarFiltering");
}

void VisualFrontEnd::epipolar2d2dFiltering(std::shared_ptr<Frame>& pframe, std::shared_ptr<MapManager>& pmap, bool isleft)//mono_stereo
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_EpipolarFiltering");

    // Get prev. KF (direct access as Front-End is thread safe)
//    auto pkf = pmap->map_pkfs_.at(pframe->kfid_);//原代码
    // todo 防止越界
    std::shared_ptr<Frame> pkf;
    auto it = pmap->map_pkfs_.find(pframe->kfid_);
    if (it != pmap->map_pkfs_.end()) {
        // 如果找到 key，则获取 pkf
        pkf = it->second;
        // 后续使用 pkf
    } else {
        // 如果未找到 key，则输出错误信息并跳过程序
        std::cerr << "Error: Key " << pframe->kfid_ << " not found in map_pkfs_!" << std::endl;
        return; // 或者 continue，根据你的程序逻辑选择跳过的方式
    }


    if( pkf == nullptr ) {
        std::cerr << "\nERROR! Previous Kf does not exist yet (epipolar2d2d()).\n";
        exit(-1);
    }

    // Get cur. Frame nb kps
    size_t nbkps = pframe->nbkps_;

    if( nbkps < 8 ) {
        if( pslamstate_->debug_ )
            std::cout << "\nNot enough kps to compute Essential Matrix\n";
        return;
    }

    // Setup Essential Matrix computation for OpenGV-based filtering
    std::vector<int> vkpsids, voutliersidx;
    vkpsids.reserve(nbkps);
    voutliersidx.reserve(nbkps);

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vkfbvs, vcurbvs;
    vkfbvs.reserve(nbkps);
    vcurbvs.reserve(nbkps);

    size_t nbparallax = 0;
    float avg_parallax = 0.;

    // In stereo mode, we consider 3d kps as better tracks and therefore
    // use only them for computing E with RANSAC, 2d kps are then removed based
    // on the resulting Fundamental Mat.
    bool epifrom3dkps = false;
    if( (pslamstate_->stereo_ || pslamstate_->mono_stereo_) && pframe->nb3dkps_ > 30 ) {
        epifrom3dkps = true;
    }

    // Compute rotation compensated parallax
    Eigen::Matrix3d Rkfcur;
    if(isleft){
        Rkfcur = pkf->getRcw() * pframe->getRwc();// 左当前帧到左关键帧=世界到左关键帧*左当前帧到世界
    } else {
        Eigen::Matrix3d R_right_left = pslamstate_->T_right_left_.rotationMatrix();
        Eigen::Matrix3d R_left_right = pslamstate_->T_left_right_.rotationMatrix();
        Rkfcur = R_right_left * pkf->getRcw() * pframe->getRwc() * R_left_right;  // 右当前帧到右关键帧=世界到左关键帧*左当前帧到世界
    }

    // Init bearing vectors and check parallax
    for( const auto &it : pframe->mapkps_ ) {

        if( epifrom3dkps ) {
            if( !it.second.is3d_ ) {
                continue;
            }
        }

        auto &kp = it.second;

        // Get the prev. KF related kp if it exists
        auto kfkp = pkf->getKeypointById(kp.lmid_);

        if( kfkp.lmid_ != kp.lmid_ ) {
            continue;
        }

        // Store the bvs and their ids
        vkfbvs.push_back(kfkp.bv_);
        vcurbvs.push_back(kp.bv_);
        vkpsids.push_back(kp.lmid_);

        cv::Point2f rotpx = pkf->projCamToImage(Rkfcur * kp.bv_);// 原算法用左目，现在Rkfcur是左目或右目，projCamToImage用左目模型但是左右目模型参数一样，因此不用修改

        // Compute parallax
        avg_parallax += cv::norm(rotpx - kfkp.unpx_);// 这里用的怎么是unpx_?Frame::computeKeypoint中使用的左右目相机模型畸变参数一致所以应该没问题
        nbparallax++;
    }

    if( nbkps < 8 ) {
        if( pslamstate_->debug_ )
            std::cout << "\nNot enough kps to compute Essential Matrix\n";
        return;
    }

    // Average parallax
    avg_parallax /= nbparallax;

    if( avg_parallax < 2. * pslamstate_->fransac_err_ ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> Not enough parallax (" << avg_parallax
                      << " px) to compute 5-pt Essential Matrix\n";
        return;
    }

    bool do_optimize = false;

    // In monocular case, we'll use the resulting motion if tracking is poor
    if( pslamstate_->mono_ && pmap->nbkfs_ > 2
        && pframe->nb3dkps_ < 30 )
    {
        do_optimize = true;
    }

    Eigen::Matrix3d Rkfc;
    Eigen::Vector3d tkfc;

    if( pslamstate_->debug_ ) {
        std::cout << "\n \t>>> 5-pt EssentialMatrix Ransac :";
        std::cout << "\n \t>>> only on 3d kps : " << epifrom3dkps;
        std::cout << "\n \t>>> nb pts : " << nbkps;
        std::cout << " / avg. parallax : " << avg_parallax;
        std::cout << " / nransac_iter_ : " << pslamstate_->nransac_iter_;
        std::cout << " / fransac_err_ : " << pslamstate_->fransac_err_;
        std::cout << "\n\n";
    }

    bool success =
            MultiViewGeometry::compute5ptEssentialMatrix(
                    vkfbvs, vcurbvs,
                    pslamstate_->nransac_iter_,
                    pslamstate_->fransac_err_,
                    do_optimize,
                    pslamstate_->bdo_random,
                    pcurframe_->pcalib_leftcam_->fx_,
                    pcurframe_->pcalib_leftcam_->fy_,//每个frame都有左目相机参数吗？不对吧.就用原本的pcurframe_吧，反正数值一样
                    Rkfc, tkfc,//用的是左目或右目的点，因此得到的结果是左目或右目的
                    voutliersidx);

    if( pslamstate_->debug_ )
        std::cout << "\n \t>>> Epipolar nb outliers : " << voutliersidx.size();

    if( !success) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> No pose could be computed from 5-pt EssentialMatrix\n";
        return;
    }

    if( voutliersidx.size() > 0.5 * vkfbvs.size() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> Too many outliers, skipping as might be degenerate case\n";
        return;
    }

    // Remove outliers
    for( const auto & idx : voutliersidx ) {
        // MapManager is responsible for all the removing operations.
//        pmap->removeObsFromCurFrameById(vkpsids.at(idx));//原函数用的是左目currentframe和，修改，传入pframe
        pmap->removeObsFromCurFrameById(vkpsids.at(idx), pframe);
    }

    // In case we wanted to use the resulting motion
    // (mono mode - can help when tracking is poor)
    if( do_optimize && pmap->nbkfs_ > 2 )   // todo 这里根本就不用改，只有单目的时候do_optimize=true才有用
    {
//        if(isleft){
            // Get motion model translation scale from last KF
            Sophus::SE3d Tkfw = pkf->getTcw();//左目Tcw  world到左目
            Sophus::SE3d Tkfcur = Tkfw * pframe->getTwc();//不管哪个相机的frame的Twc都是左目的，因此算出来的也是左目  左当前帧到左关键帧=世界到左关键帧*左当前帧到世界

            double scale = Tkfcur.translation().norm();//左目尺度
            tkfc.normalize();//左目或右目的

            // Update current pose with Essential Mat. relative motion
            // and current trans. scale
            Sophus::SE3d Tkfc(Rkfc, scale * tkfc);//左目或右目的 左当前帧到左关键帧

            pframe->setTwc(pkf->getTwc() * Tkfc);// 左关键帧到世界*左当前帧到左关键帧=左当前帧到世界
            pcurframe_->setTwc(pkf->getTwc() * Tkfc);// 左目右目的frame内，点、地图可以各自求各自的，但是位姿应该都赋值
//        } else {
//            // Get motion model translation scale from last KF
//            Sophus::SE3d Tkfw = pkf->getTcw();//左目Tcw  world到左目
//            Sophus::SE3d Tkfcur = pslamstate_->T_right_left_ * Tkfw * pframe->getTwc() * pslamstate_->T_left_right_ ;// 右当前帧到右关键帧=T左到右*世界到左关键帧*左当前帧到世界*T右到左
//
//            double scale = Tkfcur.translation().norm();//右目尺度
//
//            tkfc.normalize();//左目或右目的
//
//            // Update current pose with Essential Mat. relative motion
//            // and current trans. scale
//            Sophus::SE3d Tkfc(Rkfc, scale * tkfc);//左目或右目的 右当前帧到右关键帧
//
//            pframe->setTwc(pkf->getTwc() * Tkfc * pslamstate_->T_left_right_);//Twc也要保存左目结果。左关键帧到世界*左当前帧到左关键帧*T右到左=右当前帧到世界
//        }

    }

    // In case we only used 3d kps for computing E (stereo mode)
    if( epifrom3dkps ) {

        if( pslamstate_->debug_ )
            std::cout << "\n Applying found Essential Mat to 2D kps!\n";

        Sophus::SE3d Tidentity;
        Sophus::SE3d Tkfcur(Rkfc, tkfc);//左目或右目的

        Eigen::Matrix3d Fkfcur = MultiViewGeometry::computeFundamentalMat12(Tidentity, Tkfcur, pframe->pcalib_leftcam_->K_);//应该不用修改

        std::vector<int> vbadkpids;
        vbadkpids.reserve(pframe->nb2dkps_);

        for( const auto &it : pframe->mapkps_ )
        {
            if( it.second.is3d_ ) {
                continue;
            }

            auto &kp = it.second;

            // Get the prev. KF related kp if it exists
            auto kfkp = pkf->getKeypointById(kp.lmid_);

            // Normalized coord.
            Eigen::Vector3d curpt(kp.unpx_.x, kp.unpx_.y, 1.);// 用的是unpx_
            Eigen::Vector3d kfpt(kfkp.unpx_.x, kfkp.unpx_.y, 1.);

            float epi_err = MultiViewGeometry::computeSampsonDistance(Fkfcur, curpt, kfpt);

            if( epi_err > pslamstate_->fransac_err_ ) {
                vbadkpids.push_back(kp.lmid_);
            }
        }

        for( const auto & kpid : vbadkpids ) {
            pmap->removeObsFromCurFrameById(kpid, pframe);
        }

        if( pslamstate_->debug_ )
            std::cout << "\n Nb of 2d kps removed : " << vbadkpids.size() << " \n";
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_EpipolarFiltering");
}


void VisualFrontEnd::computePose()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_computePose");

    // Get cur nb of 3D kps    
    size_t nb3dkps = pcurframe_->nb3dkps_;

    if( nb3dkps < 4 ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> Not enough kps to compute P3P / PnP\n";
        return;
    }

    // Setup P3P-Ransac computation for OpenGV-based Pose estimation
    // + motion-only BA with Ceres
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vbvs, vwpts;
    std::vector<int> vkpids, voutliersidx, vscales;

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > vkps;

    vbvs.reserve(nb3dkps);
    vwpts.reserve(nb3dkps);
    vkpids.reserve(nb3dkps);
    voutliersidx.reserve(nb3dkps);

    vkps.reserve(nb3dkps);
    vscales.reserve(nb3dkps);

    bool bdop3p = bp3preq_ || pslamstate_->dop3p_;

    // Store every 3D bvs, MPs and their related ids
    for( const auto &it : pcurframe_->mapkps_ ) 
    {
        if( !it.second.is3d_ ) {
            continue;
        }

        auto &kp = it.second;
        // auto plm = pmap_->getMapPoint(kp.lmid_);
        auto plm = pmap_->map_plms_.at(kp.lmid_);
        if( plm == nullptr ) {
            continue;
        }

        if( bdop3p ) {
            vbvs.push_back(kp.bv_);
        }

        vkps.push_back(Eigen::Vector2d(kp.unpx_.x, kp.unpx_.y));
        vwpts.push_back(plm->getPoint());
        vscales.push_back(kp.scale_);
        vkpids.push_back(kp.lmid_);
    }

    Sophus::SE3d Twc = pcurframe_->getTwc();
    bool do_optimize = false;
    bool success = false;

    if( bdop3p ) 
    {
        if( pslamstate_->debug_ ) {
            std::cout << "\n \t>>>P3P Ransac : ";
            std::cout << "\n \t>>> nb 3d pts : " << nb3dkps;
            std::cout << " / nransac_iter_ : " << pslamstate_->nransac_iter_;
            std::cout << " / fransac_err_ : " << pslamstate_->fransac_err_;
            std::cout << "\n\n";
        }

        // Only effective with OpenGV
        bool use_lmeds = true;

        success = 
            MultiViewGeometry::p3pRansac(
                            vbvs, vwpts, 
                            pslamstate_->nransac_iter_, 
                            pslamstate_->fransac_err_, 
                            do_optimize, 
                            pslamstate_->bdo_random, 
                            pcurframe_->pcalib_leftcam_->fx_, 
                            pcurframe_->pcalib_leftcam_->fy_, 
                            Twc,
                            voutliersidx,
                            use_lmeds);

        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> P3P-LMeds nb outliers : " << voutliersidx.size();

        // Check that pose estim. was good enough
        size_t nbinliers = vwpts.size() - voutliersidx.size();

        if( !success
            || nbinliers < 5
            || Twc.translation().array().isInf().any()
            || Twc.translation().array().isNaN().any() )
        {
            if( pslamstate_->debug_ )
                std::cout << "\n \t>>> Not enough inliers for reliable pose est. Resetting KF state\n";

            resetFrame();

            return;
        } 

        // Pose seems to be OK!

        // Update frame pose
        pcurframe_->setTwc(Twc);

        // Remove outliers before PnP refinement (a bit dirty)
        int k = 0;
        for( const auto &idx : voutliersidx ) {
            // MapManager is responsible for all removing operations
            pmap_->removeObsFromCurFrameById(vkpids.at(idx-k));
            vkps.erase(vkps.begin() + idx - k);
            vwpts.erase(vwpts.begin() + idx - k);
            vkpids.erase(vkpids.begin() + idx - k);
            vscales.erase(vscales.begin() + idx - k);
            k++;
        }

        // Clear before robust PnP refinement using Ceres
        voutliersidx.clear();
    }

    // Ceres-based PnP (motion-only BA)
    bool buse_robust = true;
    bool bapply_l2_after_robust = pslamstate_->apply_l2_after_robust_;
    
    size_t nbmaxiters = 5;

    success =
        MultiViewGeometry::ceresPnP(
                        vkps, vwpts, 
                        vscales,
                        Twc, 
                        nbmaxiters, 
                        pslamstate_->robust_mono_th_, 
                        buse_robust, 
                        bapply_l2_after_robust,
                        pcurframe_->pcalib_leftcam_->fx_, pcurframe_->pcalib_leftcam_->fy_,
                        pcurframe_->pcalib_leftcam_->cx_, pcurframe_->pcalib_leftcam_->cy_,
                        voutliersidx);
    
    // Check that pose estim. was good enough
    size_t nbinliers = vwpts.size() - voutliersidx.size();

    if( pslamstate_->debug_ )
        std::cout << "\n \t>>> Ceres PnP nb outliers : " << voutliersidx.size();

    if( !success
        || nbinliers < 5
        || voutliersidx.size() > 0.5 * vwpts.size()
        || Twc.translation().array().isInf().any()
        || Twc.translation().array().isNaN().any() )
    {
        if( !bdop3p ) {
            // Weird results, skipping here and applying p3p next
            bp3preq_ = true;
        }
        else if( pslamstate_->mono_ ) {

            if( pslamstate_->debug_ )
                std::cout << "\n \t>>> Not enough inliers for reliable pose est. Resetting KF state\n";

            resetFrame();
        } 
        // else {
            // resetFrame();
            // motion_model_.reset();
        // }

        return;
    } 

    // Pose seems to be OK!

    // Update frame pose
    pcurframe_->setTwc(Twc);

    // Set p3p req to false as it is triggered either because
    // of bad PnP or by bad klt tracking
    bp3preq_ = false;

    // Remove outliers
    for( const auto & idx : voutliersidx ) {
        // MapManager is responsible for all removing operations
        pmap_->removeObsFromCurFrameById(vkpids.at(idx));
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_computePose");
}

void VisualFrontEnd::computePose(std::shared_ptr<Frame>& pframe, std::shared_ptr<MapManager>& pmap, const std::vector<cv::Mat> &curpyl, const std::vector<cv::Mat> &curpyr, bool isleft)
{
//    std::cout<<"--------------------------------------------"<<std::endl;
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_computePose");

    // Get cur nb of 3D kps
    size_t nb3dkps = pframe->nb3dkps_;  // 0
//    std::cout << "nb3dkps: " << nb3dkps << std::endl;

    if( nb3dkps < 4 ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> Not enough kps to compute P3P / PnP\n";
        return;
    }

    // Setup P3P-Ransac computation for OpenGV-based Pose estimation
    // + motion-only BA with Ceres
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vbvs, vwpts;
    std::vector<int> vkpids, voutliersidx, vscales;

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > vkps;

    vbvs.reserve(nb3dkps);
    vwpts.reserve(nb3dkps);
    vkpids.reserve(nb3dkps);
    voutliersidx.reserve(nb3dkps);

    vkps.reserve(nb3dkps);
    vscales.reserve(nb3dkps);

    bool bdop3p;
    if(isleft){
        bdop3p = bp3preq_ || pslamstate_->dop3p_;
    } else {
        bdop3p = bp3preq_r_ || pslamstate_->dop3p_;
    }

    // Store every 3D bvs, MPs and their related ids
    for( const auto &it : pframe->mapkps_ )
    {
        if( !it.second.is3d_ ) {
            continue;
        }

        auto &kp = it.second;

        // 检查索引是否存在
        auto map_it = pmap->map_plms_.find(kp.lmid_);
        if (map_it == pmap->map_plms_.end()) {
            std::cerr << "Warning2: Keypoint with lmid " << kp.lmid_ << " not found in map_plms_.\n";
            continue; // 跳过当前关键点
        }

        // auto plm = pmap_->getMapPoint(kp.lmid_);
        auto plm = pmap->map_plms_.at(kp.lmid_);
        if( plm == nullptr ) {
            continue;
        }

        if( bdop3p ) {
                vbvs.push_back(kp.bv_);
        }

        vkps.push_back(Eigen::Vector2d(kp.unpx_.x, kp.unpx_.y));

        vwpts.push_back(plm->getPoint());
//        if(isleft){
//            std::cout<<"left plm->getPoint(): "<<plm->getPoint().transpose()<<std::endl;
//        } else {
//            std::cout<<"right plm->getPoint(): "<<plm->getPoint().transpose()<<std::endl;
//        }

        vscales.push_back(kp.scale_);
        vkpids.push_back(kp.lmid_);
    }

    //可视化测试点对不对
    cv::Mat draw_img_l = curpyl[0].clone();
    cv::Mat draw_img_r = curpyr[0].clone();
    if (draw_img_l.channels() == 1)
        cv::cvtColor(draw_img_l, draw_img_l, cv::COLOR_GRAY2BGR);
    if (draw_img_r.channels() == 1)
        cv::cvtColor(draw_img_r, draw_img_r, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < vkps.size(); i++) {
        if(isleft){
            cv::circle(draw_img_l, cv::Point2f(vkps[i](0), vkps[i](1)), 2, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::circle(draw_img_r, cv::Point2f(vkps[i](0), vkps[i](1)), 2, cv::Scalar(0, 255, 0), 2);
        }
    }
//    if(isleft){
//        cv::imshow("draw_img_l", draw_img_l);
//        cv::imwrite("/home/hl/project/ov2_diverg_ws/test/draw_img_l.png", draw_img_l);
//        cv::waitKey(1);
//    } else {
//        cv::imshow("draw_img_r", draw_img_r);
//        cv::imwrite("/home/hl/project/ov2_diverg_ws/test/draw_img_r.png", draw_img_r);
//        cv::waitKey(1);
//    }

    Sophus::SE3d Twc;
    if(isleft){
        Twc = pframe->getTwc();// 左目到世界*左目点=左目到世界*左目点
//        std::cout << "left Twc 0000: " << Twc.matrix() << std::endl;
    } else {
        Twc = pframe->getTwc() * pslamstate_->T_left_right_;//右目到世界*右目点=左目到世界*右目到左目*右目点  这个Twc存的是右目位姿，用于右目独立求解位姿，后面再转换为左目位姿存入frame
//        std::cout << "right Twc 0000: " << Twc.matrix() << std::endl;
    }
    bool do_optimize = false;
    bool success = false;

    if( bdop3p )
    {
        if( pslamstate_->debug_ ) {
            std::cout << "\n \t>>>P3P Ransac : ";
            std::cout << "\n \t>>> nb 3d pts : " << nb3dkps;
            std::cout << " / nransac_iter_ : " << pslamstate_->nransac_iter_;
            std::cout << " / fransac_err_ : " << pslamstate_->fransac_err_;
            std::cout << "\n\n";
        }

        // Only effective with OpenGV
        bool use_lmeds = true;

        success =
                MultiViewGeometry::p3pRansac(
                        vbvs, vwpts,
                        pslamstate_->nransac_iter_,
                        pslamstate_->fransac_err_,
                        do_optimize,
                        pslamstate_->bdo_random,
                        pcurframe_->pcalib_leftcam_->fx_,
                        pcurframe_->pcalib_leftcam_->fy_,
                        Twc,
                        voutliersidx,
                        use_lmeds);
        if(isleft){
//            std::cout << "left p3p success = " << success << "-----" << std::endl;// 0 success=false
//            std::cout << "left p3p Twc (translation): " << Twc.translation().transpose() << std::endl;//左目平移向量
        } else {
//            std::cout << "right p3p success = " << success << "-----" << std::endl;// 0 success=false
//            std::cout << "right p3p Twc (translation): " << Twc.translation().transpose() << std::endl;//右目平移向量
        }

        if( pslamstate_->debug_ )
            std::cout << "\t>>> P3P-LMeds nb outliers : " << voutliersidx.size()<< " / inliers: " << vwpts.size() - voutliersidx.size() << " / total: " << vwpts.size() << std::endl;

        // Check that pose estim. was good enough
        size_t nbinliers = vwpts.size() - voutliersidx.size();

        if( !success
            || nbinliers < 5
            || Twc.translation().array().isInf().any()
            || Twc.translation().array().isNaN().any() )
        {
            if( pslamstate_->debug_ )
                std::cout << "\t>>> Not enough inliers for reliable pose est. Resetting KF state\n";

//            resetFrame();// todo mono_stereo
            resetFrame(pframe, pmap);

            return;
        }

        // Pose seems to be OK!

        // Update frame pose
//        pframe->setTwc(Twc);
        if(isleft){
            pframe->setTwc(Twc);//左目  世界点=左目到世界*左目点
        } else {
            pframe->setTwc(Twc * pslamstate_->T_right_left_);//右目 Twc是求出的右目位姿，转换为左目位姿存入frame  世界点=左目到世界*左目点=右目到世界*左目到右目*左目点
        }

        // Remove outliers before PnP refinement (a bit dirty)
        int k = 0;
        for( const auto &idx : voutliersidx ) {
            // MapManager is responsible for all removing operations
            pmap->removeObsFromCurFrameById(vkpids.at(idx-k), pframe);
            vkps.erase(vkps.begin() + idx - k);
            vwpts.erase(vwpts.begin() + idx - k);
            vkpids.erase(vkpids.begin() + idx - k);
            vscales.erase(vscales.begin() + idx - k);
            k++;
        }

        // Clear before robust PnP refinement using Ceres
        voutliersidx.clear();
    }

    // Ceres-based PnP (motion-only BA)
    bool buse_robust = true;
    bool bapply_l2_after_robust = pslamstate_->apply_l2_after_robust_;

    size_t nbmaxiters = 5;

    success =
            MultiViewGeometry::ceresPnP(
                    vkps, vwpts,
                    vscales,
                    Twc,
                    nbmaxiters,
                    pslamstate_->robust_mono_th_,
                    buse_robust,
                    bapply_l2_after_robust,
                    pcurframe_->pcalib_leftcam_->fx_, pcurframe_->pcalib_leftcam_->fy_,
                    pcurframe_->pcalib_leftcam_->cx_, pcurframe_->pcalib_leftcam_->cy_,
                    voutliersidx);
    if(isleft){
//        std::cout << "left pnp success = " << success << "-------" << std::endl;// 0 success=false
//        std::cout << "left pnp Twc (translation): " << Twc.translation().transpose() << std::endl;
    } else {
//        std::cout << "pnp right Twc (translation): " << (Twc.matrix() * pslamstate_->T_right_left_.matrix()).block<3, 1>(0, 3).transpose() << std::endl;
//        std::cout << "right pnp success = " << success << "-------" << std::endl;// 0 success=false
//        std::cout << "right pnp Twc (translation): " << Twc.translation().transpose() << std::endl;
    }

    // Check that pose estim. was good enough
    size_t nbinliers = vwpts.size() - voutliersidx.size();

    if( pslamstate_->debug_ )
        std::cout << "\t>>> Ceres PnP nb outliers : " << voutliersidx.size()<< " / inliers: " << nbinliers << " / total: " << vwpts.size() << std::endl;

    if( !success
        || nbinliers < 5
        || voutliersidx.size() > 0.5 * vwpts.size()
        || Twc.translation().array().isInf().any()
        || Twc.translation().array().isNaN().any())
    {
        if(isleft){
//            std::cout << "left fail\n";
        } else {
//            std::cout << "right fail\n";
        }

        if( !bdop3p ) {
            // Weird results, skipping here and applying p3p next
            if(isleft){
                bp3preq_ = true;
            } else {
                bp3preq_r_ = true;
            }
        }
        else if( pslamstate_->mono_ ) {

            if( pslamstate_->debug_ )
                std::cout << "\n \t>>> Not enough inliers for reliable pose est. Resetting KF state\n";

            resetFrame();//单目，不管
        }
        // else {
        // resetFrame();
        // motion_model_.reset();
        // }

        return;
    }

    if( Twc.translation().norm() > 10000. ){    // todo add 如果计算出的平移太大，计算错误，排除
//        std::cout<<"t toooooooooooo large!!!"<<std::endl;
        return;
    }

    // Pose seems to be OK!

    // Update frame pose
//    std::cout << "Twc: " << Twc.matrix() << std::endl;
//    pframe->setTwc(Twc);
    if(isleft){
        pframe->setTwc(Twc);//左目
    } else {
        pframe->setTwc(Twc * pslamstate_->T_right_left_);//右目 Twc是求出的右目位姿，转换为左目位姿存入frame
    }

    // Set p3p req to false as it is triggered either because
    // of bad PnP or by bad klt tracking
    if(isleft){
        bp3preq_ = false;
    } else {
        bp3preq_r_ = false;
    }

    // Remove outliers
    for( const auto & idx : voutliersidx ) {
        // MapManager is responsible for all removing operations
        pmap->removeObsFromCurFrameById(vkpids.at(idx), pframe);
    }

    // todo 验证：输出计算的位姿结果。新函数下 纯用左目 和 纯用右目 计算的位姿结果应基本相等。
//    if(isleft){
//        std::cout << "Twc only_left: " << Twc.matrix() << std::endl;//  左到世界
//        std::cout << "Twc only_left: todo"  << std::endl;//  左到世界
//    } else {
//         //输出 右到世界Twc*左到右T_right_left_
//        std::cout << "Twc only_right: " << Twc.matrix() * pslamstate_->T_right_left_.matrix() << std::endl;// 右到世界*左到右
//    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_computePose");
}



bool VisualFrontEnd::checkReadyForInit()
{
    double avg_rot_parallax = computeParallax(pcurframe_->kfid_, false);

    std::cout << "\n \t>>> Init current parallax (" << avg_rot_parallax <<" px)\n"; 

    if( avg_rot_parallax > pslamstate_->finit_parallax_ ) {
        auto cb = std::chrono::high_resolution_clock::now();
        
        // Get prev. KF
        auto pkf = pmap_->map_pkfs_.at(pcurframe_->kfid_);
        if( pkf == nullptr ) {
            return false;
        }

        // Get cur. Frame nb kps
        size_t nbkps = pcurframe_->nbkps_;

        if( nbkps < 8 ) {
            std::cout << "\nNot enough kps to compute 5-pt Essential Matrix\n";
            return false;
        }

        // Setup Essential Matrix computation for OpenGV-based filtering
        std::vector<int> vkpsids, voutliersidx;
        vkpsids.reserve(nbkps);
        voutliersidx.reserve(nbkps);

        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vkfbvs, vcurbvs;
        vkfbvs.reserve(nbkps);
        vcurbvs.reserve(nbkps);

        Eigen::Matrix3d Rkfcur = pkf->getTcw().rotationMatrix() * pcurframe_->getTwc().rotationMatrix();
        int nbparallax = 0;
        float avg_rot_parallax = 0.;

        // Get bvs and compute the rotation compensated parallax for all cur kps
        // for( const auto &kp : pcurframe_->getKeypoints() ) {
        for( const auto &it : pcurframe_->mapkps_ ) {
            auto &kp = it.second;
            // Get the prev. KF related kp if it exists
            auto kfkp = pkf->getKeypointById(kp.lmid_);

            if( kfkp.lmid_ != kp.lmid_ ) {
                continue;
            }

            // Store the bvs and their ids
            vkfbvs.push_back(kfkp.bv_);
            vcurbvs.push_back(kp.bv_);
            vkpsids.push_back(kp.lmid_);

            // Compute rotation compensated parallax
            Eigen::Vector3d rotbv = Rkfcur * kp.bv_;

            Eigen::Vector3d unpx = pcurframe_->pcalib_leftcam_->K_ * rotbv;
            cv::Point2f rotpx(unpx.x() / unpx.z(), unpx.y() / unpx.z());

            avg_rot_parallax += cv::norm(rotpx - kfkp.unpx_);
            nbparallax++;
        }

        if( nbparallax < 8 ) {
            std::cout << "\nNot enough prev KF kps to compute 5-pt Essential Matrix\n";
            return false;
        }

        // Average parallax
        avg_rot_parallax /= (nbparallax);

        if( avg_rot_parallax < pslamstate_->finit_parallax_ ) {
            std::cout << "\n \t>>> Not enough parallax (" << avg_rot_parallax <<" px) to compute 5-pt Essential Matrix\n";
            return false;
        }

        bool do_optimize = true;

        Eigen::Matrix3d Rkfc;
        Eigen::Vector3d tkfc;
        Rkfc.setIdentity();
        tkfc.setZero();

        std::cout << "\n \t>>> 5-pt EssentialMatrix Ransac :";
        std::cout << "\n \t>>> nb pts : " << nbkps;
        std::cout << " / avg. parallax : " << avg_rot_parallax;
        std::cout << " / nransac_iter_ : " << pslamstate_->nransac_iter_;
        std::cout << " / fransac_err_ : " << pslamstate_->fransac_err_;
        std::cout << " / bdo_random : " << pslamstate_->bdo_random;
        std::cout << "\n\n";
        
        bool success = 
            MultiViewGeometry::compute5ptEssentialMatrix
                    (vkfbvs, vcurbvs, pslamstate_->nransac_iter_, pslamstate_->fransac_err_, 
                    do_optimize, pslamstate_->bdo_random, 
                    pcurframe_->pcalib_leftcam_->fx_, 
                    pcurframe_->pcalib_leftcam_->fy_, 
                    Rkfc, tkfc, 
                    voutliersidx);

        std::cout << "\n \t>>> Epipolar nb outliers : " << voutliersidx.size();

        if( !success ) {
            std::cout << "\n \t>>> No pose could be computed from 5-pt EssentialMatrix\n";
            return false;
        }

        // Remove outliers from cur. Frame
        for( const auto & idx : voutliersidx ) {
            // MapManager is responsible for all the removing operations.
            pmap_->removeObsFromCurFrameById(vkpsids.at(idx));
        }

        // Arbitrary scale
        tkfc.normalize();
        tkfc = tkfc.eval() * 0.25;

        std::cout << "\n \t>>> Essential Mat init : " << tkfc.transpose();

        pcurframe_->setTwc(Rkfc, tkfc);
        
        auto ce = std::chrono::high_resolution_clock::now();
        std::cout << "\n \t>>> Essential Mat Intialization run time : " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(ce-cb).count()
            << "[ms]" << std::endl;

        return true;
    }

    return false;
}

bool VisualFrontEnd::checkNewKfReq()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_checkNewKfReq");

    // Get prev. KF
    auto pkfit = pmap_->map_pkfs_.find(pcurframe_->kfid_);

    if( pkfit == pmap_->map_pkfs_.end() ) {
        return false; // Should not happen
    }
    auto pkf = pkfit->second;

    // Compute median parallax
    double med_rot_parallax = 0.;

    // unrot : false / median : true / only_2d : false
    med_rot_parallax = computeParallax(pkf->kfid_, true, true, false);

    // Id diff with last KF
    int nbimfromkf = pcurframe_->id_-pkf->id_;

    if( pcurframe_->noccupcells_ < 0.33 * pslamstate_->nbmaxkps_
        && nbimfromkf >= 5
        && !pslamstate_->blocalba_is_on_ )
    {
        return true;
    }

    if( pcurframe_->nb3dkps_ < 20 &&
        nbimfromkf >= 2 )
    {
        return true;
    }

    if( pcurframe_->nb3dkps_ > 0.5 * pslamstate_->nbmaxkps_ 
        && (pslamstate_->blocalba_is_on_ || nbimfromkf < 2) )
    {
        return false;
    }

    // Time diff since last KF in sec.
    double time_diff = pcurframe_->img_time_ - pkf->img_time_;

    if( pslamstate_->stereo_ && time_diff > 1. 
        && !pslamstate_->blocalba_is_on_ )
    {
        return true;
    }

    bool cx = med_rot_parallax >= pslamstate_->finit_parallax_ / 2.
        || (pslamstate_->stereo_ && !pslamstate_->blocalba_is_on_ && pcurframe_->id_-pkf->id_ > 2);

    bool c0 = med_rot_parallax >= pslamstate_->finit_parallax_;
    bool c1 = pcurframe_->nb3dkps_ < 0.75 * pkf->nb3dkps_;
    bool c2 = pcurframe_->noccupcells_ < 0.5 * pslamstate_->nbmaxkps_
                && pcurframe_->nb3dkps_ < 0.85 * pkf->nb3dkps_
                && !pslamstate_->blocalba_is_on_;
    
    bool bkfreq = (c0 || c1 || c2) && cx;

    if( bkfreq && pslamstate_->debug_ ) {
        
        std::cout << "\n\n----------------------------------------------------------------------";
        std::cout << "\n>>> Check Keyframe conditions :";
        std::cout << "\n> pcurframe_->id_ = " << pcurframe_->id_ << " / prev kf frame_id : " << pkf->id_;
        std::cout << "\n> Prev KF nb 3d kps = " << pkf->nb3dkps_ << " / Cur Frame = " << pcurframe_->nb3dkps_;
        std::cout << " / Cur Frame occup cells = " << pcurframe_->noccupcells_ << " / parallax = " << med_rot_parallax;
        std::cout << "\n-------------------------------------------------------------------\n\n";
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_checkNewKfReq");

    return bkfreq;
}

bool VisualFrontEnd::checkNewKfReq(std::shared_ptr<Frame>& pframe, std::shared_ptr<MapManager>& pmap, bool isleft)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_checkNewKfReq");

    // Get prev. KF
    auto pkfit = pmap->map_pkfs_.find(pframe->kfid_);

    if( pkfit == pmap->map_pkfs_.end() ) {
        return false; // Should not happen
    }
    auto pkf = pkfit->second;

    // Compute median parallax
    double med_rot_parallax = 0.;

    // unrot : false / median : true / only_2d : false
//    med_rot_parallax = computeParallax(pkf->kfid_, true, true, false);// todo mono_stereo
    med_rot_parallax = computeParallax(pkf->kfid_, pframe, pmap, isleft, true, true, false);

    // Id diff with last KF
    int nbimfromkf = pframe->id_-pkf->id_;

    if( pframe->noccupcells_ < 0.33 * pslamstate_->nbmaxkps_
        && nbimfromkf >= 5
        && !pslamstate_->blocalba_is_on_ )
    {
        return true;
    }

    if( pframe->nb3dkps_ < 20 &&
        nbimfromkf >= 2 )
    {
        return true;
    }

    if( pframe->nb3dkps_ > 0.5 * pslamstate_->nbmaxkps_
        && (pslamstate_->blocalba_is_on_ || nbimfromkf < 2) )
    {
        return false;
    }

    // Time diff since last KF in sec.
    double time_diff = pframe->img_time_ - pkf->img_time_;

    if( (pslamstate_->stereo_ || pslamstate_->mono_stereo_) && time_diff > 1.
//    if( (pslamstate_->stereo_ || pslamstate_->mono_stereo_) && time_diff > 2.   //test：关键帧变化不大，效果变化不大
        && !pslamstate_->blocalba_is_on_ )
    {
        return true;
    }

    //test
//    pslamstate_->finit_parallax_ *= 2.8;  // 提高视差阈值（最重要）：稍作扩大，能明显减少关键帧，但效果不好

    bool cx = med_rot_parallax >= pslamstate_->finit_parallax_ / 2.
//    bool cx = med_rot_parallax >= pslamstate_->finit_parallax_ *1.2 / 2. //test：关键帧变化不大，效果变差
              || ((pslamstate_->stereo_ || pslamstate_->mono_stereo_) && !pslamstate_->blocalba_is_on_ && pframe->id_-pkf->id_ > 2);

    bool c0 = med_rot_parallax >= pslamstate_->finit_parallax_;
    bool c1 = pframe->nb3dkps_ < 0.75 * pkf->nb3dkps_;
    bool c2 = pframe->noccupcells_ < 0.5 * pslamstate_->nbmaxkps_
              && pframe->nb3dkps_ < 0.85 * pkf->nb3dkps_
              && !pslamstate_->blocalba_is_on_;

//test：关键帧变化不大，效果变化不大
//    bool c1 = pframe->nb3dkps_ < 0.5 * pkf->nb3dkps_;
//    bool c2 = pframe->noccupcells_ < 0.3 * pslamstate_->nbmaxkps_
//              && pframe->nb3dkps_ < 0.7 * pkf->nb3dkps_
//              && !pslamstate_->blocalba_is_on_;

    bool bkfreq = (c0 || c1 || c2) && cx;

    if( bkfreq && pslamstate_->debug_ ) {

        std::cout << "\n\n----------------------------------------------------------------------";
        std::cout << "\n>>> Check Keyframe conditions :";
        std::cout << "\n> pframe->id_ = " << pframe->id_ << " / prev kf frame_id : " << pkf->id_;
        std::cout << "\n> Prev KF nb 3d kps = " << pkf->nb3dkps_ << " / Cur Frame = " << pframe->nb3dkps_;
        std::cout << " / Cur Frame occup cells = " << pframe->noccupcells_ << " / parallax = " << med_rot_parallax;
        std::cout << "\n-------------------------------------------------------------------\n\n";
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_checkNewKfReq");

    return bkfreq;
}

bool VisualFrontEnd::checkNewKfReq_diverg(std::shared_ptr<Frame>& pframe_l, std::shared_ptr<Frame>& pframe_r, std::shared_ptr<MapManager>& pmap_l, std::shared_ptr<MapManager>& pmap_r)
{
//    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
//        Profiler::Start("2.FE_TM_checkNewKfReq");
//
//    // Get prev. KF
//    auto pkfit_l = pmap_l->map_pkfs_.find(pframe_l->kfid_);
//    auto pkfit_r = pmap_r->map_pkfs_.find(pframe_r->kfid_);
//
//    if( pkfit_l == pmap_l->map_pkfs_.end() || pkfit_r == pmap_r->map_pkfs_.end() ) {
//        return false; // Should not happen
//    }
//    auto pkf_l = pkfit_l->second;
//    auto pkf_r = pkfit_r->second;
//
//    // Compute median parallax
//    double med_rot_parallax = 0.;
//
//    // unrot : false / median : true / only_2d : false
////    med_rot_parallax = computeParallax(pkf->kfid_, true, true, false);// todo mono_stereo
//    med_rot_parallax = computeParallax(pkf->kfid_, pframe, pmap, isleft, true, true, false);//两个相机视差应该怎么算
//
//    // Id diff with last KF
//    int nbimfromkf = pframe->id_-pkf->id_;
//
//    if( pframe->noccupcells_ < 0.33 * pslamstate_->nbmaxkps_    //diverg
//        && nbimfromkf >= 5
//        && !pslamstate_->blocalba_is_on_ )
//    {
//        return true;
//    }
//
//    if( pframe->nb3dkps_ < 20 &&
//        nbimfromkf >= 2 )
//    {
//        return true;
//    }
//
//    if( pframe->nb3dkps_ > 0.5 * pslamstate_->nbmaxkps_
//        && (pslamstate_->blocalba_is_on_ || nbimfromkf < 2) )
//    {
//        return false;
//    }
//
//    // Time diff since last KF in sec.
//    double time_diff = pframe->img_time_ - pkf->img_time_;
//
//    if( (pslamstate_->stereo_ || pslamstate_->mono_stereo_) && time_diff > 1.
//        //    if( (pslamstate_->stereo_ || pslamstate_->mono_stereo_) && time_diff > 2.   //test：关键帧变化不大，效果变化不大
//        && !pslamstate_->blocalba_is_on_ )
//    {
//        return true;
//    }
//
//    //test
////    pslamstate_->finit_parallax_ *= 2.8;  // 提高视差阈值（最重要）：稍作扩大，能明显减少关键帧，但效果不好
//
//    bool cx = med_rot_parallax >= pslamstate_->finit_parallax_ / 2.
//              //    bool cx = med_rot_parallax >= pslamstate_->finit_parallax_ *1.2 / 2. //test：关键帧变化不大，效果变差
//              || ((pslamstate_->stereo_ || pslamstate_->mono_stereo_) && !pslamstate_->blocalba_is_on_ && pframe->id_-pkf->id_ > 2);
//
//    bool c0 = med_rot_parallax >= pslamstate_->finit_parallax_;
//    bool c1 = pframe->nb3dkps_ < 0.75 * pkf->nb3dkps_;
//    bool c2 = pframe->noccupcells_ < 0.5 * pslamstate_->nbmaxkps_
//              && pframe->nb3dkps_ < 0.85 * pkf->nb3dkps_
//              && !pslamstate_->blocalba_is_on_;
//
////test：关键帧变化不大，效果变化不大
////    bool c1 = pframe->nb3dkps_ < 0.5 * pkf->nb3dkps_;
////    bool c2 = pframe->noccupcells_ < 0.3 * pslamstate_->nbmaxkps_
////              && pframe->nb3dkps_ < 0.7 * pkf->nb3dkps_
////              && !pslamstate_->blocalba_is_on_;
//
//    bool bkfreq = (c0 || c1 || c2) && cx;
//
//    if( bkfreq && pslamstate_->debug_ ) {
//
//        std::cout << "\n\n----------------------------------------------------------------------";
//        std::cout << "\n>>> Check Keyframe conditions :";
//        std::cout << "\n> pframe->id_ = " << pframe->id_ << " / prev kf frame_id : " << pkf->id_;
//        std::cout << "\n> Prev KF nb 3d kps = " << pkf->nb3dkps_ << " / Cur Frame = " << pframe->nb3dkps_;
//        std::cout << " / Cur Frame occup cells = " << pframe->noccupcells_ << " / parallax = " << med_rot_parallax;
//        std::cout << "\n-------------------------------------------------------------------\n\n";
//    }
//
//    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
//        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_checkNewKfReq");
//
//    return bkfreq;
}


// This function computes the parallax (in px.) between cur. Frame 
// and the provided KF id.
float VisualFrontEnd::computeParallax(const int kfid, bool do_unrot, bool bmedian, bool b2donly)
{
    // Get prev. KF
    auto pkfit = pmap_->map_pkfs_.find(kfid);
    
    if( pkfit == pmap_->map_pkfs_.end() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n[Visual Front End] Error in computeParallax ! Prev KF #" 
                    << kfid << " does not exist!\n";
        return 0.;
    }

    // Compute relative rotation between cur Frame 
    // and prev. KF if required
    Eigen::Matrix3d Rkfcur(Eigen::Matrix3d::Identity());
    if( do_unrot ) {
        Eigen::Matrix3d Rkfw = pkfit->second->getRcw();
        Eigen::Matrix3d Rwcur = pcurframe_->getRwc();
        Rkfcur = Rkfw * Rwcur;
    }

    // Compute parallax 
    float avg_parallax = 0.;
    int nbparallax = 0;

    std::set<float> set_parallax;

    // Compute parallax for all kps seen in prev. KF{
    for( const auto &it : pcurframe_->mapkps_ ) 
    {
        if( b2donly && it.second.is3d_ ) {
            continue;
        }

        auto &kp = it.second;
        // Get prev. KF kp if it exists
        auto kfkp = pkfit->second->getKeypointById(kp.lmid_);

        if( kfkp.lmid_ != kp.lmid_ ) {
            continue;
        }

        // Compute parallax with unpx pos.
        cv::Point2f unpx = kp.unpx_;

        // Rotate bv into KF cam frame and back project into image
        if( do_unrot ) {
            unpx = pkfit->second->projCamToImage(Rkfcur * kp.bv_);
        }

        // Compute rotation-compensated parallax
        float parallax = cv::norm(unpx - kfkp.unpx_);
        avg_parallax += parallax;
        nbparallax++;

        if( bmedian ) {
            set_parallax.insert(parallax);
        }
    }

    if( nbparallax == 0 ) {
        return 0.;
    }

    // Average parallax
    avg_parallax /= nbparallax;

    if( bmedian ) 
    {
        auto it = set_parallax.begin();
        std::advance(it, set_parallax.size() / 2);
        avg_parallax = *it;
    }

    return avg_parallax;
}

float VisualFrontEnd::computeParallax(const int kfid, std::shared_ptr<Frame>& pframe, std::shared_ptr<MapManager>& pmap, bool isleft, bool do_unrot, bool bmedian, bool b2donly)
{
    // Get prev. KF
    auto pkfit = pmap->map_pkfs_.find(kfid);

    if( pkfit == pmap->map_pkfs_.end() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n[Visual Front End] Error in computeParallax ! Prev KF #"
                      << kfid << " does not exist!\n";
        return 0.;
    }

    // Compute relative rotation between cur Frame
    // and prev. KF if required
    Eigen::Matrix3d Rkfcur(Eigen::Matrix3d::Identity());
    if( do_unrot ) {
        Eigen::Matrix3d Rkfw = pkfit->second->getRcw();
        Eigen::Matrix3d Rwcur = pframe->getRwc();
        Rkfcur = Rkfw * Rwcur;// 左目/右目
    }

    // Compute parallax
    float avg_parallax = 0.;
    int nbparallax = 0;

    std::set<float> set_parallax;

    // Compute parallax for all kps seen in prev. KF{
    for( const auto &it : pframe->mapkps_ )
    {
        if( b2donly && it.second.is3d_ ) {
            continue;
        }

        auto &kp = it.second;
        // Get prev. KF kp if it exists
        auto kfkp = pkfit->second->getKeypointById(kp.lmid_);

        if( kfkp.lmid_ != kp.lmid_ ) {
            continue;
        }

        // Compute parallax with unpx pos.
        cv::Point2f unpx = kp.unpx_;

        // Rotate bv into KF cam frame and back project into image
        if( do_unrot ) {
            if(isleft){
                unpx = pkfit->second->projCamToImage(Rkfcur * kp.bv_);//
            } else {
                unpx = pkfit->second->projCamToRightImage(Rkfcur * kp.bv_);//用的原本自带的投影函数
            }
        }

        // Compute rotation-compensated parallax
        float parallax = cv::norm(unpx - kfkp.unpx_);
        avg_parallax += parallax;
        nbparallax++;

        if( bmedian ) {
            set_parallax.insert(parallax);
        }
    }

    if( nbparallax == 0 ) {
        return 0.;
    }

    // Average parallax
    avg_parallax /= nbparallax;

    if( bmedian )
    {
        auto it = set_parallax.begin();
        std::advance(it, set_parallax.size() / 2);
        avg_parallax = *it;
    }

    return avg_parallax;
}

void VisualFrontEnd::preprocessImage(cv::Mat &img_raw)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_preprocessImage");

    // Set cur raw img
    // left_raw_img_ = img_raw;

    // Update prev img
    if( !pslamstate_->btrack_keyframetoframe_ ) {
        // cur_img_.copyTo(prev_img_);
        cv::swap(cur_img_, prev_img_);
    }

    // Update cur img
    if( pslamstate_->use_clahe_ ) {
        ptracker_->pclahe_->apply(img_raw, cur_img_);
    } else {
        cur_img_ = img_raw;
    }

    // Pre-building the pyramid used for KLT speed-up
    if( pslamstate_->do_klt_ ) {

        // If tracking from prev image, swap the pyramid
        if( !cur_pyr_.empty() && !pslamstate_->btrack_keyframetoframe_ ) {
            prev_pyr_.swap(cur_pyr_);
        }

        cv::buildOpticalFlowPyramid(cur_img_, cur_pyr_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_preprocessImage");
}

void VisualFrontEnd::preprocessImage(cv::Mat &iml_raw, cv::Mat &imr_raw, cv::Mat &imlm_raw, cv::Mat &imls_raw, cv::Mat &imrm_raw, cv::Mat &imrs_raw)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_preprocessImage");

    // Set cur raw img
    // left_raw_img_ = img_raw;

    // Update prev img
    if( !pslamstate_->btrack_keyframetoframe_ ) {
        // cur_img_.copyTo(prev_img_);
        cv::swap(cur_img_, prev_img_);
        cv::swap(cur_imgl_, prev_imgl_);
        cv::swap(cur_imgr_, prev_imgr_);
        cv::swap(cur_imglm_, prev_imglm_);
        cv::swap(cur_imgls_, prev_imgls_);
        cv::swap(cur_imgrm_, prev_imgrm_);
        cv::swap(cur_imgrs_, prev_imgrs_);
    }

    // Update cur img
    if( pslamstate_->use_clahe_ ) {
        ptracker_->pclahe_->apply(iml_raw, cur_img_);
        ptracker_->pclahe_->apply(iml_raw, cur_imgl_);
        ptracker_->pclahe_->apply(imr_raw, cur_imgr_);
        if(pslamstate_->angle_m >1){//单目区不能太小
            ptracker_->pclahe_m_->apply(imlm_raw, cur_imglm_);
            ptracker_->pclahe_m_->apply(imrm_raw, cur_imgrm_);
        }
        ptracker_->pclahe_s_->apply(imls_raw, cur_imgls_);
        ptracker_->pclahe_s_->apply(imrs_raw, cur_imgrs_);
    } else {
        cur_img_ = iml_raw;
        cur_imgl_ = iml_raw;
        cur_imgr_ = imr_raw;
        cur_imglm_ = imlm_raw;
        cur_imgls_ = imls_raw;
        cur_imgrm_ = imrm_raw;
        cur_imgrs_ = imrs_raw;
    }

    // Pre-building the pyramid used for KLT speed-up
    if( pslamstate_->do_klt_ ) {

        // If tracking from prev image, swap the pyramid
        if( !cur_pyrl_.empty() && !cur_pyrr_.empty() && !cur_pyrlm_.empty() && !cur_pyrls_.empty() && !cur_pyrrm_.empty() && !cur_pyrrs_.empty() && !pslamstate_->btrack_keyframetoframe_ ) {
            prev_pyr_.swap(cur_pyr_);
            prev_pyrl_.swap(cur_pyrl_);
            prev_pyrr_.swap(cur_pyrr_);
            prev_pyrlm_.swap(cur_pyrlm_);
            prev_pyrls_.swap(cur_pyrls_);
            prev_pyrrm_.swap(cur_pyrrm_);
            prev_pyrrs_.swap(cur_pyrrs_);
        }

        cv::buildOpticalFlowPyramid(cur_img_, cur_pyr_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
        cv::buildOpticalFlowPyramid(cur_imgl_, cur_pyrl_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
        cv::buildOpticalFlowPyramid(cur_imgr_, cur_pyrr_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
        if(pslamstate_->angle_m >1){
            cv::buildOpticalFlowPyramid(cur_imglm_, cur_pyrlm_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
            cv::buildOpticalFlowPyramid(cur_imgrm_, cur_pyrrm_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
        }
        cv::buildOpticalFlowPyramid(cur_imgls_, cur_pyrls_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
        cv::buildOpticalFlowPyramid(cur_imgrs_, cur_pyrrs_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_preprocessImage");
}


// Reset current Frame state
void VisualFrontEnd::resetFrame()
{
    auto mapkps = pcurframe_->mapkps_;
    for( const auto &kpit : mapkps ) {
        pmap_->removeObsFromCurFrameById(kpit.first);
    }
    pcurframe_->mapkps_.clear();
    pcurframe_->vgridkps_.clear();
    pcurframe_->vgridkps_.resize( pcurframe_->ngridcells_ );

    // Do not clear those as we keep the same pose
    // and hence keep a chance to retrack the previous map
    //
    // pcurframe_->map_covkfs_.clear();
    // pcurframe_->set_local_mapids_.clear();

    pcurframe_->nbkps_ = 0;
    pcurframe_->nb2dkps_ = 0;
    pcurframe_->nb3dkps_ = 0;
    pcurframe_->nb_stereo_kps_ = 0;

    pcurframe_->noccupcells_ = 0;
}

void VisualFrontEnd::resetFrame(std::shared_ptr<Frame>& pframe, std::shared_ptr<MapManager>& pmap)
{
    auto mapkps = pframe->mapkps_;
    for( const auto &kpit : mapkps ) {
        pmap->removeObsFromCurFrameById(kpit.first, pframe);
    }
    pframe->mapkps_.clear();
    pframe->vgridkps_.clear();
    pframe->vgridkps_.resize( pframe->ngridcells_ );
    pframe->vgridkps_m_.clear();
    pframe->vgridkps_m_.resize( pframe->ngridcells_m_ );
    pframe->vgridkps_s_.clear();
    pframe->vgridkps_s_.resize( pframe->ngridcells_s_ );

    // Do not clear those as we keep the same pose
    // and hence keep a chance to retrack the previous map
    //
    // pcurframe_->map_covkfs_.clear();
    // pcurframe_->set_local_mapids_.clear();

    pframe->nbkps_ = 0;
    pframe->nb2dkps_ = 0;
    pframe->nb3dkps_ = 0;
    pframe->nb_stereo_kps_ = 0;

    pframe->noccupcells_ = 0;
}

// Reset VisualFrontEnd
void VisualFrontEnd::reset()
{
    cur_img_.release();
    prev_img_.release();
    cur_imgl_.release();
    prev_imgr_.release();
    cur_imglm_.release();
    cur_imgls_.release();
    cur_imgrm_.release();
    cur_imgrs_.release();

    // left_raw_img_.release();

    cur_pyr_.clear();
    prev_pyr_.clear();
    cur_pyrl_.clear();
    prev_pyrl_.clear();
    cur_pyrr_.clear();
    prev_pyrr_.clear();
    cur_pyrlm_.clear();
    prev_pyrlm_.clear();
    cur_pyrls_.clear();
    prev_pyrls_.clear();
    cur_pyrrm_.clear();
    prev_pyrrm_.clear();
    cur_pyrrs_.clear();
    prev_pyrrs_.clear();
    kf_pyr_.clear();
    kf_pyrl_.clear();
    kf_pyrr_.clear();
    kf_pyrlm_.clear();
    kf_pyrls_.clear();
    kf_pyrrm_.clear();
    kf_pyrrs_.clear();
}
