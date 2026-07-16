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

#include "frame.hpp"

Frame::Frame()
    : id_(-1), kfid_(0), img_time_(0.), nbkps_(0), nb2dkps_(0), nb3dkps_(0), nb_stereo_kps_(0),
      Frl_(Eigen::Matrix3d::Zero()), Fcv_(cv::Mat::zeros(3,3,CV_64F))
{}


Frame::Frame(std::shared_ptr<CameraCalibration> pcalib_left, const size_t ncellsize)
    : id_(-1), kfid_(0), img_time_(0.), ncellsize_(ncellsize), nbkps_(0),
      nb2dkps_(0), nb3dkps_(0), nb_stereo_kps_(0),
      pcalib_leftcam_(pcalib_left)
{
    // Init grid from images size
    nbwcells_ = static_cast<size_t>(ceilf( static_cast<float>(pcalib_leftcam_->img_w_) / ncellsize_ ));
    nbhcells_ = static_cast<size_t>(ceilf( static_cast<float>(pcalib_leftcam_->img_h_) / ncellsize_ ));
    ngridcells_ =  nbwcells_ * nbhcells_ ;
    noccupcells_ = 0;
    
    vgridkps_.resize( ngridcells_ );
}


Frame::Frame(std::shared_ptr<CameraCalibration> pcalib_left, std::shared_ptr<CameraCalibration> pcalib_right, const size_t ncellsize)
    : id_(-1), kfid_(0), img_time_(0.), ncellsize_(ncellsize), nbkps_(0), nb2dkps_(0), nb3dkps_(0), nb_stereo_kps_(0),
    pcalib_leftcam_(pcalib_left), pcalib_rightcam_(pcalib_right)
{
    Eigen::Vector3d t = pcalib_rightcam_->Tcic0_.translation();
    Eigen::Matrix3d tskew;
    tskew << 0., -t(2), t(1),
            t(2), 0., -t(0),
            -t(1), t(0), 0.;

    Eigen::Matrix3d R = pcalib_rightcam_->Tcic0_.rotationMatrix();

    Frl_ = pcalib_rightcam_->K_.transpose().inverse() * tskew * R * pcalib_leftcam_->iK_;

    cv::eigen2cv(Frl_, Fcv_);

    // Init grid from images size
    nbwcells_ = ceil( (float)pcalib_leftcam_->img_w_ / ncellsize_ );
    nbhcells_ = ceil( (float)pcalib_leftcam_->img_h_ / ncellsize_ );
    ngridcells_ =  nbwcells_ * nbhcells_ ;
    noccupcells_ = 0;
    
    vgridkps_.resize( ngridcells_ );
}

Frame::Frame(std::shared_ptr<CameraCalibration> pcalib_left, std::shared_ptr<CameraCalibration> pcalib_right, std::shared_ptr<CameraCalibration> pcalib_left_mono, std::shared_ptr<CameraCalibration> pcalib_right_mono, std::shared_ptr<CameraCalibration> pcalib_left_stereo, std::shared_ptr<CameraCalibration> pcalib_right_stereo, const size_t ncellsize, double theta)
        : id_(-1), kfid_(0), img_time_(0.), ncellsize_(ncellsize), nbkps_(0), nb2dkps_(0), nb3dkps_(0), nb_stereo_kps_(0),
          pcalib_leftcam_(pcalib_left), pcalib_rightcam_(pcalib_right), pcalib_leftcam_m_(pcalib_left_mono), pcalib_rightcam_m_(pcalib_right_mono), pcalib_leftcam_s_(pcalib_left_stereo), pcalib_rightcam_s_(pcalib_right_stereo), theta_(theta)
{
    Eigen::Vector3d t = pcalib_rightcam_->Tcic0_.translation();//这里用的是Tcic0_=T_left_leftm_.t，之前传入的T_left_leftm_=Tc0ci_
    Eigen::Matrix3d tskew;
    tskew << 0., -t(2), t(1),
            t(2), 0., -t(0),
            -t(1), t(0), 0.;

    Eigen::Matrix3d R = pcalib_rightcam_->Tcic0_.rotationMatrix();

    Frl_ = pcalib_rightcam_->K_.transpose().inverse() * tskew * R * pcalib_leftcam_->iK_;// 原算法使用的 左右相机 发散双目

    Trl_ = Sophus::SE3d(R, t);
//    std::cout<<"1.0 Trl_ = "<<Trl_.matrix()<<std::endl;

    cv::eigen2cv(Frl_, Fcv_);

    // 111-mono_stereo
    // 虚拟双目区基础矩阵 测试程序：test_T.cpp
    Eigen::Matrix4d T_left_lefts = pcalib_leftcam_s_->Tc0ci_.matrix(); // 左双目区到左目
    Eigen::Matrix4d T_right_rights = pcalib_rightcam_s_->Tc0ci_.matrix(); // 右双目区到右目
    Eigen::Matrix4d T_left_right = pcalib_rightcam_->Tc0ci_.matrix(); // 右目到左目
    // 计算左双目区到右双目区的变换
    Eigen::Matrix4d T_lefts_rights = T_left_lefts.inverse() * T_left_right * T_right_rights;//  rs到ls=l到ls*r到l*rs到r
    std::cout << "T_lefts_rights:" << std::endl << T_lefts_rights << std::endl;//这个结果和测试程序一样，但下面求Frl要用反过来的

    Eigen::Matrix4d T_lefts_left = pcalib_leftcam_s_->Tcic0_.matrix(); // 左目到左双目区
    Eigen::Matrix4d T_rights_right = pcalib_rightcam_s_->Tcic0_.matrix(); // 右目到右双目区
    Eigen::Matrix4d T_right_left = pcalib_rightcam_->Tcic0_.matrix(); // 左目到右目
    Eigen::Matrix4d T_rights_lefts = T_rights_right * T_right_left * T_lefts_left.inverse();// ls到rs=r到rs*l到r*ls到l
    std::cout << "T_rights_lefts:" << std::endl << T_rights_lefts << std::endl;//这个结果和测试程序一样，但下面求Frl要用反过来的
    //提取平移和旋转矩阵
    Eigen::Matrix3d R_rights_lefts = T_rights_lefts.block<3, 3>(0, 0);
    Eigen::Vector3d t_rights_lefts = T_rights_lefts.block<3, 1>(0, 3);
    Eigen::Matrix3d tskew_s;
    tskew_s << 0., -t_rights_lefts(2), t_rights_lefts(1),
            t_rights_lefts(2), 0., -t_rights_lefts(0),
            -t_rights_lefts(1), t_rights_lefts(0), 0.;
    Frl_s_ = pcalib_rightcam_s_->K_.transpose().inverse() * tskew_s * R_rights_lefts * pcalib_leftcam_s_->iK_;// 左右双目 发散双目
    cv::eigen2cv(Frl_s_, Fcv_s_);

    // Init grid from images size
    nbwcells_ = ceil( (float)pcalib_leftcam_->img_w_ / ncellsize_ );
    nbhcells_ = ceil( (float)pcalib_leftcam_->img_h_ / ncellsize_ );
    ngridcells_ =  nbwcells_ * nbhcells_ ;
    noccupcells_ = 0;

    vgridkps_.resize( ngridcells_ );

    // 111-mono_stereo
    nbwcells_s_ = ceil( (float)pcalib_leftcam_s_->img_w_ / ncellsize_ );
    nbhcells_s_ = ceil( (float)pcalib_leftcam_s_->img_h_ / ncellsize_ );
    ngridcells_s_ =  nbwcells_s_ * nbhcells_s_ ;
    nbwcells_m_ = ceil( (float)pcalib_leftcam_m_->img_w_ / ncellsize_ );
    nbhcells_m_ = ceil( (float)pcalib_leftcam_m_->img_h_ / ncellsize_ );
    ngridcells_m_ =  nbwcells_m_ * nbhcells_m_ ;
//    std::cout<<"nbwcells_s_ = "<<nbwcells_s_<<std::endl;
    vgridkps_m_.resize( ngridcells_m_ );
    vgridkps_s_.resize( ngridcells_s_ );
//    std::cout<<"vgridkps_s_.size = "<<vgridkps_s_.size()<<std::endl;

    theta_ = theta;
    std::cout<<"theta_ = "<<theta_<<std::endl;
    Sophus::SE3d fixed_Twc(
//                Sophus::SO3d::rotY(30 * M_PI / 180), // 例如绕Z轴旋转30度
            Sophus::SO3d::rotY(-1 * theta_ * M_PI / 180), // 例如绕Z轴旋转30度
            Eigen::Vector3d(0.0, 0.0, 0.0)       // 固定平移向量
    );
    Twc_ = fixed_Twc;// todo test fake init
    Tcw_ = fixed_Twc.inverse();// todo test fake init
}

//Frame::Frame(const Frame &F)
//    : id_(F.id_), kfid_(F.kfid_), img_time_(F.img_time_), mapkps_(F.mapkps_), vgridkps_(F.vgridkps_), ngridcells_(F.ngridcells_), noccupcells_(F.noccupcells_),
//    ncellsize_(F.ncellsize_), nbwcells_(F.nbwcells_), nbhcells_(F.nbhcells_), nbkps_(F.nbkps_), nb2dkps_(F.nb2dkps_), nb3dkps_(F.nb3dkps_),
//    nb_stereo_kps_(F.nb_stereo_kps_), Twc_(F.Twc_), Tcw_(F.Tcw_), pcalib_leftcam_(F.pcalib_leftcam_),
//    pcalib_rightcam_(F.pcalib_rightcam_), Frl_(F.Frl_), Fcv_(F.Fcv_), map_covkfs_(F.map_covkfs_), set_local_mapids_(F.set_local_mapids_)
//{}
Frame::Frame(const Frame &F)    //新的拷贝构造函数，把新建的变量也拷贝
        : id_(F.id_), kfid_(F.kfid_), img_time_(F.img_time_), mapkps_(F.mapkps_), vgridkps_(F.vgridkps_), vgridkps_m_(F.vgridkps_m_), vgridkps_s_(F.vgridkps_s_), ngridcells_(F.ngridcells_), noccupcells_(F.noccupcells_),
          ncellsize_(F.ncellsize_), nbwcells_(F.nbwcells_), nbhcells_(F.nbhcells_), nbwcells_s_(F.nbwcells_s_), nbhcells_s_(F.nbhcells_s_), ngridcells_s_(F.ngridcells_s_), nbwcells_m_(F.nbwcells_m_), nbhcells_m_(F.nbhcells_m_), ngridcells_m_(F.ngridcells_m_), nbkps_(F.nbkps_), nb2dkps_(F.nb2dkps_), nb3dkps_(F.nb3dkps_),
          nb_stereo_kps_(F.nb_stereo_kps_), Twc_(F.Twc_), Tcw_(F.Tcw_), pcalib_leftcam_(F.pcalib_leftcam_),
          pcalib_rightcam_(F.pcalib_rightcam_), pcalib_leftcam_m_(F.pcalib_leftcam_m_), pcalib_rightcam_m_(F.pcalib_rightcam_m_), pcalib_leftcam_s_(F.pcalib_leftcam_s_), pcalib_rightcam_s_(F.pcalib_rightcam_s_), Frl_(F.Frl_), Fcv_(F.Fcv_), Frl_s_(F.Frl_s_), Fcv_s_(F.Fcv_s_), map_covkfs_(F.map_covkfs_), set_local_mapids_(F.set_local_mapids_)
{}

// Set the image time and id
void Frame::updateFrame(const int id, const double time) 
{
    id_= id;
    img_time_ = time;
}

// Return vector of keypoint objects
std::vector<Keypoint> Frame::getKeypoints() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {
        v.push_back(kp.second);
    }
    return v;
}


// Return vector of 2D keypoint objects
std::vector<Keypoint> Frame::getKeypoints2d() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> v;
    v.reserve(nb2dkps_);
    for( const auto & kp : mapkps_ ) {
        if( !kp.second.is3d_ ) {
            v.push_back(kp.second);
        }
    }
    return v;
}
std::vector<Keypoint> Frame::getKeypoints2d_l() const   //只保留只在左目看到的点
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> v;
    v.reserve(nb2dkps_);
    for( const auto & kp : mapkps_ ) {
        if( !kp.second.is3d_ ) {
            // todotodo 数值不一样，不应该是一半
            if( kp.second.px_.x > pcalib_rightcam_->img_w_ - pcalib_rightcam_s_->img_w_crop_) {
//            if( kp.second.px_.x > pcalib_rightcam_->img_w_ - pcalib_rightcam_s_->img_w_/2.) {
                continue;
            }
            v.push_back(kp.second);
        }
    }
    return v;
}
std::vector<Keypoint> Frame::getKeypoints2d_r() const   //只保留只在右目看到的点
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> v;
    v.reserve(nb2dkps_);
    for( const auto & kp : mapkps_ ) {
        if( !kp.second.is3d_ ) {
            // todotodo 数值不一样，不应该是一半
            if( kp.second.px_.x < pcalib_rightcam_s_->img_w_crop_) {
//            if( kp.second.px_.x < pcalib_rightcam_s_->img_w_/2.) {
                continue;
            }
            v.push_back(kp.second);
        }
    }
    return v;
}

// Return vector of 3D keypoint objects
std::vector<Keypoint> Frame::getKeypoints3d() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> v;
    v.reserve(nb3dkps_);
    for( const auto &kp : mapkps_ ) {
        if( kp.second.is3d_ ) {
            v.push_back(kp.second);
        }
    }
    return v;
}

// Return vector of stereo keypoint objects
std::vector<Keypoint> Frame::getKeypointsStereo() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> v;
    v.reserve(nb_stereo_kps_);
    for( const auto &kp : mapkps_ ) {
        if( kp.second.is_stereo_ ) {
            v.push_back(kp.second);
        }
    }
    return v;
}

// Return vector of keypoints' raw pixel positions
std::vector<cv::Point2f> Frame::getKeypointsPx() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<cv::Point2f> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {
        v.push_back(kp.second.px_);
    }
    return v;
}

// Return vector of keypoints' undistorted pixel positions
std::vector<cv::Point2f> Frame::getKeypointsUnPx() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<cv::Point2f> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {
        v.push_back(kp.second.unpx_);
    }
    return v;
}

// Return vector of keypoints' bearing vectors
std::vector<Eigen::Vector3d> Frame::getKeypointsBv() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Eigen::Vector3d> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {
        v.push_back(kp.second.bv_);
    }
    return v;
}

// Return vector of keypoints' related landmarks' id
std::vector<int> Frame::getKeypointsId() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<int> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {
        v.push_back(kp.first);
    }
    return v;
}

Keypoint Frame::getKeypointById(const int lmid) const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return Keypoint();
    }

    return it->second;
}


std::vector<Keypoint> Frame::getKeypointsByIds(const std::vector<int> &vlmids) const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> vkp;
    vkp.reserve(vlmids.size());
    for( const auto &lmid : vlmids ) {
        auto it = mapkps_.find(lmid);
        if( it != mapkps_.end() ) {
            vkp.push_back(it->second);
        }
    }

    return vkp;
}


// Return vector of keypoints' descriptor
std::vector<cv::Mat> Frame::getKeypointsDesc() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<cv::Mat> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {
        v.push_back(kp.second.desc_);
    }

    return v;
}


// Compute keypoint from raw pixel position
inline void Frame::computeKeypoint(const cv::Point2f &pt, Keypoint &kp)
{
    kp.px_ = pt;
    kp.unpx_ = pcalib_leftcam_->undistortImagePoint(pt);

    Eigen::Vector3d hunpx(kp.unpx_.x, kp.unpx_.y, 1.);
    kp.bv_ = pcalib_leftcam_->iK_ * hunpx;
    kp.bv_.normalize();
}

inline void Frame::computeKeypoint_s(const cv::Point2f &pt, Keypoint &kp)
{
    kp.px_ = pt;
    kp.unpx_ = pcalib_leftcam_s_->undistortImagePoint(pt);//

    Eigen::Vector3d hunpx(kp.unpx_.x, kp.unpx_.y, 1.);
    kp.bv_ = pcalib_leftcam_s_->iK_ * hunpx;//
    kp.bv_.normalize();
}

// Create keypoint from raw pixel position
inline Keypoint Frame::computeKeypoint(const cv::Point2f &pt, const int lmid)
{
    Keypoint kp;
    kp.lmid_ = lmid;
    computeKeypoint(pt,kp);
    return kp;
}

inline Keypoint Frame::computeKeypoint_s(const cv::Point2f &pt, const int lmid)
{
    Keypoint kp;
    kp.lmid_ = lmid;
    computeKeypoint_s(pt,kp);
    return kp;
}


// Add keypoint object to vector of kps
void Frame::addKeypoint(const Keypoint &kp)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    if( mapkps_.count(kp.lmid_) ) {
        std::cout << "\nWEIRD!  Trying to add a KP with an already existing lmid... Not gonna do it!\n";
        return;
    }

    mapkps_.emplace(kp.lmid_, kp);
    addKeypointToGrid(kp);

    nbkps_++;
    if( kp.is3d_ ) {
        nb3dkps_++;
    } else {
        nb2dkps_++;
    }
}

void Frame::addKeypoint_s(const Keypoint &kp)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    if( mapkps_.count(kp.lmid_) ) {
        std::cout << "\nWEIRD!  Trying to add a KP with an already existing lmid... Not gonna do it!\n";
        return;
    }

    mapkps_.emplace(kp.lmid_, kp);
    addKeypointToGrid_s(kp);

    nbkps_++;
    if( kp.is3d_ ) {
        nb3dkps_++;
    } else {
        nb2dkps_++;
    }
}

// Add new keypoint from raw pixel position
void Frame::addKeypoint(const cv::Point2f &pt, const int lmid)
{
    Keypoint kp = computeKeypoint(pt, lmid);//内部用的左目参数，但左右目参数一样

    addKeypoint(kp);
}

void Frame::addKeypoint_s(const cv::Point2f &pt, const int lmid)
{
    Keypoint kp = computeKeypoint_s(pt, lmid);//内部用的左目参数，但左右目参数一样

    addKeypoint_s(kp);
}

// Add new keypoint w. desc
void Frame::addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc) 
{
    Keypoint kp = computeKeypoint(pt, lmid);
    kp.desc_ = desc;

    addKeypoint(kp);
}

void Frame::addKeypoint_s(const cv::Point2f &pt, const int lmid, const cv::Mat &desc)
{
    Keypoint kp = computeKeypoint_s(pt, lmid);//
    kp.desc_ = desc;

    addKeypoint_s(kp);//
}

// Add new keypoint w. desc & scale
void Frame::addKeypoint(const cv::Point2f &pt, const int lmid, const int scale)
{
    Keypoint kp = computeKeypoint(pt, lmid);
    kp.scale_ = scale;

    addKeypoint(kp);
}

// Add new keypoint w. desc & scale
void Frame::addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc, const int scale)
{
    Keypoint kp = computeKeypoint(pt, lmid);
    kp.desc_ = desc;
    kp.scale_ = scale;

    addKeypoint(kp);
}

// Add new keypoint w. desc & scale & angle
void Frame::addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc, const int scale, const float angle)
{
    Keypoint kp = computeKeypoint(pt, lmid);
    kp.desc_ = desc;
    kp.scale_ = scale;
    kp.angle_ = angle;

    addKeypoint(kp);
}

void Frame::updateKeypoint(const int lmid, const cv::Point2f &pt)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    } 

    Keypoint upkp = it->second;

    if( upkp.is_stereo_ ) {
        nb_stereo_kps_--;
        upkp.is_stereo_ = false;
    }

    computeKeypoint(pt, upkp);// 用新点pt替换旧点upkp的值，新值保存到upkp里 原函数用的左目，但是其实发现左目右目参数一样
    
    updateKeypointInGrid(it->second, upkp);
    it->second = upkp;
}

void Frame::updateKeypointDesc(const int lmid, const cv::Mat &desc)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }

    it->second.desc_ = desc;
}

void Frame::updateKeypointAngle(const int lmid, const float angle)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }

    it->second.angle_ = angle;
}

bool Frame::updateKeypointId(const int prevlmid, const int newlmid, const bool is3d)
{
    std::unique_lock<std::mutex> lock(kps_mutex_);

    if( mapkps_.count(newlmid) ) {
        return false;
    }

    auto it = mapkps_.find(prevlmid);
    if( it == mapkps_.end() ) {
        return false;
    }

    Keypoint upkp = it->second;
    lock.unlock();
    upkp.lmid_ = newlmid;
    upkp.is_retracked_ = true;
    upkp.is3d_ = is3d;
    removeKeypointById(prevlmid);
    addKeypoint(upkp);

    return true;
}

// Compute stereo keypoint from raw pixel position
void Frame::computeStereoKeypoint(const cv::Point2f &pt, Keypoint &kp)
{
    kp.rpx_ = pt;
    kp.runpx_ = pcalib_rightcam_->undistortImagePoint(pt);

    Eigen::Vector3d bv(kp.runpx_.x, kp.runpx_.y, 1.);
    bv = pcalib_rightcam_->iK_ * bv.eval();
    bv.normalize();

    kp.rbv_ = bv;

    if( !kp.is_stereo_ ) {
        kp.is_stereo_ = true;
        nb_stereo_kps_++;
    }
}

void Frame::computeStereoKeypoint_s(const cv::Point2f &pt, Keypoint &kp)    //现在存的是双目区点坐标
{
    kp.rpx_ = pt;
    kp.runpx_ = pcalib_rightcam_s_->undistortImagePoint(pt);

    Eigen::Vector3d bv(kp.runpx_.x, kp.runpx_.y, 1.);
    bv = pcalib_rightcam_s_->iK_ * bv.eval();
    bv.normalize();

    kp.rbv_ = bv;

    if( !kp.is_stereo_ ) {
        kp.is_stereo_ = true;
        nb_stereo_kps_++;
    }
}

void Frame::updateKeypointStereo(const int lmid, const cv::Point2f &pt)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }

    computeStereoKeypoint(pt, it->second);
}

void Frame::updateKeypointStereo_s(const int lmid, const cv::Point2f &pt)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }

    computeStereoKeypoint_s(pt, it->second);
}

inline void Frame::removeKeypoint(const Keypoint &kp)
{
    removeKeypointById(kp.lmid_);
}

void Frame::removeKeypointById(const int lmid)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }

    removeKeypointFromGrid(it->second);

    if( it->second.is3d_ ) {
        nb3dkps_--;
    } else {
        nb2dkps_--;
    }
    nbkps_--;
    if( it->second.is_stereo_ ) {
        nb_stereo_kps_--;
    }
    mapkps_.erase(lmid);
}

void Frame::removeKeypointById_s(const int lmid)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }

    removeKeypointFromGrid_s(it->second);//mono_stereo

    if( it->second.is3d_ ) {
        nb3dkps_--;
    } else {
        nb2dkps_--;
    }
    nbkps_--;
    if( it->second.is_stereo_ ) {
        nb_stereo_kps_--;
    }
    mapkps_.erase(lmid);
}


inline void Frame::removeStereoKeypoint(const Keypoint &kp)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    removeStereoKeypointById(kp.lmid_);
}

void Frame::removeStereoKeypointById(const int lmid)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }
    
    if( it->second.is_stereo_ ) {
        it->second.is_stereo_ = false;
        nb_stereo_kps_--;
    }
}

void Frame::turnKeypoint3d(const int lmid)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }

    if( !it->second.is3d_ ) {
        it->second.is3d_ = true;
        nb3dkps_++;
        nb2dkps_--;
    }
}

bool Frame::isObservingKp(const int lmid) const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);//开始时经常闪退
//    std::unique_lock<std::mutex> lock(kps_mutex_);//改成这个闪退少了，但还是有闪退
    return mapkps_.count(lmid);
}

void Frame::addKeypointToGrid(const Keypoint &kp)
{
    std::lock_guard<std::mutex> lock(grid_mutex_);

    int idx = getKeypointCellIdx(kp.px_);

    if( vgridkps_.at(idx).empty() ) {
        noccupcells_++;
    }

    vgridkps_.at(idx).push_back(kp.lmid_);
}

void Frame::addKeypointToGrid_s(const Keypoint &kp)
{
    std::lock_guard<std::mutex> lock(grid_mutex_);

    int idx = getKeypointCellIdx_s(kp.px_);//

    if( vgridkps_s_.at(idx).empty() ) {//
        noccupcells_++;
    }

    vgridkps_s_.at(idx).push_back(kp.lmid_);//
}

void Frame::removeKeypointFromGrid(const Keypoint &kp)
{
    std::lock_guard<std::mutex> lock(grid_mutex_);

    int idx = getKeypointCellIdx(kp.px_);

    if( idx < 0 || idx >= (int)vgridkps_.size() ) {
        return;
    }

    for( size_t i = 0, iend = vgridkps_.at(idx).size() ; i < iend ; i++ )
    {
        if( vgridkps_.at(idx).at(i) == kp.lmid_ ) {
            vgridkps_.at(idx).erase(vgridkps_.at(idx).begin() + i);

            if( vgridkps_.at(idx).empty() ) {
                noccupcells_--;
            }
            break;
        }
    }
}

void Frame::removeKeypointFromGrid_s(const Keypoint &kp)
{
    std::lock_guard<std::mutex> lock(grid_mutex_);

    int idx = getKeypointCellIdx_s(kp.px_);//mono_stereo

    if( idx < 0 || idx >= (int)vgridkps_s_.size() ) {
        return;
    }

    for( size_t i = 0, iend = vgridkps_s_.at(idx).size() ; i < iend ; i++ )
    {
        if( vgridkps_s_.at(idx).at(i) == kp.lmid_ ) {
            vgridkps_s_.at(idx).erase(vgridkps_s_.at(idx).begin() + i);

            if( vgridkps_s_.at(idx).empty() ) {
                noccupcells_--;
            }
            break;
        }
    }
}

void Frame::updateKeypointInGrid(const Keypoint &prevkp, const Keypoint &newkp)
{
    // First ensure that new kp should move
    int idx = getKeypointCellIdx(prevkp.px_);

    int nidx = getKeypointCellIdx(newkp.px_);

    if( idx == nidx ) {
        // Nothing to do
        return;
    }
    else {
        // First remove kp
        removeKeypointFromGrid(prevkp);
        // Second the new kp is added to the grid
        addKeypointToGrid(newkp);
    }
}

std::vector<Keypoint> Frame::getKeypointsFromGrid(const cv::Point2f &pt) const
{
    std::lock_guard<std::mutex> lock(grid_mutex_);

    std::vector<int> voutkpids;

    int idx = getKeypointCellIdx(pt);

    if( idx < 0 || idx >= (int)vgridkps_.size() ) {
        return std::vector<Keypoint>();
    }

    if( vgridkps_.at(idx).empty() ) {
        return std::vector<Keypoint>();
    }

    for( const auto &id : vgridkps_.at(idx) )
    {
        voutkpids.push_back(id);
    }

    return getKeypointsByIds(voutkpids);
}

int Frame::getKeypointCellIdx(const cv::Point2f &pt) const
{
    int r = floor(pt.y / ncellsize_);
    int c = floor(pt.x / ncellsize_);
    return (r * nbwcells_ + c);
}

int Frame::getKeypointCellIdx_s(const cv::Point2f &pt) const
{
    int r = floor(pt.y / ncellsize_);
    int c = floor(pt.x / ncellsize_);
    return (r * nbwcells_s_ + c);
}

std::vector<Keypoint> Frame::getSurroundingKeypoints(const Keypoint &kp) const
{
    std::vector<Keypoint> vkps;
    vkps.reserve(20);

    int rkp = floor(kp.px_.y / ncellsize_);
    int ckp = floor(kp.px_.x / ncellsize_);

    std::lock_guard<std::mutex> lock(kps_mutex_);
    std::lock_guard<std::mutex> glock(grid_mutex_);
    
    for( int r = rkp-1 ; r < rkp+1 ; r++ ) {
        for( int c = ckp-1 ; c < ckp+1 ; c++ ) {
            int idx = r * nbwcells_ + c;
            if( r < 0 || c < 0 || idx > (int)vgridkps_.size() ) {
                continue;
            }
            for( const auto &id : vgridkps_.at(idx) ) {
                if( id != kp.lmid_ ) {
                    auto it = mapkps_.find(id);
                    if( it != mapkps_.end() ) {
                        vkps.push_back(it->second);
                    }
                }
            }
        }
    }
    return vkps;
}

std::vector<Keypoint> Frame::getSurroundingKeypoints_s(const Keypoint &kp) const
{
    std::vector<Keypoint> vkps;
    vkps.reserve(20);

    int rkp = floor(kp.px_.y / ncellsize_);
    int ckp = floor(kp.px_.x / ncellsize_);

    std::lock_guard<std::mutex> lock(kps_mutex_);
    std::lock_guard<std::mutex> glock(grid_mutex_);

    for( int r = rkp-1 ; r < rkp+1 ; r++ ) {
        for( int c = ckp-1 ; c < ckp+1 ; c++ ) {
//            int idx = r * nbwcells_ + c;
            int idx = r * nbwcells_s_ + c;
//            if( r < 0 || c < 0 || idx > (int)vgridkps_.size() ) {
//            if( r < 0 || c < 0 || idx > (int)vgridkps_s_.size() ) {  // todo 会超索引报错
            if( r < 0 || c < 0 || idx >= (int)vgridkps_s_.size() ) {
                continue;
            }
//            for( const auto &id : vgridkps_.at(idx) ) {
            for( const auto &id : vgridkps_s_.at(idx) ) {
                if( id != kp.lmid_ ) {
                    auto it = mapkps_.find(id);
                    if( it != mapkps_.end() ) {
                        vkps.push_back(it->second);
                    }
                }
            }
        }
    }
    return vkps;
}

std::vector<Keypoint> Frame::getSurroundingKeypoints(const cv::Point2f &pt) const
{
    std::vector<Keypoint> vkps;
    vkps.reserve(20);

    int rkp = floor(pt.y / ncellsize_);
    int ckp = floor(pt.x / ncellsize_);

    std::lock_guard<std::mutex> lock(kps_mutex_);
    std::lock_guard<std::mutex> glock(grid_mutex_);
    
    for( int r = rkp-1 ; r < rkp+1 ; r++ ) {
        for( int c = ckp-1 ; c < ckp+1 ; c++ ) {
            int idx = r * nbwcells_ + c;
            if( r < 0 || c < 0 || idx > (int)vgridkps_.size() ) {
                continue;
            }
            for( const auto &id : vgridkps_.at(idx) ) {
                auto it = mapkps_.find(id);
                if( it != mapkps_.end() ) {
                    vkps.push_back(it->second);
                }
            }
        }
    }
    return vkps;
}

std::map<int,int> Frame::getCovisibleKfMap() const
{
    std::lock_guard<std::mutex> lock(cokfs_mutex_);
    return map_covkfs_;
}

inline void Frame::updateCovisibleKfMap(const std::map<int,int> &cokfs)
{
    std::lock_guard<std::mutex> lock(cokfs_mutex_);
    map_covkfs_ = cokfs;
}

void Frame::addCovisibleKf(const int kfid)
{
    if( kfid == kfid_ ) {
        return;
    }

    std::lock_guard<std::mutex> lock(cokfs_mutex_);
    auto it = map_covkfs_.find(kfid);
    if( it != map_covkfs_.end() ) {
        it->second += 1;
    } else {
        map_covkfs_.emplace(kfid, 1);
    }
}

void Frame::removeCovisibleKf(const int kfid)
{
    if( kfid == kfid_ ) {
        return;
    }

    std::lock_guard<std::mutex> lock(cokfs_mutex_);
    map_covkfs_.erase(kfid);
}

void Frame::decreaseCovisibleKf(const int kfid)
{
    if( kfid == kfid_ ) {
        return;
    }

    std::lock_guard<std::mutex> lock(cokfs_mutex_);
    auto it = map_covkfs_.find(kfid);
    if( it != map_covkfs_.end() ) {
        if( it->second != 0 ) {
            it->second -= 1;
            if( it->second == 0 ) {
                map_covkfs_.erase(it);
            }
        }
    }
}

Sophus::SE3d Frame::getTcw() const
{
    std::lock_guard<std::mutex> lock(pose_mutex_);
    return Tcw_;
}

Sophus::SE3d Frame::getTwc() const
{
    std::lock_guard<std::mutex> lock(pose_mutex_);
    return Twc_;
}

Eigen::Matrix3d Frame::getRcw() const
{
   std::lock_guard<std::mutex> lock(pose_mutex_);
   return Tcw_.rotationMatrix();
}

Eigen::Matrix3d Frame::getRwc() const
{
   std::lock_guard<std::mutex> lock(pose_mutex_);
   return Twc_.rotationMatrix();
}

Eigen::Vector3d Frame::gettcw() const
{
   std::lock_guard<std::mutex> lock(pose_mutex_);
   return Tcw_.translation();
}

Eigen::Vector3d Frame::gettwc() const
{
   std::lock_guard<std::mutex> lock(pose_mutex_);
   return Twc_.translation();
}

void Frame::setTwc(const Sophus::SE3d &Twc)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    Twc_ = Twc;
    Tcw_ = Twc.inverse();

    // 后续调用时增加平移增量 todo test fake setTwc
//    Eigen::Vector3d translation_increment(0.001, 0.001, 0.001); // 每次增加的平移量
//    Eigen::Vector3d new_translation = Twc_.translation() + translation_increment;
//    Twc_ = Sophus::SE3d(Twc_.so3(), new_translation); // 保留原有旋转，只修改平移
//    Tcw_ = Twc_.inverse(); // 更新Tcw为Twc_的逆
//    std::cout << "****** Twc_ = " << Twc_.matrix() << std::endl;

}

inline void Frame::setTcw(const Sophus::SE3d &Tcw)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    Tcw_ = Tcw;
    Twc_ = Tcw.inverse();
}

void Frame::setTwc(const Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    Twc_.setRotationMatrix(Rwc);
    Twc_.translation() = twc;

    Tcw_ = Twc_.inverse();
}


inline void Frame::setTcw(const Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    Tcw_.setRotationMatrix(Rcw);
    Tcw_.translation() = tcw;

    Twc_ = Tcw_.inverse();
}

cv::Point2f Frame::projCamToImage(const Eigen::Vector3d &pt) const
{
    return pcalib_leftcam_->projectCamToImage(pt);//左相机坐标系到左像素坐标系
}

cv::Point2f Frame::projCamToImage_s(const Eigen::Vector3d &pt) const
{
    return pcalib_leftcam_s_->projectCamToImage(pt);//左双目相机坐标系到左双目像素坐标系
}

cv::Point2f Frame::projCamToRightImage(const Eigen::Vector3d &pt) const
{
    return pcalib_rightcam_->projectCamToImage(pcalib_rightcam_->Tcic0_ * pt);//左相机坐标系到右像素坐标系 = 右相机到右图像*左相机到右相机
}

cv::Point2f Frame::projCamToRightImage_s(const Eigen::Vector3d &pt) const
{
    return pcalib_rightcam_s_->projectCamToImage(pcalib_rightcam_s_->Tcic0_ * pcalib_rightcam_->Tcic0_ * pcalib_leftcam_s_->Tc0ci_ * pt);//左双目相机坐标系到右双目像素坐标系 = 右双目相机到右双目图像*右目到右双目*左相机到右相机*左双目到左目
}

cv::Point2f Frame::projCamToImageDist(const Eigen::Vector3d &pt) const
{
    return pcalib_leftcam_->projectCamToImageDist(pt);
}


cv::Point2f Frame::projCamToRightImageDist(const Eigen::Vector3d &pt) const
{
    return pcalib_rightcam_->projectCamToImageDist(pcalib_rightcam_->Tcic0_ * pt);  // 相机坐标系到右目图像坐标系 = 右目到右目图像（左目到右目）
}

cv::Point2f Frame::projCamToRightImageDist_s(const Eigen::Vector3d &pt) const
{
    return pcalib_rightcam_s_->projectCamToImageDist(pcalib_rightcam_s_->Tcic0_ * pcalib_rightcam_->Tcic0_ * pt);  // 左相机坐标系到右双目图像坐标系 = 右双目到右双目图像（右目到右双目*左目到右目）
}

cv::Point2f Frame::projCamToImageDist_s(const Eigen::Vector3d &pt) const
{
    return pcalib_leftcam_s_->projectCamToImageDist(pcalib_leftcam_s_->Tcic0_ * pcalib_rightcam_->Tc0ci_ * pt);  // 右相机坐标系到左双目图像坐标系 = 左双目到左双目图像（左目到左双目*右目到左目）
}


Eigen::Vector3d Frame::projCamToWorld(const Eigen::Vector3d &pt) const
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    Eigen::Vector3d wpt = Twc_ * pt;//左目到世界 （每一个帧的Twc_存的都是原算法的左目的位姿）

    return wpt;
}

Eigen::Vector3d Frame::projCamToWorld_s(const Eigen::Vector3d &pt) const
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    Eigen::Vector3d wpt = Twc_ * pcalib_leftcam_s_->Tc0ci_ * pt;//左双目到世界=左目到世界*左双目到左目

    return wpt;
}

Eigen::Vector3d Frame::projWorldToCam(const Eigen::Vector3d &pt) const
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    Eigen::Vector3d campt = Tcw_ * pt;

    return campt;
}

Eigen::Vector3d Frame::projWorldToCam_s(const Eigen::Vector3d &pt) const
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

//    Eigen::Vector3d campt = Tcw_ * pt;

    Eigen::Vector3d campt = pcalib_leftcam_s_->Tcic0_ * Tcw_ * pt;// 左到左双目*世界到左目

    return campt;
}

Eigen::Vector3d Frame::projWorldToCam_right(const Eigen::Vector3d &pt) const
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

//    Eigen::Vector3d campt = Tcw_ * pt;// mono_stereo

    // 将世界坐标系中的点转换到左相机坐标系
    Eigen::Vector3d campt_left = Tcw_ * pt;

    // 将左相机坐标系的点转换到右相机坐标系
//    std::cout<<"2.0 Trl_ = "<<Trl_.matrix()<<std::endl;//一直是单位矩阵
//    Eigen::Vector3d campt_right = Trl_ * campt_left;
    Eigen::Vector3d campt_right = pcalib_rightcam_->Tcic0_ * campt_left;

    return campt_right;
}

cv::Point2f Frame::projWorldToImage(const Eigen::Vector3d &pt) const
{
    return pcalib_leftcam_->projectCamToImage(projWorldToCam(pt));
}

cv::Point2f Frame::projWorldToImageDist(const Eigen::Vector3d &pt) const
{
    return pcalib_leftcam_->projectCamToImageDist(projWorldToCam(pt));
}

cv::Point2f Frame::projWorldToImageDist_right(const Eigen::Vector3d &pt) const  //添加右目投影
{
    return pcalib_rightcam_->projectCamToImageDist(projWorldToCam_right(pt));//右目 todo 用projWorldToCam_right还是projWorldToCam？取决于每个相机模型存储的Tcw是各自相机的位姿，还是都是左目的位姿
}

cv::Point2f Frame::projWorldToRightImage(const Eigen::Vector3d &pt) const
{
    return pcalib_rightcam_->projectCamToImage(pcalib_rightcam_->Tcic0_ * projWorldToCam(pt));
}

cv::Point2f Frame::projWorldToRightImageDist(const Eigen::Vector3d &pt) const
{
    return pcalib_rightcam_->projectCamToImageDist(pcalib_rightcam_->Tcic0_ * projWorldToCam(pt));// 世界到右目=左目到右目*世界到左目*世界
}

cv::Point2f Frame::projWorldToRightImageDist_s(const Eigen::Vector3d &pt) const // mono_stereo
{
    return pcalib_rightcam_s_->projectCamToImageDist(pcalib_rightcam_s_->Tcic0_ * pcalib_rightcam_->Tcic0_ * projWorldToCam(pt));// 世界到右双目=右目到右双目*左目到右目*世界到左目*世界
}

cv::Point2f Frame::projWorldToImageDist_s(const Eigen::Vector3d &pt) const // mono_stereo
{
    return pcalib_rightcam_s_->projectCamToImageDist(pcalib_leftcam_s_->Tcic0_ * pcalib_rightcam_->Tc0ci_ * projWorldToCam(pt));// 世界到左双目=左目到左双目*右目到左目*世界到右目*世界
}

bool Frame::isInImage(const cv::Point2f &pt) const
{
    return (pt.x >= 0 && pt.y >= 0 && pt.x < pcalib_leftcam_->img_w_ && pt.y < pcalib_leftcam_->img_h_);
}

bool Frame::isInRightImage(const cv::Point2f &pt) const
{
    return (pt.x >= 0 && pt.y >= 0 && pt.x < pcalib_rightcam_->img_w_ && pt.y < pcalib_rightcam_->img_h_);
}

bool Frame::isInRightImage_s(const cv::Point2f &pt) const   //mono_stereo
{
    return (pt.x >= 0 && pt.y >= 0 && pt.x < pcalib_rightcam_s_->img_w_ && pt.y < pcalib_rightcam_s_->img_h_);
}
bool Frame::isInRightImage_s2_l(const cv::Point2f &pt) const   //mono_stereo 三角化选择左半张图
{
    return (pt.x >= 0 && pt.y >= 0 && pt.x < pcalib_rightcam_s_->img_w_/2. && pt.y < pcalib_rightcam_s_->img_h_);
}
bool Frame::isInRightImage_s2_r(const cv::Point2f &pt) const   //mono_stereo 三角化选择右半张图
{
    return (pt.x >= pcalib_rightcam_s_->img_w_/2. && pt.y >= 0 && pt.x < pcalib_rightcam_s_->img_w_ && pt.y < pcalib_rightcam_s_->img_h_);
}

void Frame::displayFrameInfo()
{
    std::cout << "\n************************************";
    std::cout << "\nFrame #" << id_ << " (KF #" << kfid_ << ") info:\n";
    std::cout << "\n> Nb kps all (2d / 3d / stereo) : " << nbkps_ << " (" << nb2dkps_ << " / " << nb3dkps_ << " / " << nb_stereo_kps_ << ")";
    std::cout << "\n> Nb covisible kfs : " << map_covkfs_.size();
    std::cout << "\n twc : " << Twc_.translation().transpose();
    std::cout << "\n************************************\n\n";
}

void Frame::reset()
{
    id_ = -1;
    kfid_ = 0;
    img_time_ = 0.;

    std::lock_guard<std::mutex> lock(kps_mutex_);
    std::lock_guard<std::mutex> lock2(grid_mutex_);
    
    mapkps_.clear();
    vgridkps_.clear();
    vgridkps_.resize( ngridcells_ );
    vgridkps_m_.clear();
    vgridkps_m_.resize( ngridcells_m_ );
    vgridkps_s_.clear();
    vgridkps_s_.resize( ngridcells_s_ );

    nbkps_ = 0;
    nb2dkps_ = 0;
    nb3dkps_ = 0;
    nb_stereo_kps_ = 0;

    noccupcells_ = 0;

    Twc_ = Sophus::SE3d();
    Tcw_ = Sophus::SE3d();

    map_covkfs_.clear();
    set_local_mapids_.clear();
}