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

#include "feature_tracker.hpp"

#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

#include "multi_view_geometry.hpp"

void FeatureTracker::fbKltTracking(const std::vector<cv::Mat> &vprevpyr, const std::vector<cv::Mat> &vcurpyr, 
        int nwinsize, int nbpyrlvl, float ferr, float fmax_fbklt_dist, std::vector<cv::Point2f> &vkps, 
        std::vector<cv::Point2f> &vpriorkps, std::vector<bool> &vkpstatus) const
{
    // std::cout << "\n \t >>> Forward-Backward kltTracking with Pyr of Images and Motion Prior! \n";

    assert(vprevpyr.size() == vcurpyr.size());

    if( vkps.empty() ) {
        // std::cout << "\n \t >>> No kps were provided to kltTracking()!\n";
        return;
    }

    cv::Size klt_win_size(nwinsize, nwinsize);

    if( (int)vprevpyr.size() < 2*(nbpyrlvl+1) ) {
        nbpyrlvl = vprevpyr.size() / 2 - 1;
    }

    // Objects for OpenCV KLT
    size_t nbkps = vkps.size();
    vkpstatus.reserve(nbkps);

    std::vector<uchar> vstatus;
    std::vector<float> verr;
    std::vector<int> vkpsidx;
    vstatus.reserve(nbkps);
    verr.reserve(nbkps);
    vkpsidx.reserve(nbkps);

    // Tracking Forward
    cv::calcOpticalFlowPyrLK(vprevpyr, vcurpyr, vkps, vpriorkps, 
                vstatus, verr, klt_win_size,  nbpyrlvl, klt_convg_crit_, 
                (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS) 
                );

    std::vector<cv::Point2f> vnewkps;
    std::vector<cv::Point2f> vbackkps;
    vnewkps.reserve(nbkps);
    vbackkps.reserve(nbkps);

    size_t nbgood = 0;

    // Init outliers vector & update tracked kps
    for( size_t i = 0 ; i < nbkps ; i++ ) 
    {
        if( !vstatus.at(i) ) {
            vkpstatus.push_back(false);
            continue;
        }

        if( verr.at(i) > ferr ) {
            vkpstatus.push_back(false);
            continue;
        }

        if( !inBorder(vpriorkps.at(i), vcurpyr.at(0)) ) {
            vkpstatus.push_back(false);
            continue;
        }

        vnewkps.push_back(vpriorkps.at(i));
        vbackkps.push_back(vkps.at(i));
        vkpstatus.push_back(true);
        vkpsidx.push_back(i);
        nbgood++;
    }  

    if( vnewkps.empty() ) {
        return;
    }
    
    vstatus.clear();
    verr.clear();

    // std::cout << "\n \t >>> Forward kltTracking : #" << nbgood << " out of #" << nbkps << " \n";

    // Tracking Backward
    cv::calcOpticalFlowPyrLK(vcurpyr, vprevpyr, vnewkps, vbackkps, 
                vstatus, verr, klt_win_size,  0, klt_convg_crit_,
                (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS) 
                );
    
    nbgood = 0;
    for( int i = 0, iend=vnewkps.size() ; i < iend ; i++ )
    {
        int idx = vkpsidx.at(i);

        if( !vstatus.at(i) ) {
            vkpstatus.at(idx) = false;
            continue;
        }

        if( cv::norm(vkps.at(idx) - vbackkps.at(i)) > fmax_fbklt_dist ) {
            vkpstatus.at(idx) = false;
            continue;
        }

        nbgood++;
    }

    // std::cout << "\n \t >>> Backward kltTracking : #" << nbgood << " out of #" << vkpsidx.size() << " \n";
}

void FeatureTracker::fbKltTracking2(const std::vector<cv::Mat> &vprevpyr, const std::vector<cv::Mat> &vcurpyr,
                                    int nwinsize, int nbpyrlvl, float ferr, float fmax_fbklt_dist, std::vector<cv::Point2f> &vkps,
                                    std::vector<cv::Point2f> &vpriorkps, std::vector<bool> &vkpstatus, bool isleft) const
{
    // std::cout << "\n \t >>> Forward-Backward kltTracking with Pyr of Images and Motion Prior! \n";

    assert(vprevpyr.size() == vcurpyr.size());

    if( vkps.empty() ) {
        // std::cout << "\n \t >>> No kps were provided to kltTracking()!\n";
        return;
    }

    cv::Size klt_win_size(nwinsize, nwinsize);

    if( (int)vprevpyr.size() < 2*(nbpyrlvl+1) ) {
        nbpyrlvl = vprevpyr.size() / 2 - 1;
    }

    // Objects for OpenCV KLT
    size_t nbkps = vkps.size();
    vkpstatus.reserve(nbkps);

    std::vector<uchar> vstatus;
    std::vector<float> verr;
    std::vector<int> vkpsidx;
    vstatus.reserve(nbkps);
    verr.reserve(nbkps);
    vkpsidx.reserve(nbkps);

    // Tracking Forward
    cv::calcOpticalFlowPyrLK(vprevpyr, vcurpyr, vkps, vpriorkps,
                             vstatus, verr, klt_win_size,  nbpyrlvl, klt_convg_crit_,
                             (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS)
    );
    // 可视化前向跟踪的结果
    if(isleft){
        cv::Mat forward_tracking_visual = vcurpyr[0].clone();
        if (forward_tracking_visual.channels() == 1)
            cv::cvtColor(forward_tracking_visual, forward_tracking_visual, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < vkps.size(); ++i) {
            if (vstatus[i] && verr[i] <= ferr) {
                cv::circle(forward_tracking_visual, vkps[i], 3, cv::Scalar(0, 255, 0), -1);  // 绿色表示成功跟踪的点
                cv::line(forward_tracking_visual, vkps[i], vpriorkps[i], cv::Scalar(255, 0, 0), 1);  // 蓝色线条表示光流
            }
        }
//        cv::imshow("left Forward Tracking", forward_tracking_visual);
//        cv::waitKey(1);
    } else {
        cv::Mat forward_tracking_visual = vcurpyr[0].clone();
        if (forward_tracking_visual.channels() == 1)
            cv::cvtColor(forward_tracking_visual, forward_tracking_visual, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < vkps.size(); ++i) {
            if (vstatus[i] && verr[i] <= ferr) {
                cv::circle(forward_tracking_visual, vkps[i], 3, cv::Scalar(0, 255, 0), -1);  // 绿色表示成功跟踪���点
                cv::line(forward_tracking_visual, vkps[i], vpriorkps[i], cv::Scalar(255, 0, 0), 1);  // 蓝色线条表示光流
            }
        }
//        cv::imshow("right Forward Tracking", forward_tracking_visual);
//        cv::waitKey(1);
    }

    std::vector<cv::Point2f> vnewkps;
    std::vector<cv::Point2f> vbackkps;
    vnewkps.reserve(nbkps);
    vbackkps.reserve(nbkps);

    size_t nbgood = 0;

    // Init outliers vector & update tracked kps
    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        if( !vstatus.at(i) ) {
            vkpstatus.push_back(false);
            continue;
        }

        if( verr.at(i) > ferr ) {
            vkpstatus.push_back(false);
            continue;
        }

        if( !inBorder(vpriorkps.at(i), vcurpyr.at(0)) ) {
            vkpstatus.push_back(false);
            continue;
        }

        vnewkps.push_back(vpriorkps.at(i));
        vbackkps.push_back(vkps.at(i));
        vkpstatus.push_back(true);
        vkpsidx.push_back(i);
        nbgood++;
    }

    if( vnewkps.empty() ) {
        return;
    }

    vstatus.clear();
    verr.clear();

    // std::cout << "\n \t >>> Forward kltTracking : #" << nbgood << " out of #" << nbkps << " \n";

    // Tracking Backward
    cv::calcOpticalFlowPyrLK(vcurpyr, vprevpyr, vnewkps, vbackkps,
                             vstatus, verr, klt_win_size,  0, klt_convg_crit_,
                             (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS)
    );
    // 可视化后向跟踪的结果
    if(isleft){
        cv::Mat backward_tracking_visual = vprevpyr[0].clone();
        if (backward_tracking_visual.channels() == 1)
            cv::cvtColor(backward_tracking_visual, backward_tracking_visual, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < vnewkps.size(); ++i) {
            if (vstatus[i] && cv::norm(vkps[vkpsidx[i]] - vbackkps[i]) <= fmax_fbklt_dist) {
                cv::circle(backward_tracking_visual, vbackkps[i], 3, cv::Scalar(0, 255, 0), -1);  // 绿色表示成功跟踪的点
                cv::line(backward_tracking_visual, vnewkps[i], vbackkps[i], cv::Scalar(0, 0, 255), 1);  // 红色线条表示光流
            }
        }
//        cv::imshow("left Backward Tracking", backward_tracking_visual);
//        cv::waitKey(1);
    } else {
        cv::Mat backward_tracking_visual = vprevpyr[0].clone();
        if (backward_tracking_visual.channels() == 1)
            cv::cvtColor(backward_tracking_visual, backward_tracking_visual, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < vnewkps.size(); ++i) {
            if (vstatus[i] && cv::norm(vkps[vkpsidx[i]] - vbackkps[i]) <= fmax_fbklt_dist) {
                cv::circle(backward_tracking_visual, vbackkps[i], 3, cv::Scalar(0, 255, 0), -1);  // 绿色表示成功跟踪的点
                cv::line(backward_tracking_visual, vnewkps[i], vbackkps[i], cv::Scalar(0, 0, 255), 1);  // 红色线条表示光流
            }
        }
//        cv::imshow("right Backward Tracking", backward_tracking_visual);
//        cv::waitKey(1);
    }

    nbgood = 0;

/////// RANSAC filtering
//    std::vector<cv::Point2f> inliers_prev, inliers_curr;
//    // Filter outliers based on forward-backward error
//    for (int i = 0, iend = vnewkps.size(); i < iend; i++) {
//        int idx = vkpsidx.at(i);
//
//        if (!vstatus.at(i)) {
//            vkpstatus.at(idx) = false;
//            continue;
//        }
//
//        if (cv::norm(vkps.at(idx) - vbackkps.at(i)) > fmax_fbklt_dist) {
//            vkpstatus.at(idx) = false;
//            continue;
//        }
//
//        inliers_prev.push_back(vkps.at(idx));
//        inliers_curr.push_back(vpriorkps.at(idx));
//        vkpstatus.at(idx) = true;
//        nbgood++;
//    }
//
//     // RANSAC filtering
//    if (inliers_prev.size() >= 8) {
//        std::vector<uchar> ransac_status;
//        cv::Mat fundamental_matrix = cv::findFundamentalMat(
//                inliers_prev, inliers_curr, cv::FM_RANSAC, 3.0, 0.99, ransac_status);
//
//        size_t inlier_count = 0;
//        for (size_t i = 0; i < ransac_status.size(); ++i) {
//            if (ransac_status[i]) {
//                inlier_count++;
//            } else {
//                vkpstatus[i] = false;  // Mark as outlier
//            }
//        }
//        std::cout << "\n >>> RANSAC Filtering: " << inlier_count << " inliers retained out of "
//                  << inliers_prev.size() << " matches.\n";
//
//        // 可视化 RANSAC 筛选后的结果
//        cv::Mat ransac_visual = vcurpyr[0].clone();
//        if (ransac_visual.channels() == 1)
//            cv::cvtColor(ransac_visual, ransac_visual, cv::COLOR_GRAY2BGR);
//
//        for (size_t i = 0; i < inliers_prev.size(); ++i) {
//            if (ransac_status[i]) {
//                cv::circle(ransac_visual, inliers_curr[i], 3, cv::Scalar(0, 255, 0), -1); // 内点
//                cv::line(ransac_visual, inliers_prev[i], inliers_curr[i], cv::Scalar(255, 0, 0), 1); // 光流线
//            } else {
//                cv::circle(ransac_visual, inliers_curr[i], 3, cv::Scalar(0, 0, 255), -1); // 外点
//            }
//        }
//        cv::imshow("RANSAC Matches", ransac_visual);
//        cv::waitKey(1);
//    }

///////////

    // add Filtering based on distance threshold (image size / 2)
    cv::Size image_size = vcurpyr[0].size();  // Get image size (use first pyramid level)
    float max_dist_threshold_x = image_size.width * 0.5; // half of the image size
    float max_dist_threshold_y = image_size.height * 0.5; // half of the image size

    for( int i = 0, iend=vnewkps.size() ; i < iend ; i++ )
    {
        int idx = vkpsidx.at(i);

        if( !vstatus.at(i) ) {
            vkpstatus.at(idx) = false;
            continue;
        }

        // Compute the distance between the original point and the backward-tracked point
        float dist_x = abs(vkps.at(idx).x - vbackkps.at(i).x);
        float dist_y = abs(vkps.at(idx).y - vbackkps.at(i).y);

//        if( cv::norm(vkps.at(idx) - vbackkps.at(i)) > fmax_fbklt_dist ) {
        if( dist_x > max_dist_threshold_x || dist_y > max_dist_threshold_y || cv::norm(vkps.at(idx) - vbackkps.at(i)) > fmax_fbklt_dist ) {
            vkpstatus.at(idx) = false;
            continue;
        }

        nbgood++;
    }

//     std::cout << "\n \t >>> Backward kltTracking : #" << nbgood << " out of #" << vkpsidx.size() << " \n";
}

void FeatureTracker::getLineMinSAD(const cv::Mat &iml, const cv::Mat &imr, 
    const cv::Point2f &pt,  const int nwinsize, float &xprior, 
    float &l1err, bool bgoleft) const
{
    xprior = -1;

    if( nwinsize % 2 == 0 ) {
        std::cerr << "\ngetLineMinSAD requires an odd window size\n";
        return;
    }

    const float x = pt.x;
    const float y = pt.y;
    int halfwin = nwinsize / 2;

    if( x - halfwin < 0 ) 
        halfwin += (x-halfwin);
    if( x + halfwin >= imr.cols )
        halfwin += (x+halfwin - imr.cols - 1);
    if( y - halfwin < 0 )
        halfwin += (y-halfwin);
    if( y + halfwin >= imr.rows )
        halfwin += (y+halfwin - imr.rows - 1);
    
    if( halfwin <= 0 ) {
        return;
    }

    cv::Size winsize(2 * halfwin + 1, 2 * halfwin + 1);

    int nbwinpx = (winsize.width * winsize.height);

    float minsad = 255.;
    // int minxpx = -1;

    cv::Mat patch, target;

    cv::getRectSubPix(iml, winsize, pt, patch);

    if( bgoleft ) {
        for( float c = x ; c >= halfwin ; c-=1. )
        {
            cv::getRectSubPix(imr, winsize, cv::Point2f(c, y), target);
            l1err = cv::norm(patch, target, cv::NORM_L1);
            l1err /= nbwinpx;

            if( l1err < minsad ) {
                minsad = l1err;
                xprior = c;
            }
        }
    } else {
        for( float c = x ; c < imr.cols - halfwin ; c+=1. )
        {
            cv::getRectSubPix(imr, winsize, cv::Point2f(c, y), target);
            l1err = cv::norm(patch, target, cv::NORM_L1);
            l1err /= nbwinpx;

            if( l1err < minsad ) {
                minsad = l1err;
                xprior = c;
            }
        }
    }

    l1err = minsad;
}



/**
 * \brief Perform a forward-backward calcOpticalFlowPyrLK() tracking with OpenCV.
 *
 * \param[in] pt  Opencv 2D point.
 * \return True if pt is within image borders, False otherwise
 */
bool FeatureTracker::inBorder(const cv::Point2f &pt, const cv::Mat &im) const
{
    const float BORDER_SIZE = 1.;

    return BORDER_SIZE <= pt.x && pt.x < im.cols - BORDER_SIZE && BORDER_SIZE <= pt.y && pt.y < im.rows - BORDER_SIZE;
}
