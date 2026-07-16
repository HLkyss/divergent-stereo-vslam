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

#include "mapper.hpp"
#include "opencv2/video/tracking.hpp"

Mapper::Mapper(std::shared_ptr<SlamParams> pslamstate, std::shared_ptr<MapManager> pmap, 
            std::shared_ptr<Frame> pframe)
    : pslamstate_(pslamstate), pmap_(pmap), pcurframe_(pframe)
    , pestimator_( new Estimator(pslamstate_, pmap_, pmap_l_, pmap_r_) )
    , ploopcloser_( new LoopCloser(pslamstate_, pmap_) )
{
    std::thread mapper_thread(&Mapper::run, this);
    mapper_thread.detach();

    std::cout << "\nMapper Object is created!\n";
}

Mapper::Mapper(std::shared_ptr<SlamParams> pslamstate, std::shared_ptr<MapManager> pmap, std::shared_ptr<MapManager> pmap_l, std::shared_ptr<MapManager> pmap_r, std::shared_ptr<MapManager> pmap_lm, std::shared_ptr<MapManager> pmap_ls, std::shared_ptr<MapManager> pmap_rm, std::shared_ptr<MapManager> pmap_rs, std::shared_ptr<MapManager> pmap_l_to_r, std::shared_ptr<MapManager> pmap_r_to_l,    //mono_stereo
               std::shared_ptr<Frame> pframe, std::shared_ptr<Frame> pframe_l, std::shared_ptr<Frame> pframe_r, std::shared_ptr<Frame> pframe_lm, std::shared_ptr<Frame> pframe_ls, std::shared_ptr<Frame> pframe_rm, std::shared_ptr<Frame> pframe_rs)
        : pslamstate_(pslamstate), pmap_(pmap), pmap_l_(pmap_l), pmap_r_(pmap_r), pmap_lm_(pmap_lm), pmap_ls_(pmap_ls), pmap_rm_(pmap_rm), pmap_rs_(pmap_rs), pmap_l_to_r_(pmap_l_to_r), pmap_r_to_l_(pmap_r_to_l),pcurframe_(pframe), pcurframe_l_(pframe_l), pcurframe_r_(pframe_r), pcurframe_lm_(pframe_lm), pcurframe_ls_(pframe_ls), pcurframe_rm_(pframe_rm), pcurframe_rs_(pframe_rs)
        , pestimator_( new Estimator(pslamstate_, pmap_, pmap_l_, pmap_r_) )
        , ploopcloser_( new LoopCloser(pslamstate_, pmap_) )
{
    std::thread mapper_thread(&Mapper::run, this);// todo mono_stereo
//    std::thread mapper_thread_left(&Mapper::run2, this, std::ref(pmap_l), std::ref(pframe_l), true); todo 2.0 没做完，后面再做 两个线程分别进行左右目建图
//    std::thread mapper_thread_right(&Mapper::run2, this, std::ref(pmap_r), std::ref(pframe_r), false);

    mapper_thread.detach();
//    mapper_thread_left.detach();
//    mapper_thread_right.detach();

    std::cout << "\nMapper Object is created!\n";
}

void Mapper::run()  //直接在原函数上修改了，后面有需求可以保留原函数，把修改写到新函数里
{
    std::cout << "\nMapper is ready to process Keyframes!\n";
    
    Keyframe kf;

    std::thread estimator_thread(&Estimator::run, pestimator_);// todo mono_stereo stage2 暂时注释掉
//    std::thread lc_thread(&LoopCloser::run, ploopcloser_);// todo mono_stereo stage3

    while( !bexit_required_ ) {

        bool getnewkf = getNewKf(kf);
//        std::cout << "getnewkf = " << getnewkf << std::endl;
        if( getnewkf )
        {
//            std::cout << "getnewkf = " << getnewkf << std::endl;
            if( pslamstate_->debug_ )
                std::cout << "\n\n - [Mapper (back-End)]: New KF to process : KF #" 
                    << kf.kfid_ << "\n";

            if( pslamstate_->debug_ || pslamstate_->log_timings_ )
                Profiler::Start("0.Keyframe-Processing_Mapper");

            // Get new KF ptr
//            std::cout<<"kf.kfid_ = "<<kf.kfid_<<std::endl;
            std::shared_ptr<Frame> pnewkf = pmap_->getKeyframe(kf.kfid_);
            std::shared_ptr<Frame> pnewkf_l = pmap_l_->getKeyframe(kf.kfid_);// mono_stereo
            std::shared_ptr<Frame> pnewkf_r = pmap_r_->getKeyframe(kf.kfid_);
            std::shared_ptr<Frame> pnewkf_ls = pmap_ls_->getKeyframe(kf.kfid_);
            std::shared_ptr<Frame> pnewkf_rs = pmap_rs_->getKeyframe(kf.kfid_);
            assert( pnewkf );
            assert( pnewkf_l );
            assert( pnewkf_r );
            assert( pnewkf_ls );
            assert( pnewkf_rs );

            // Triangulate stereo
            if( pslamstate_->stereo_ ) 
            {
                if( pslamstate_->debug_ )
                    std::cout << "\n\n - [Mapper (back-End)]: Applying stereo matching!\n";

                cv::Mat imright;
                if( pslamstate_->use_clahe_ ) {
                    pmap_->ptracker_->pclahe_->apply(kf.imrightraw_, imright);
                } else {
                    imright = kf.imrightraw_;
                }
                std::vector<cv::Mat> vpyr_imright;
                cv::buildOpticalFlowPyramid(imright, vpyr_imright, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);

                pmap_->stereoMatching(*pnewkf, kf.vpyr_imleft_, vpyr_imright);
                
                if( pnewkf->nb2dkps_ > 0 && pnewkf->nb_stereo_kps_ > 0 ) {
                    
                    if( pslamstate_->debug_ ) {
                        std::cout << "\n\n - [Mapper (back-End)]: Stereo Triangulation!\n";

                        std::cout << "\n\n  \t >>> (BEFORE STEREO TRIANGULATION) New KF nb 2d kps / 3d kps / stereokps : ";
                        std::cout << pnewkf->nb2dkps_ << " / " << pnewkf->nb3dkps_ 
                            << " / " << pnewkf->nb_stereo_kps_ << "\n";
                    }

                    std::lock_guard<std::mutex> lock(pmap_->map_mutex_);

                    triangulateStereo(*pnewkf);

                    if( pslamstate_->debug_ ) {
                        std::cout << "\n\n  \t >>> (AFTER STEREO TRIANGULATION) New KF nb 2d kps / 3d kps / stereokps : ";
                        std::cout << pnewkf->nb2dkps_ << " / " << pnewkf->nb3dkps_ 
                            << " / " << pnewkf->nb_stereo_kps_ << "\n";
                    }
                }
            }

            if( pslamstate_->mono_stereo_ )
            {
                if( pslamstate_->debug_ )
                    std::cout << "\n\n - [Mapper (back-End)]: Applying stereo matching!\n";

                cv::Mat imright;
                cv::Mat imleft_s, imright_s;//
                if( pslamstate_->use_clahe_ ) {
                    pmap_->ptracker_->pclahe_->apply(kf.imrightraw_, imright);
                    pmap_rs_->ptracker_->pclahe_->apply(kf.imrightraw_s_, imright_s);//
                } else {
                    imright = kf.imrightraw_;
                    imright_s = kf.imrightraw_s_;//
                }
                std::vector<cv::Mat> vpyr_imright;
                std::vector<cv::Mat> vpyr_imright_s;//
                cv::buildOpticalFlowPyramid(imright, vpyr_imright, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
                cv::buildOpticalFlowPyramid(imright_s, vpyr_imright_s, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);//

//                pmap_->stereoMatching(*pnewkf, kf.vpyr_imleft_, vpyr_imright);// 这个可以先留着，但是应该可以去掉

//                pmap_l_->stereoMatching_s(*pnewkf_ls, kf.vpyr_imleft_s_, vpyr_imright_s);// mono_stereo 用pmap_还是其他的应该都可以
                pmap_l_->stereoMatching_s2(*pnewkf_ls, *pnewkf_l, *pnewkf_r, kf.vpyr_imleft_, kf.vpyr_imright_, kf.vpyr_imleft_s_, vpyr_imright_s);// mono_stereo 用pmap_还是其他的应该都可以。
                // 最终修改的是pnewkf_l下面的点结果（帧frame下面的地图点mapkps），更新3d点。双目区只在这里用，所以也不更新双目区的帧了，后面不要再用了！！
                //修改：将所有frame的3d点都更新，pnewkf_r也更新
                // todo stage2 右目也要单独匹配吧
                pmap_l_->stereoMatching_s2_r(*pnewkf_rs, *pnewkf_l, *pnewkf_r, kf.vpyr_imleft_, kf.vpyr_imright_, kf.vpyr_imleft_s_, vpyr_imright_s);// mono_stereo 用pmap_还是其他的应该都可以。

//                if( pnewkf->nb2dkps_ > 0 && pnewkf->nb_stereo_kps_ > 0 ) {
                if( pnewkf_l->nb2dkps_ > 0 && pnewkf_l->nb_stereo_kps_ > 0 ) {// mono_stereo

                    if( pslamstate_->debug_ ) {
                        std::cout << "\n\n - [Mapper (back-End)]: Stereo Triangulation mono_stereo!\n";

                        std::cout << "\t >>> (BEFORE STEREO TRIANGULATION mono_stereo) New KF nb 2d kps / 3d kps / stereokps : "
                                  << pnewkf_l->nb2dkps_ << " / " << pnewkf_l->nb3dkps_
                                  << " / " << pnewkf_l->nb_stereo_kps_ << std::endl;
                    }

                    std::lock_guard<std::mutex> lock(pmap_->map_mutex_);// todo 锁
                    std::lock_guard<std::mutex> lock_l(pmap_l_->map_mutex_);// mono_stereo
                    std::lock_guard<std::mutex> lock_r(pmap_r_->map_mutex_);
                    std::lock_guard<std::mutex> lock_ls(pmap_ls_->map_mutex_);
                    std::lock_guard<std::mutex> lock_rs(pmap_rs_->map_mutex_);

//                    triangulateStereo(*pnewkf);
//                    triangulateStereo_s(*pnewkf_ls);// mono_stereo 传入关键帧，获得关键帧特征点，计算深度，更新pmap地图点
                    triangulateStereo_s2(*pnewkf_ls, *pnewkf_l, *pnewkf_r, kf.vpyr_imleft_, kf.vpyr_imright_, kf.vpyr_imleft_s_, kf.vpyr_imright_s_);// mono_stereo 传入关键帧，获得关键帧特征点，计算深度，更新pmap地图点
                    // todo stage2 用右目点三角化
                    triangulateStereo_s2_r(*pnewkf_rs, *pnewkf_l, *pnewkf_r, kf.vpyr_imleft_, kf.vpyr_imright_, kf.vpyr_imleft_s_, kf.vpyr_imright_s_);// mono_stereo 传入关键帧，获得关键帧特征点，计算深度，更新pmap地图点


                    if( pslamstate_->debug_ ) {
                        std::cout << "\n\n  \t >>> (AFTER STEREO TRIANGULATION mono_stereo) New KF nb 2d kps / 3d kps / stereokps : ";
                        std::cout << pnewkf_l->nb2dkps_ << " / " << pnewkf_l->nb3dkps_
                                  << " / " << pnewkf_l->nb_stereo_kps_ << "\n";
                    }
                }
            }

            // Triangulate temporal
//            if( pnewkf->nb2dkps_ > 0 && pnewkf->kfid_ > 0 )
            if( pnewkf_l->nb2dkps_ > 0 && pnewkf_l->kfid_ > 0 ) //
            {
                if( pslamstate_->debug_ ) {
                    std::cout << "\n\n - [Mapper (back-End)]: Temporal Triangulation!\n";

                    std::cout << "\n\n  \t >>> (BEFORE TEMPORAL TRIANGULATION) New KF nb 2d kps / 3d kps / stereokps : ";
                    std::cout << pnewkf->nb2dkps_ << " / " << pnewkf->nb3dkps_ 
                        << " / " << pnewkf->nb_stereo_kps_ << "\n";
                }

                std::lock_guard<std::mutex> lock(pmap_->map_mutex_);
                std::lock_guard<std::mutex> lock_l(pmap_l_->map_mutex_);//
                std::lock_guard<std::mutex> lock_r(pmap_r_->map_mutex_);//
                std::lock_guard<std::mutex> lock_lm(pmap_lm_->map_mutex_);//
                std::lock_guard<std::mutex> lock_rm(pmap_rm_->map_mutex_);//
                std::lock_guard<std::mutex> lock_ls(pmap_ls_->map_mutex_);//
                std::lock_guard<std::mutex> lock_rs(pmap_rs_->map_mutex_);//

//                triangulateTemporal(*pnewkf); // todo
                triangulateTemporal(*pnewkf_l, pmap_l_, true, kf.imleftraw_);//得到的地图点分别存入左右当前帧
                triangulateTemporal(*pnewkf_r, pmap_r_, false, kf.imrightraw_);//得到的地图点分别存入左右当前帧

                if( pslamstate_->debug_ ) {
                    std::cout << "\n\n  \t >>> (AFTER TEMPORAL TRIANGULATION) New KF nb 2d kps / 3d kps / stereokps : ";
                    std::cout << pnewkf->nb2dkps_ << " / " << pnewkf->nb3dkps_ << " / " << pnewkf->nb_stereo_kps_ << "\n";
                }
            }

            // If Mono mode, check if reset is required
            if( pslamstate_->mono_ && pslamstate_->bvision_init_ ) 
            {
                if( kf.kfid_ == 1 && pnewkf->nb3dkps_ < 30 ) {
                    std::cout << "\n Bad initialization detected! Resetting\n";
                    pslamstate_->breset_req_ = true;
                    reset();
                    continue;
                } 
                else if( kf.kfid_ < 10 && pnewkf->nb3dkps_ < 3 ) {
                    std::cout << "\n Reset required : Nb 3D kps #" 
                            << pnewkf->nb3dkps_;
                    pslamstate_->breset_req_ = true;
                    reset();
                    continue;
                }
            }


            // Update the MPs and the covisilbe graph between KFs
            // (done here for real-time performance reason)
//            pmap_->updateFrameCovisibility(*pnewkf);
            pmap_l_->updateFrameCovisibility(*pnewkf_l, true);// mono_stereo
            pmap_r_->updateFrameCovisibility(*pnewkf_r, false);
            //双目区地图不知道用不用，暂时没写

            // Dirty but useful for visualization
//            pcurframe_->map_covkfs_ = pnewkf->map_covkfs_;
            pcurframe_l_->map_covkfs_ = pnewkf_l->map_covkfs_;// mono_stereo
            pcurframe_r_->map_covkfs_ = pnewkf_r->map_covkfs_;

            if( pslamstate_->use_brief_ && kf.kfid_ > 0 
                && !bnewkfavailable_ ) 
            {
                if( pslamstate_->bdo_track_localmap_ )
                {
                    if( pslamstate_->debug_ )
                        std::cout << "\n\n - [Mapper (back-End)]: matchingToLocalMap()!\n";
//                    matchingToLocalMap(*pnewkf); // todo
                    matchingToLocalMap(*pnewkf_l, pmap_l_, pcurframe_l_, true);// mono_stereo
                    matchingToLocalMap(*pnewkf_r, pmap_r_, pcurframe_r_, false);// mono_stereo
                }
            }

            // Send new KF to estimator for BA
//            pestimator_->addNewKf(pnewkf);
            pestimator_->addNewKf(pnewkf_l, pnewkf_r);// todo stage2

            // Send KF along with left image to LC thread
            if( pslamstate_->buse_loop_closer_ ) {
                ploopcloser_->addNewKf(pnewkf, kf.imleftraw_);// todo 暂不修改 stage3
            }

            if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "0.Keyframe-Processing_Mapper");

        } else {
            std::chrono::microseconds dura(100);
            std::this_thread::sleep_for(dura);
        }
    }

    pestimator_->bexit_required_ = true;
    ploopcloser_->bexit_required_ = true;

    estimator_thread.join(); // todo stage2
//    lc_thread.join(); // todo stage3
    
    std::cout << "\nMapper is stopping!\n";
}

void Mapper::run2(std::shared_ptr<MapManager>& pmap, std::shared_ptr<Frame>& pframe, bool isleft)
{
    std::cout << "\nMapper is ready to process Keyframes!\n";

    Keyframe kf;

//    std::thread estimator_thread(&Estimator::run, pestimator_);// todo mono_stereo
//    std::thread lc_thread(&LoopCloser::run, ploopcloser_);// todo mono_stereo

    while( !bexit_required_ ) {

//        if( getNewKf(kf) ) //  mono_stereo
        if( getNewKf(kf, isleft) )
        {
            if( pslamstate_->debug_ )
                std::cout << "\n\n - [Mapper (back-End)]: New KF to process : KF #"
                          << kf.kfid_ << "\n";

            if( pslamstate_->debug_ || pslamstate_->log_timings_ )
                Profiler::Start("0.Keyframe-Processing_Mapper");

            // Get new KF ptr
            std::shared_ptr<Frame> pnewkf = pmap->getKeyframe(kf.kfid_);
            assert( pnewkf );

            // Triangulate stereo
            if( pslamstate_->stereo_ )
            {
                if( pslamstate_->debug_ )
                    std::cout << "\n\n - [Mapper (back-End)]: Applying stereo matching!\n";

                cv::Mat imright;
                if( pslamstate_->use_clahe_ ) {
                    pmap->ptracker_->pclahe_->apply(kf.imrightraw_, imright);
                } else {
                    imright = kf.imrightraw_;
                }
                std::vector<cv::Mat> vpyr_imright;
                cv::buildOpticalFlowPyramid(imright, vpyr_imright, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);

                pmap->stereoMatching(*pnewkf, kf.vpyr_imleft_, vpyr_imright);

                if( pnewkf->nb2dkps_ > 0 && pnewkf->nb_stereo_kps_ > 0 ) {

                    if( pslamstate_->debug_ ) {
                        std::cout << "\n\n - [Mapper (back-End)]: Stereo Triangulation!\n";

                        std::cout << "\n\n  \t >>> (BEFORE STEREO TRIANGULATION) New KF nb 2d kps / 3d kps / stereokps : ";
                        std::cout << pnewkf->nb2dkps_ << " / " << pnewkf->nb3dkps_
                                  << " / " << pnewkf->nb_stereo_kps_ << "\n";
                    }

                    std::lock_guard<std::mutex> lock(pmap->map_mutex_);

                    triangulateStereo(*pnewkf);

                    if( pslamstate_->debug_ ) {
                        std::cout << "\n\n  \t >>> (AFTER STEREO TRIANGULATION) New KF nb 2d kps / 3d kps / stereokps : ";
                        std::cout << pnewkf->nb2dkps_ << " / " << pnewkf->nb3dkps_
                                  << " / " << pnewkf->nb_stereo_kps_ << "\n";
                    }
                }
            }

            if( pslamstate_->mono_stereo_ )
            {
                if( pslamstate_->debug_ )
                    std::cout << "\n\n - [Mapper (back-End)]: Applying stereo matching!\n";

//                cv::Mat imright;
                cv::Mat imleft_s, imright_s;
                if( pslamstate_->use_clahe_ ) {
//                    pmap->ptracker_->pclahe_->apply(kf.imrightraw_, imright);
                    pmap->ptracker_->pclahe_->apply(kf.imrightraw_s_, imright_s);
                } else {
//                    imright = kf.imrightraw_;
                    imright_s = kf.imrightraw_s_;
                }
//                std::vector<cv::Mat> vpyr_imright;
                std::vector<cv::Mat> vpyr_imright_s;
//                cv::buildOpticalFlowPyramid(imright, vpyr_imright, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
                cv::buildOpticalFlowPyramid(imright_s, vpyr_imright_s, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);

//                pmap->stereoMatching(*pnewkf, kf.vpyr_imleft_, vpyr_imright);
                pmap->stereoMatching_s(*pnewkf, kf.vpyr_imleft_s_, vpyr_imright_s);// todo mono_stereo 如果里面用了相机内参、图像大小之类的，要修改成双目区虚拟相机配置

                if( pnewkf->nb2dkps_ > 0 && pnewkf->nb_stereo_kps_ > 0 ) {

                    if( pslamstate_->debug_ ) {
                        std::cout << "\n\n - [Mapper (back-End)]: Stereo Triangulation!\n";

                        std::cout << "\n\n  \t >>> (BEFORE STEREO TRIANGULATION) New KF nb 2d kps / 3d kps / stereokps : ";
                        std::cout << pnewkf->nb2dkps_ << " / " << pnewkf->nb3dkps_
                                  << " / " << pnewkf->nb_stereo_kps_ << "\n";
                    }

                    std::lock_guard<std::mutex> lock(pmap->map_mutex_);//todo mono_stereo 锁
//                    std::unique_ptr<std::lock_guard<std::mutex>> lock;    //
//                    if (isleft) {
//                        lock = std::make_unique<std::lock_guard<std::mutex>>(pmap_l_->map_mutex_);
//                    } else {
//                        lock = std::make_unique<std::lock_guard<std::mutex>>(pmap_r_->map_mutex_);
//                    }


                    triangulateStereo(*pnewkf);

                    if( pslamstate_->debug_ ) {
                        std::cout << "\n\n  \t >>> (AFTER STEREO TRIANGULATION) New KF nb 2d kps / 3d kps / stereokps : ";
                        std::cout << pnewkf->nb2dkps_ << " / " << pnewkf->nb3dkps_
                                  << " / " << pnewkf->nb_stereo_kps_ << "\n";
                    }
                }
            }

            // Triangulate temporal
            if( pnewkf->nb2dkps_ > 0 && pnewkf->kfid_ > 0 )
            {
                if( pslamstate_->debug_ ) {
                    std::cout << "\n\n - [Mapper (back-End)]: Temporal Triangulation!\n";

                    std::cout << "\n\n  \t >>> (BEFORE TEMPORAL TRIANGULATION) New KF nb 2d kps / 3d kps / stereokps : ";
                    std::cout << pnewkf->nb2dkps_ << " / " << pnewkf->nb3dkps_
                              << " / " << pnewkf->nb_stereo_kps_ << "\n";
                }

//                std::lock_guard<std::mutex> lock(pmap->map_mutex_);//todo mono_stereo 锁
                std::unique_ptr<std::lock_guard<std::mutex>> lock;
                if (isleft) {
                    lock = std::make_unique<std::lock_guard<std::mutex>>(pmap_l_->map_mutex_);
                } else {
                    lock = std::make_unique<std::lock_guard<std::mutex>>(pmap_r_->map_mutex_);
                }

                triangulateTemporal(*pnewkf);

                if( pslamstate_->debug_ ) {
                    std::cout << "\n\n  \t >>> (AFTER TEMPORAL TRIANGULATION) New KF nb 2d kps / 3d kps / stereokps : ";
                    std::cout << pnewkf->nb2dkps_ << " / " << pnewkf->nb3dkps_ << " / " << pnewkf->nb_stereo_kps_ << "\n";
                }
            }

            // If Mono mode, check if reset is required
            if( pslamstate_->mono_ && pslamstate_->bvision_init_ )
            {
                if( kf.kfid_ == 1 && pnewkf->nb3dkps_ < 30 ) {
                    std::cout << "\n Bad initialization detected! Resetting\n";
                    pslamstate_->breset_req_ = true;
                    reset();
                    continue;
                }
                else if( kf.kfid_ < 10 && pnewkf->nb3dkps_ < 3 ) {
                    std::cout << "\n Reset required : Nb 3D kps #"
                              << pnewkf->nb3dkps_;
                    pslamstate_->breset_req_ = true;
                    reset();
                    continue;
                }
            }

            // Update the MPs and the covisilbe graph between KFs
            // (done here for real-time performance reason)
            pmap->updateFrameCovisibility(*pnewkf);

            // Dirty but useful for visualization
            pframe->map_covkfs_ = pnewkf->map_covkfs_;

            if( pslamstate_->use_brief_ && kf.kfid_ > 0
                && !bnewkfavailable_ )
            {
                if( pslamstate_->bdo_track_localmap_ )
                {
                    if( pslamstate_->debug_ )
                        std::cout << "\n\n - [Mapper (back-End)]: matchingToLocalMap()!\n";
                    matchingToLocalMap(*pnewkf);
                }
            }

            // Send new KF to estimator for BA
            pestimator_->addNewKf(pnewkf);

            // Send KF along with left image to LC thread
            if( pslamstate_->buse_loop_closer_ ) {
                ploopcloser_->addNewKf(pnewkf, kf.imleftraw_);
            }

            if( pslamstate_->debug_ || pslamstate_->log_timings_ )
                Profiler::StopAndDisplay(pslamstate_->debug_, "0.Keyframe-Processing_Mapper");

        } else {
            std::chrono::microseconds dura(100);
            std::this_thread::sleep_for(dura);
        }
    }

    pestimator_->bexit_required_ = true;
    ploopcloser_->bexit_required_ = true;

//    estimator_thread.join();
//    lc_thread.join();

    std::cout << "\nMapper is stopping!\n";
}


void Mapper::triangulateTemporal(Frame &frame)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_TriangulateTemporal");

    // Get New KF kps / pose
    std::vector<Keypoint> vkps = frame.getKeypoints2d();

    Sophus::SE3d Twcj = frame.getTwc();

    if( vkps.empty() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t >>> No kps to temporal triangulate...\n";
        return;
    }

    // Setup triangulatation for OpenGV-based mapping
    size_t nbkps = vkps.size();

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vleftbvs, vrightbvs;
    vleftbvs.reserve(nbkps);
    vrightbvs.reserve(nbkps);

    // Init a pkf object that will point to the prev KF to use
    // for triangulation
    std::shared_ptr<Frame> pkf;
    pkf.reset( new Frame() );
    pkf->kfid_ = -1;

    // Relative motions between new KF and prev. KFs
    int relkfid = -1;
    Sophus::SE3d Tcicj, Tcjci;
    Eigen::Matrix3d Rcicj;

    // New 3D MPs projections
    cv::Point2f left_px_proj, right_px_proj;
    float ldist, rdist;
    Eigen::Vector3d left_pt, right_pt, wpt;

    int good = 0, candidates = 0;

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vwpts;
    std::vector<int> vlmids;
    vwpts.reserve(nbkps);
    vlmids.reserve(nbkps);

    // We go through all the 2D kps in new KF
    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        // Get the related MP and check if it is ready 
        // to be triangulated 
        std::shared_ptr<MapPoint> plm = pmap_->getMapPoint(vkps.at(i).lmid_);

        if( plm == nullptr ) {
            pmap_->removeMapPointObs(vkps.at(i).lmid_, frame.kfid_);
            continue;
        }

        // If MP is already 3D continue (should not happen)
        if( plm->is3d_ ) {
            continue;
        }

        // Get the set of KFs sharing observation of this 2D MP
        std::set<int> co_kf_ids = plm->getKfObsSet();

        // Continue if new KF is the only one observing it
        if( co_kf_ids.size() < 2 ) {
            continue;
        }

        int kfid = *co_kf_ids.begin();

        if( frame.kfid_ == kfid ) {
            continue;
        }

        // Get the 1st KF observation of the related MP
        pkf = pmap_->getKeyframe(kfid);
        
        if( pkf == nullptr ) {
            continue;
        }

        // Compute relative motion between new KF and selected KF
        // (only if req.)
        if( relkfid != kfid ) {
            Sophus::SE3d Tciw = pkf->getTcw();
            Tcicj = Tciw * Twcj;
            Tcjci = Tcicj.inverse();
            Rcicj = Tcicj.rotationMatrix();

            relkfid = kfid;
        }

        // If no motion between both KF, skip
        if( pslamstate_->stereo_ && Tcicj.translation().norm() < 0.01 ) {
            continue;
        }
        
        // Get obs kp
        Keypoint kfkp = pkf->getKeypointById(vkps.at(i).lmid_);
        if( kfkp.lmid_ != vkps.at(i).lmid_ ) {
            continue;
        }

        // Check rotation-compensated parallax
        cv::Point2f rotpx = frame.projCamToImage(Rcicj * vkps.at(i).bv_);
        double parallax = cv::norm(kfkp.unpx_ - rotpx);

        candidates++;

        // Compute 3D pos and check if its good or not
        left_pt = computeTriangulation(Tcicj, kfkp.bv_, vkps.at(i).bv_);

        // Project into right cam (new KF)
        right_pt = Tcjci * left_pt;

        // Ensure that the 3D MP is in front of both camera
        if( left_pt.z() < 0.1 || right_pt.z() < 0.1 ) {
            if( parallax > 20. ) {
                pmap_->removeMapPointObs(kfkp.lmid_, frame.kfid_);
            }
            continue;
        }

        // Remove MP with high reprojection error
        left_px_proj = pkf->projCamToImage(left_pt);
        right_px_proj = frame.projCamToImage(right_pt);
        ldist = cv::norm(left_px_proj - kfkp.unpx_);
        rdist = cv::norm(right_px_proj - vkps.at(i).unpx_);

        if( ldist > pslamstate_->fmax_reproj_err_ 
            || rdist > pslamstate_->fmax_reproj_err_ ) {
            if( parallax > 20. ) {
                pmap_->removeMapPointObs(kfkp.lmid_, frame.kfid_);
            }
            continue;
        }

        // The 3D pos is good, update SLAM MP and related KF / Frame
        wpt = pkf->projCamToWorld(left_pt);
        pmap_->updateMapPoint(vkps.at(i).lmid_, wpt, 1./left_pt.z());

        good++;
    }

    if( pslamstate_->debug_ )
        std::cout << "\n \t >>> Temporal Mapping : " << good << " 3D MPs out of " 
            << candidates << " kps !\n";

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_TriangulateTemporal");
}

void Mapper::triangulateTemporal(Frame &frame, std::shared_ptr<MapManager>& pmap, bool isleft, cv::Mat& img)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_TriangulateTemporal");

    // Get New KF kps / pose
//    std::vector<Keypoint> vkps = frame.getKeypoints2d();//当前关键帧上的点
    std::vector<Keypoint> vkps;//当前关键帧上的点
    if(isleft){
        vkps = frame.getKeypoints2d_l();
    } else {
        vkps = frame.getKeypoints2d_r();
    }

    Sophus::SE3d Twcj = frame.getTwc();//左当前关键帧到世界

    if( vkps.empty() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t >>> No kps to temporal triangulate...\n";
        return;
    }

    // Setup triangulatation for OpenGV-based mapping
    size_t nbkps = vkps.size();

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vleftbvs, vrightbvs;
    vleftbvs.reserve(nbkps);
    vrightbvs.reserve(nbkps);

    // Init a pkf object that will point to the prev KF to use
    // for triangulation
    std::shared_ptr<Frame> pkf;
    pkf.reset( new Frame() );//应该不用改
    pkf->kfid_ = -1;

    // Relative motions between new KF and prev. KFs
    int relkfid = -1;
    Sophus::SE3d Tcicj, Tcjci;
    Eigen::Matrix3d Rcicj;

    // New 3D MPs projections
    cv::Point2f left_px_proj, right_px_proj;
    float ldist, rdist;
    Eigen::Vector3d left_pt, right_pt, wpt;

    int good = 0, candidates = 0;

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vwpts;
    std::vector<int> vlmids;
    vwpts.reserve(nbkps);
    vlmids.reserve(nbkps);

    cv::Mat img_test = img.clone();//用于可视化
    if (img_test.channels() == 1)
        cv::cvtColor(img_test, img_test, cv::COLOR_GRAY2BGR);
    std::string img_name;//用于可视化

    int ok=0;
    int notok=0;
    // We go through all the 2D kps in new KF
    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        // Get the related MP and check if it is ready
        // to be triangulated
        std::shared_ptr<MapPoint> plm = pmap->getMapPoint(vkps.at(i).lmid_);//

        if( plm == nullptr ) {
            pmap->removeMapPointObs(vkps.at(i).lmid_, frame.kfid_);//
//            std::cout<<"error 111111"<<std::endl;
            continue;
        }

        // If MP is already 3D continue (should not happen)
        if( plm->is3d_ ) {
//            std::cout<<"error 222222"<<std::endl;
            continue;
        }

        // Get the set of KFs sharing observation of this 2D MP
        std::set<int> co_kf_ids = plm->getKfObsSet();// 有关共视图的部分还没关注

        // Continue if new KF is the only one observing it
        if( co_kf_ids.size() < 2 ) {
//            std::cout<<"error 33333"<<std::endl;
            continue;
        }

        int kfid = *co_kf_ids.begin();

        if( frame.kfid_ == kfid ) {
//            std::cout<<"error 444444"<<std::endl;
            continue;
        }

        // Get the 1st KF observation of the related MP
        pkf = pmap->getKeyframe(kfid);//

        if( pkf == nullptr ) {
//            std::cout<<"error 555555"<<std::endl;
            continue;
        }

        // Compute relative motion between new KF and selected KF
        // (only if req.)
        if( relkfid != kfid ) {
            Sophus::SE3d Tciw = pkf->getTcw();//世界到左选择关键帧
            if(isleft){
                Tcicj = Tciw * Twcj;// 左当前关键帧到左选择关键帧=世界到左关键帧*左当前关键帧到世界
            } else {
                Tcicj = pslamstate_->T_right_left_ * Tciw * Twcj * pslamstate_->T_left_right_;// 右当前关键帧到右选择关键帧=左到右*世界到左关键帧*左当前关键帧到世界*右到左
            }
            Tcjci = Tcicj.inverse();
            Rcicj = Tcicj.rotationMatrix();

            relkfid = kfid;
        }

        // If no motion between both KF, skip
        if( (pslamstate_->stereo_ || pslamstate_->mono_stereo_) && Tcicj.translation().norm() < 0.01 ) {
//            std::cout<<"error 666666"<<std::endl;//这个很多，位姿计算有问题
//            std::cout<<"Tcicj.translation().norm() = "<<Tcicj.translation().norm()<<std::endl;//有问题，发现是个定值
            continue;
        }

        // Get obs kp
        Keypoint kfkp = pkf->getKeypointById(vkps.at(i).lmid_); //选择关键帧上的点
        if( kfkp.lmid_ != vkps.at(i).lmid_ ) {
//            std::cout<<"error 777777"<<std::endl;
            continue;
        }

        // Check rotation-compensated parallax
        cv::Point2f rotpx;
        rotpx = frame.projCamToImage(Rcicj * vkps.at(i).bv_);// frame是左/右当前关键帧  左/右相机坐标系到左/右像素坐标系*左/右当前关键帧到左/右选择关键帧*左/右当前关键帧点

        double parallax = cv::norm(kfkp.unpx_ - rotpx); //选择关键帧上的像素点-当前关键帧上的点在选择关键帧上的投影像素点

        candidates++;

        // Compute 3D pos and check if its good or not
        left_pt = computeTriangulation(Tcicj, kfkp.bv_, vkps.at(i).bv_);// Tcicj:当前关键帧到选择关键帧 kfkp:选择关键帧上的点 vkps:当前关键帧上的点 结果left_pt:选择关键帧坐标系下的三维点

        // Project into right cam (new KF)
        right_pt = Tcjci * left_pt;//   当前关键帧坐标系下的三维点=选择关键帧到当前关键帧*选择关键帧坐标系下的三维点

        // Ensure that the 3D MP is in front of both camera
        if( left_pt.z() < 0.05 || right_pt.z() < 0.05 ) { // 0.1
            if( parallax > 20. ) {
                pmap->removeMapPointObs(kfkp.lmid_, frame.kfid_);//
            }
//            std::cout<<"error 888888"<<std::endl;
            continue;
        }

        // Remove MP with high reprojection error
        left_px_proj = pkf->projCamToImage(left_pt);
        right_px_proj = frame.projCamToImage(right_pt);

        //可视化投影点
        cv::circle(img_test, cv::Point2f(left_px_proj.x, left_px_proj.y), 5, cv::Scalar(0, 255, 0), 1);
//        cv::circle(img_test, cv::Point2f(right_px_proj.x, right_px_proj.y), 5, cv::Scalar(0, 255, 0), 1);

        ldist = cv::norm(left_px_proj - kfkp.unpx_);
        rdist = cv::norm(right_px_proj - vkps.at(i).unpx_);

        if( ldist > pslamstate_->fmax_reproj_err_
            || rdist > pslamstate_->fmax_reproj_err_ ) {
            if( parallax > 20. ) {
                pmap->removeMapPointObs(kfkp.lmid_, frame.kfid_);//
            }
//            std::cout<<"error 999999"<<std::endl;
            continue;
        }

        // The 3D pos is good, update SLAM MP and related KF / Frame
//        wpt = pkf->projCamToWorld(left_pt);
//        pmap->updateMapPoint(vkps.at(i).lmid_, wpt, 1./left_pt.z());//
        if(isleft){
            wpt = pkf->projCamToWorld(left_pt);//   左相机到世界*左选择关键帧坐标系下的三维点
            pmap->updateMapPoint(vkps.at(i).lmid_, wpt, true, 1./left_pt.z());//这个kfanch_invdepth是什么
        } else {
            wpt = pkf->projCamToWorld(pslamstate_->T_left_right_*left_pt);// 左相机到世界*右到左*右选择关键帧坐标系下的三维点
            pmap->updateMapPoint(vkps.at(i).lmid_, wpt, false, 1./left_pt.z());//
        }

////////////////////
        //感觉左右目单目区3d点反了，可视化验证。改成叠加，累计画在一张图上
//        cv::Point2f kp;
//        if(isleft){
//            kp = pkf->projWorldToImageDist(wpt);    //将wpt投影到左相机得到kp
//            img_name = "/home/hl/project/ov2_diverg_ws/test/imgl_temporal.png";
//        } else {
//            kp = pkf->projWorldToImageDist_right(wpt);//函数似乎有问题
////            std::cout<<"1. kp.x = "<<kp.x<<", kp.y = "<<kp.y<<std::endl;
////            kp = pkf->projWorldToRightImage(wpt);// 改成这个可以了，说明上面的函数可能有问题，所有用到此函数的都检查一下
////            std::cout<<"2. kp.x = "<<kp.x<<", kp.y = "<<kp.y<<std::endl;
//            img_name = "/home/hl/project/ov2_diverg_ws/test/imgr_temporal.png";
//        }
//        //检查kp坐标是否超出图像范围
//        if(kp.x<0 || kp.x>img_test.cols || kp.y<0 || kp.y>img_test.rows){
//            std::cout<<"kp.x = "<<kp.x<<", kp.y = "<<kp.y<<"kp out of range"<<std::endl;
//            notok++;
//            continue;
//        } else {
//            ok++;
//        }
//        std::cout<<"ok = "<<ok<<", notok = "<<notok<<std::endl;// 正常应该全在图像范围内
//        cv::circle(img_test, kp, 3, cv::Scalar(0, 0, 255), -1);
//        cv::imshow("test", img_test);
//        cv::imwrite(img_name, img_test);//

        good++;
    }
//    if(isleft){
//        img_name = "/home/hl/project/ov2_diverg_ws/test/imgl_temporal_all.png";
//        cv::imshow("imgl_temporal_all", img_test);
//        cv::waitKey(1);
//        cv::imwrite(img_name, img_test);//
//    } else {
//        img_name = "/home/hl/project/ov2_diverg_ws/test/imgr_temporal_all.png";
//        cv::imshow("imgr_temporal_all", img_test);
//        cv::waitKey(1);
//        cv::imwrite(img_name, img_test);//
//    }

    if( pslamstate_->debug_ )
        std::cout << "\t >>> Temporal Mapping : " << good << " 3D MPs out of "
                  << candidates << " kps !\n";

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_TriangulateTemporal");
}

void Mapper::triangulateStereo(Frame &frame)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_TriangulateStereo");

    // INIT STEREO TRIANGULATE
    std::vector<Keypoint> vkps;

    // Get the new KF stereo kps
    vkps = frame.getKeypointsStereo();

    if( vkps.empty() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t >>> No kps to stereo triangulate...\n";
        return;
    }

    // Store the stereo kps along with their idx
    std::vector<int> vstereoidx;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vleftbvs, vrightbvs;

    size_t nbkps = vkps.size();

    vleftbvs.reserve(nbkps);
    vrightbvs.reserve(nbkps);

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vwpts;
    std::vector<int> vlmids;
    vwpts.reserve(nbkps);
    vlmids.reserve(nbkps);

    // Get the extrinsic transformation
    Sophus::SE3d Tlr = frame.pcalib_rightcam_->getExtrinsic();
    Sophus::SE3d Trl = Tlr.inverse();

    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        if( !vkps.at(i).is3d_ && vkps.at(i).is_stereo_ )
        {
            vstereoidx.push_back(i);
            vleftbvs.push_back(vkps.at(i).bv_);
            vrightbvs.push_back(vkps.at(i).rbv_);
        }
    }

    if( vstereoidx.empty() ) {
        return;
    }

    size_t nbstereo = vstereoidx.size();

    cv::Point2f left_px_proj, right_px_proj;
    float ldist, rdist;
    Eigen::Vector3d left_pt, right_pt, wpt;

    int kpidx;

    int good = 0;

    // For each stereo kp
    for( size_t i = 0 ; i < nbstereo ; i++ ) 
    {
        kpidx = vstereoidx.at(i);

        if( pslamstate_->bdo_stereo_rect_ ) {
            float disp = vkps.at(kpidx).unpx_.x - vkps.at(kpidx).runpx_.x;

            if( disp < 0. ) {
                frame.removeStereoKeypointById(vkps.at(kpidx).lmid_);
                continue;
            }

            float z = frame.pcalib_leftcam_->fx_ * frame.pcalib_rightcam_->Tcic0_.translation().norm() / fabs(disp);

            left_pt << vkps.at(kpidx).unpx_.x, vkps.at(kpidx).unpx_.y, 1.;
            left_pt = z * frame.pcalib_leftcam_->iK_ * left_pt.eval();
        } else {
            // Triangulate in left cam frame
            left_pt = computeTriangulation(Tlr, vleftbvs.at(i), vrightbvs.at(i));
        }

        // Project into right cam frame
        right_pt = Trl * left_pt;

        if( left_pt.z() < 0.1 || right_pt.z() < 0.1 ) {
            frame.removeStereoKeypointById(vkps.at(kpidx).lmid_);
            continue;
        }

        // Remove MP with high reprojection error
        left_px_proj = frame.projCamToImage(left_pt);
        right_px_proj = frame.projCamToRightImage(left_pt);
        ldist = cv::norm(left_px_proj - vkps.at(kpidx).unpx_);
        rdist = cv::norm(right_px_proj - vkps.at(kpidx).runpx_);

        if( ldist > pslamstate_->fmax_reproj_err_
            || rdist > pslamstate_->fmax_reproj_err_ ) {
            frame.removeStereoKeypointById(vkps.at(kpidx).lmid_);
            continue;
        }

        // Project MP in world frame
        wpt = frame.projCamToWorld(left_pt);

        pmap_->updateMapPoint(vkps.at(kpidx).lmid_, wpt, 1./left_pt.z());

        good++;
    }

    if( pslamstate_->debug_ )
        std::cout << "\n \t >>> Stereo Mapping : " << good << " 3D MPs out of " 
            << nbstereo << " kps !\n";

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_TriangulateStereo");
}

void Mapper::triangulateStereo_s(Frame &frame)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_TriangulateStereo");

    // INIT STEREO TRIANGULATE
    std::vector<Keypoint> vkps;

    // Get the new KF stereo kps
    vkps = frame.getKeypointsStereo();

    if( vkps.empty() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t >>> No kps to stereo triangulate...\n";
        return;
    }

    // Store the stereo kps along with their idx
    std::vector<int> vstereoidx;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vleftbvs, vrightbvs;

    size_t nbkps = vkps.size();

    vleftbvs.reserve(nbkps);
    vrightbvs.reserve(nbkps);

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vwpts;
    std::vector<int> vlmids;
    vwpts.reserve(nbkps);
    vlmids.reserve(nbkps);

    // Get the extrinsic transformation
//    Sophus::SE3d Tlr = frame.pcalib_rightcam_s_->getExtrinsic();//源代码：右到左
    // pcalib_rightcam_s_->Tc0ci_:右双目区到右目  pcalib_rightcam_->Tc0ci_:右目到左目 pcalib_leftcam_s_->Tcic0_:左目到左双目
    Sophus::SE3d Tlr = frame.pcalib_leftcam_s_->Tcic0_*frame.pcalib_rightcam_->Tc0ci_*frame.pcalib_rightcam_s_->getExtrinsic();//mono_stereo 右双目到左双目=左到左双目*右到左*右双目到右
    Sophus::SE3d Trl = Tlr.inverse();

    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        if( !vkps.at(i).is3d_ && vkps.at(i).is_stereo_ )
        {
            vstereoidx.push_back(i);
            vleftbvs.push_back(vkps.at(i).bv_);
            vrightbvs.push_back(vkps.at(i).rbv_);
        }
    }

    if( vstereoidx.empty() ) {
        return;
    }

    size_t nbstereo = vstereoidx.size();

    cv::Point2f left_px_proj, right_px_proj;
    float ldist, rdist;
    Eigen::Vector3d left_pt, right_pt, wpt, left_pt_1;

    int kpidx;

    int good = 0;

    // For each stereo kp
    for( size_t i = 0 ; i < nbstereo ; i++ )
    {
        kpidx = vstereoidx.at(i);

//        if( pslamstate_->bdo_stereo_rect_ ) {   //用视差求深度，暂时没用但应该也可以用
            float disp = vkps.at(kpidx).unpx_.x - vkps.at(kpidx).runpx_.x;

            if( disp < 0. ) {
                frame.removeStereoKeypointById(vkps.at(kpidx).lmid_);
                continue;
            }

//            float z = frame.pcalib_leftcam_->fx_ * frame.pcalib_rightcam_->Tcic0_.translation().norm() / fabs(disp);
            float z = frame.pcalib_leftcam_s_->fx_ * frame.pcalib_rightcam_->Tcic0_.translation().norm() / fabs(disp);//mono_stereo 视差求深度：z=f*b/d  “左右目”和“左双目右双目”的基线长度一样，所以就用frame.pcalib_rightcam_->Tcic0_可以

            left_pt_1 << vkps.at(kpidx).unpx_.x, vkps.at(kpidx).unpx_.y, 1.;
            left_pt_1 = z * frame.pcalib_leftcam_s_->iK_ * left_pt_1.eval();
//        } else {
            // Triangulate in left cam frame
            left_pt = computeTriangulation(Tlr, vleftbvs.at(i), vrightbvs.at(i));
//        }
//        std::cout<<"left_pt_1.z() (depth by disparity) = "<<left_pt_1.z()<<std::endl;
//        std::cout<<"left_pt.z() (depth by triangulation) = "<<left_pt.z()<<std::endl;
//        std::cout<<"depth diff between 2 methods = "<<left_pt.z() - left_pt_1.z()<<std::endl;//验证深度求解正确

        // Project into right cam frame
        right_pt = Trl * left_pt;

        if( left_pt.z() < 0.1 || right_pt.z() < 0.1 ) {
            frame.removeStereoKeypointById(vkps.at(kpidx).lmid_);
            continue;
        }

        // Remove MP with high reprojection error
        left_px_proj = frame.projCamToImage_s(left_pt);//mono_stereo
        right_px_proj = frame.projCamToRightImage_s(left_pt);//mono_stereo
        ldist = cv::norm(left_px_proj - vkps.at(kpidx).unpx_);
        rdist = cv::norm(right_px_proj - vkps.at(kpidx).runpx_);

        if( ldist > pslamstate_->fmax_reproj_err_
            || rdist > pslamstate_->fmax_reproj_err_ ) {
            frame.removeStereoKeypointById(vkps.at(kpidx).lmid_);
            continue;
        }

        // Project MP in world frame
        //原算法：存到左目地图
//        wpt = frame.projCamToWorld(left_pt);//左目到世界
//        pmap_->updateMapPoint(vkps.at(kpidx).lmid_, wpt, 1./left_pt.z());//存到左目地图pmap里
        wpt = frame.projCamToWorld_s(left_pt);// mono_stereo 左双目到世界
        pmap_->updateMapPoint(vkps.at(kpidx).lmid_, wpt, 1./left_pt.z());// mono_stereo 存到左目地图pmap_l里
        pmap_l_->updateMapPoint(vkps.at(kpidx).lmid_, wpt, true, 1./left_pt.z());// mono_stereo 存到左目地图pmap_l里
        pmap_r_->updateMapPoint(vkps.at(kpidx).lmid_, wpt, false, 1./left_pt.z());// mono_stereo 存到右目地图pmap_r里， todo 问题：结果会不会不一样？
        //建图也要用到3d点信息，所以双目区也要地图？ (BEFORE STEREO TRIANGULATION mono_stereo) New KF nb 2d kps / 3d kps / stereokps
        pmap_ls_->updateMapPoint_s(vkps.at(kpidx).lmid_, wpt, true, 1./left_pt.z());// mono_stereo 存到左双目地图pmap_ls里
        pmap_rs_->updateMapPoint_s(vkps.at(kpidx).lmid_, wpt, false, 1./left_pt.z());// mono_stereo 存到左双目地图pmap_ls里
        //要存的点都是wpt吗

        good++;
    }

    if( pslamstate_->debug_ )
        std::cout << "\n \t >>> Stereo Mapping : " << good << " 3D MPs out of "
                  << nbstereo << " kps !\n";

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_TriangulateStereo");
}

void Mapper::triangulateStereo_s2(Frame &frame_s, Frame &frame, Frame &frame_r, const std::vector<cv::Mat> &vleftpyr_origin, const std::vector<cv::Mat> &vrightpyr_origin, const std::vector<cv::Mat> &vleftpyr_s, const std::vector<cv::Mat> &vrightpyr_s) // 其实得到了左右目的匹配关系，也可以不用双目区求深度，就用左右目求深度也可以。但是视差方法不能用了！！！
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_TriangulateStereo");

    // INIT STEREO TRIANGULATE
    std::vector<Keypoint> vkps;

    // Get the new KF stereo kps
    vkps = frame.getKeypointsStereo();//左目 只用左目？

    if( vkps.empty() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t >>> No kps to stereo triangulate...\n";
        return;
    }

    // Store the stereo kps along with their idx
    std::vector<int> vstereoidx;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vleftbvs, vrightbvs;//原图坐标
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vleftbvs_s, vrightbvs_s;//双目区坐标

    size_t nbkps = vkps.size();

    vleftbvs.reserve(nbkps);//原图坐标
    vrightbvs.reserve(nbkps);//原图坐标
    vleftbvs_s.reserve(nbkps);//双目区坐标
    vrightbvs_s.reserve(nbkps);//双目区原图坐标

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vwpts;
    std::vector<int> vlmids;
    vwpts.reserve(nbkps);
    vlmids.reserve(nbkps);

    // Get the extrinsic transformation
    Sophus::SE3d Tlr = frame.pcalib_rightcam_->getExtrinsic();//源代码：右到左
    // pcalib_rightcam_s_->Tc0ci_:右双目区到右目  pcalib_rightcam_->Tc0ci_:右目到左目 pcalib_leftcam_s_->Tcic0_:左目到左双目
    Sophus::SE3d Tlr2 = frame.pcalib_leftcam_s_->Tcic0_ * frame.pcalib_rightcam_->Tc0ci_ * frame.pcalib_rightcam_s_->getExtrinsic();//mono_stereo 右双目到左双目=左到左双目*右到左*右双目到右
    Sophus::SE3d Trl = Tlr.inverse();//左到右
    Sophus::SE3d Trl2 = Tlr2.inverse();//左双目到右双目

/*    cv::Mat img_left = vleftpyr_origin[0].clone();//用于可视化检查
    cv::Mat img_right = vrightpyr_origin[0].clone();
    cv::Mat img_left_s = vleftpyr_s[0].clone();//用于可视化检查
    cv::Mat img_right_s = vrightpyr_s[0].clone();
    // 如果图像是灰度图，转换为彩色图
    if (img_left.channels() == 1)
        cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
    if (img_right.channels() == 1)
        cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);
    if (img_left_s.channels() == 1)
        cv::cvtColor(img_left_s, img_left_s, cv::COLOR_GRAY2BGR);
    if (img_right_s.channels() == 1)
        cv::cvtColor(img_right_s, img_right_s, cv::COLOR_GRAY2BGR);
    cv::Mat combined_img, combined_img_s;
    cv::hconcat(img_left, img_right, combined_img);
    cv::hconcat(img_left_s, img_right_s, combined_img_s);
    //保存结果，用于在compute_depth.cpp程序单独测试深度求解
    cv::imwrite("/home/hl/project/ov2_diverg_ws/test/test_depth/img_left.jpg", img_left);
    cv::imwrite("/home/hl/project/ov2_diverg_ws/test/test_depth/img_right.jpg", img_right);
    cv::imwrite("/home/hl/project/ov2_diverg_ws/test/test_depth/img_left_s.jpg", img_left_s);
    cv::imwrite("/home/hl/project/ov2_diverg_ws/test/test_depth/img_right_s.jpg", img_right_s);
    std::ofstream origin_points_file("/home/hl/project/ov2_diverg_ws/test/test_depth/origin_points.txt");
    std::ofstream stereo_points_file("/home/hl/project/ov2_diverg_ws/test/test_depth/stereo_points.txt");*/

    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        if( !vkps.at(i).is3d_ && vkps.at(i).is_stereo_ )
        {
            vstereoidx.push_back(i);
            vleftbvs.push_back(vkps.at(i).bv_);//原图相机坐标
            vrightbvs.push_back(vkps.at(i).rbv_);//原图相机坐标

            cv::Point2f pt_l = vkps.at(i).unpx_;// 左目点
            Eigen::Vector3d pt_l_point3D((pt_l.x - pslamstate_->cxl_)/pslamstate_->fxl_, (pt_l.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化平面3d坐标
            Eigen::Vector3d pt_l_point3D1 = pslamstate_->R_sl.inverse() * pt_l_point3D;//左目到左双目
            pt_l_point3D1.normalize(); // 归一化后 pt_l_point3D1 的模长为 1
            vleftbvs_s.push_back(pt_l_point3D1);// 双目区坐标

            cv::Point2f pt_r = vkps.at(i).runpx_;// 右目点
            Eigen::Vector3d pt_r_point3D((pt_r.x - pslamstate_->cxl_)/pslamstate_->fxl_, (pt_r.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化平面3d坐标
            Eigen::Vector3d pt_r_point3D1 = pslamstate_->R_sr.inverse() * pt_r_point3D;//右目到右双目
            pt_r_point3D1.normalize(); // 归一化后 pt_r_point3D1 的模长为 1
            vrightbvs_s.push_back(pt_r_point3D1);//双目区坐标

/*            cv::Point2f test1 = frame.projCamToImage(vkps.at(i).bv_);//
            cv::Point2f test2 = frame.projCamToImage(vkps.at(i).rbv_);//
            std::cout<<"vkps.at(i).bv_ = "<<test1<<std::endl;
            std::cout<<"vkps.at(i).rbv_ = "<<test2<<std::endl;
            std::cout<<"vkps.at(i).unpx_ = "<<vkps.at(i).unpx_<<std::endl;
            std::cout<<"vkps.at(i).runpx_ = "<<vkps.at(i).runpx_<<std::endl;//验证原图点坐标没问题
            //检查 验证左右目匹配点坐标是对的
            cv::Scalar random_color(rand() % 256, rand() % 256, rand() % 256);
            cv::circle(combined_img, pt_l, 3, random_color, -1); // 红色圆点
            cv::circle(combined_img, cv::Point(pt_r.x + img_left.cols, pt_r.y), 3, random_color, -1); // 红色圆点
            cv::line(combined_img, pt_l, cv::Point(pt_r.x + img_left.cols, pt_r.y), random_color, 1);
            cv::imwrite("/home/hl/project/ov2_diverg_ws/test/match_origin1.jpg", combined_img);//可视化检查左右目原图匹配点，验证没问题
            origin_points_file << pt_l.x << " " << pt_l.y << " " << pt_r.x << " " << pt_r.y << "\n";

            cv::circle(combined_img_s, frame_s.projCamToImage_s(pt_l_point3D1), 3, random_color, -1); // 红色圆点
            cv::circle(combined_img_s, cv::Point(frame_s.projCamToImage_s(pt_r_point3D1).x + img_left_s.cols, frame_s.projCamToImage_s(pt_r_point3D1).y), 3, random_color, -1); // 红色圆点
            cv::line(combined_img_s, frame_s.projCamToImage_s(pt_l_point3D1), cv::Point(frame_s.projCamToImage_s(pt_r_point3D1).x + img_left_s.cols, frame_s.projCamToImage_s(pt_r_point3D1).y), random_color, 1);
            cv::imwrite("/home/hl/project/ov2_diverg_ws/test/match_s1.jpg", combined_img_s);//可视化检查左右目原图匹配点，验证没问题
            stereo_points_file << frame_s.projCamToImage_s(pt_l_point3D1).x << " " << frame_s.projCamToImage_s(pt_l_point3D1).y << " " << frame_s.projCamToImage_s(pt_r_point3D1).x << " " << frame_s.projCamToImage_s(pt_r_point3D1).y << "\n";*/
        }
    }
//    origin_points_file.close();
//    stereo_points_file.close();
//    cv::imwrite("/home/hl/project/ov2_diverg_ws/test/match_origin_all.jpg", combined_img);//可视化检查左右目原图匹配点，验证没问题
//    cv::imwrite("/home/hl/project/ov2_diverg_ws/test/match_s_all.jpg", combined_img_s);//可视化检查左右目原图匹配点，验证没问题

    if( vstereoidx.empty() ) {
        return;
    }

    size_t nbstereo = vstereoidx.size();

    cv::Point2f left_px_proj, right_px_proj;
    float ldist, rdist;
    Eigen::Vector3d left_pt, left_pt2, right_pt2, right_pt, wpt, left_pt_1;

    int kpidx;

    int good = 0;

    // For each stereo kp
    for( size_t i = 0 ; i < nbstereo ; i++ )
    {
        kpidx = vstereoidx.at(i);

//        if( pslamstate_->bdo_stereo_rect_ ) {   //用视差求深度，暂时没用但应该也可以用
//        float disp = vkps.at(kpidx).unpx_.x - vkps.at(kpidx).runpx_.x;//应该改为左双目区坐标视差
        cv::Point2f pt_l = vkps.at(kpidx).unpx_;// 左目点
        Eigen::Vector3d pt_l_point3D((pt_l.x - pslamstate_->cxl_)/pslamstate_->fxl_, (pt_l.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化3d坐标
        Eigen::Vector3d pt_l_point3D1 = pslamstate_->R_sl.inverse() * pt_l_point3D;
        cv::Point2f pt_ls = frame_s.projCamToImage_s(pt_l_point3D1);//左目到左双目
        cv::Point2f pt_r = vkps.at(kpidx).runpx_;// 右目点 右目到右双目
        Eigen::Vector3d pt_r_point3D((pt_r.x - pslamstate_->cxl_)/pslamstate_->fxl_, (pt_r.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化3d坐标
        Eigen::Vector3d pt_r_point3D1 = pslamstate_->R_sr.inverse() * pt_r_point3D;
        cv::Point2f pt_rs = frame_s.projCamToImage_s(pt_r_point3D1);//
        float disp = pt_ls.x - pt_rs.x;//改为左双目区坐标视差

        if( disp < 0. ) {
            frame.removeStereoKeypointById(vkps.at(kpidx).lmid_);//应该只需要去除frame的
//            frame_r.removeStereoKeypointById(vkps.at(kpidx).lmid_);
            continue;
        }

//            float z = frame.pcalib_leftcam_->fx_ * frame.pcalib_rightcam_->Tcic0_.translation().norm() / fabs(disp);
        float z = frame.pcalib_leftcam_s_->fx_ * frame.pcalib_rightcam_->Tcic0_.translation().norm() / fabs(disp);//mono_stereo 视差求深度：z=f*b/d  “左右目”和“左双目右双目”的基线长度一样，所以就用frame.pcalib_rightcam_->Tcic0_可以

//        left_pt_1 << vkps.at(kpidx).unpx_.x, vkps.at(kpidx).unpx_.y, 1.;
//        left_pt_1 = z * frame.pcalib_leftcam_s_->iK_ * left_pt_1.eval();//应该改为双目区坐标
//        left_pt_1 << pt_l.x, pt_l.y, 1.;
        left_pt_1 << pt_ls.x, pt_ls.y, 1.;
        left_pt_1 = z * frame.pcalib_leftcam_s_->iK_ * left_pt_1.eval();//改为双目区相机坐标   从像素坐标转到归一化相机坐标，再乘深度
//        } else {
        // Triangulate in left cam frame
//        std::cout<<"vleftbvs.at(i) = "<<vleftbvs.at(i).transpose()<<std::endl;
//        std::cout<<"vrightbvs.at(i) = "<<vrightbvs.at(i).transpose()<<std::endl;
        left_pt = computeTriangulation(Tlr, vleftbvs.at(i), vrightbvs.at(i));//用原图坐标
        //将left_pt转换到左双目区相机坐标
//        left_pt = pslamstate_->R_sl * left_pt;  //左目到左双目= 世界到左双目*左目到世界
        left_pt = pslamstate_->T_left_lefts_.inverse() * left_pt;  //左目到左双目= 世界到左双目*左目到世界

//        std::cout<<"Tlr2 = "<< Tlr2.matrix()<<std::endl;
//        std::cout<<"vleftbvs_s.at(i) = "<<vleftbvs_s.at(i).transpose()<<std::endl;
//        std::cout<<"vrightbvs_s.at(i) = "<<vrightbvs_s.at(i).transpose()<<std::endl;
        left_pt2 = computeTriangulation(Tlr2, vleftbvs_s.at(i), vrightbvs_s.at(i));//用双目区坐标
//        }
        // 检查深度结果
//        std::cout<<"left_pt_1.z() (depth by disparity) = "<<left_pt_1.z()<<std::endl;
//        std::cout<<"left_pt.z() (depth by triangulation) = "<<left_pt.z()<<std::endl;//有问题，数值很小
//        std::cout<<"left_pt.z() (depth by triangulation_s) = "<<left_pt2.z()<<std::endl;
//        std::cout<<"depth diff between 2 methods = "<<left_pt2.z() - left_pt_1.z()<<std::endl;//用双目求的结果：视差和三角化结果一致

        // Project into right cam frame
        right_pt = Trl * left_pt;//原图坐标
        right_pt2 = Trl2 * left_pt2;//双目区坐标

//        if( left_pt2.z() < 0.1 || right_pt2.z() < 0.1 ) {
        if( left_pt2.z() < 0.1 || right_pt2.z() < 0.1 || left_pt2.z() >15 || right_pt2.z() >15) { // todo add 太远的点也删掉
            frame.removeStereoKeypointById(vkps.at(kpidx).lmid_);//应该只需要去除frame的
//            frame_r.removeStereoKeypointById(vkps.at(kpidx).lmid_);
            continue;
        }

        // Remove MP with high reprojection error
//        left_px_proj = frame.projCamToImage(left_pt);
//        right_px_proj = frame.projCamToRightImage(left_pt);
        left_px_proj = frame.projCamToImage_s(left_pt2);//mono_stereo
        right_px_proj = frame.projCamToRightImage_s(left_pt2);//mono_stereo
//        ldist = cv::norm(left_px_proj - vkps.at(kpidx).unpx_);//应该换成双目区坐标
//        rdist = cv::norm(right_px_proj - vkps.at(kpidx).runpx_);
        ldist = cv::norm(left_px_proj - pt_ls);//换成双目区坐标
        rdist = cv::norm(right_px_proj - pt_rs);

        if( ldist > pslamstate_->fmax_reproj_err_
            || rdist > pslamstate_->fmax_reproj_err_ ) {
            frame.removeStereoKeypointById(vkps.at(kpidx).lmid_);//应该只需要去除frame的
//            frame_r.removeStereoKeypointById(vkps.at(kpidx).lmid_);
//            std::cout<<"reproj_err filter------------------------"<<std::endl;
            continue;
        }

        // Project MP in world frame
        //原算法：存到左目地图
//        wpt = frame.projCamToWorld(left_pt);//左目到世界
//        pmap_->updateMapPoint(vkps.at(kpidx).lmid_, wpt, 1./left_pt.z());//存到左目地图pmap里
        wpt = frame.projCamToWorld_s(left_pt2);// mono_stereo 左双目到世界
//        wpt = left_pt2;// todo test
//        pmap_->updateMapPoint(vkps.at(kpidx).lmid_, wpt, 1./left_pt2.z());// mono_stereo 存到左目地图pmap_l里
        pmap_l_->updateMapPoint(vkps.at(kpidx).lmid_, wpt, true, 1./left_pt2.z());// mono_stereo 存到左目地图pmap_l里
//        pmap_r_->updateMapPoint(vkps.at(kpidx).lmid_, wpt, false, 1./left_pt2.z());// mono_stereo 存到右目地图pmap_r里， todo 问题：结果会不会不一样？
        //建图也要用到3d点信息，所以双目区也要地图？ (BEFORE STEREO TRIANGULATION mono_stereo) New KF nb 2d kps / 3d kps / stereokps
//        pmap_ls_->updateMapPoint_s(vkps.at(kpidx).lmid_, wpt, true, 1./left_pt.z());// mono_stereo 存到左双目地图pmap_ls里
//        pmap_rs_->updateMapPoint_s(vkps.at(kpidx).lmid_, wpt, false, 1./left_pt.z());// mono_stereo 存到左双目地图pmap_ls里
        //要存的点都是wpt吗

        good++;
    }

    if( pslamstate_->debug_ )
        std::cout << "\n \t >>> Stereo Mapping : " << good << " 3D MPs out of "
                  << nbstereo << " kps !\n";

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_TriangulateStereo");
}

void Mapper::triangulateStereo_s2_r(Frame &frame_s, Frame &frame, Frame &frame_r, const std::vector<cv::Mat> &vleftpyr_origin, const std::vector<cv::Mat> &vrightpyr_origin, const std::vector<cv::Mat> &vleftpyr_s, const std::vector<cv::Mat> &vrightpyr_s) // 其实得到了左右目的匹配关系，也可以不用双目区求深度，就用左右目求深度也可以。但是视差方法不能用了！！！
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_TriangulateStereo");

    // INIT STEREO TRIANGULATE
    std::vector<Keypoint> vkps;

    // Get the new KF stereo kps
    vkps = frame_r.getKeypointsStereo();//左目 只用左目？

    if( vkps.empty() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t >>> No kps to stereo triangulate...\n";
        return;
    }

    // Store the stereo kps along with their idx
    std::vector<int> vstereoidx;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vleftbvs, vrightbvs;//原图坐标
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vleftbvs_s, vrightbvs_s;//双目区坐标

    size_t nbkps = vkps.size();

    vleftbvs.reserve(nbkps);//原图坐标
    vrightbvs.reserve(nbkps);//原图坐标
    vleftbvs_s.reserve(nbkps);//双目区坐标
    vrightbvs_s.reserve(nbkps);//双目区原图坐标

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vwpts;
    std::vector<int> vlmids;
    vwpts.reserve(nbkps);
    vlmids.reserve(nbkps);

    // Get the extrinsic transformation
    Sophus::SE3d Tlr = frame.pcalib_rightcam_->getExtrinsic();//源代码：右到左
    // pcalib_rightcam_s_->Tc0ci_:右双目区到右目  pcalib_rightcam_->Tc0ci_:右目到左目 pcalib_leftcam_s_->Tcic0_:左目到左双目
    Sophus::SE3d Tlr2 = frame.pcalib_leftcam_s_->Tcic0_ * frame.pcalib_rightcam_->Tc0ci_ * frame.pcalib_rightcam_s_->getExtrinsic();//mono_stereo 右双目到左双目=左到左双目*右到左*右双目到右
    Sophus::SE3d Trl = Tlr.inverse();//左到右
    Sophus::SE3d Trl2 = Tlr2.inverse();//左双目到右双目

    cv::Mat img_left = vleftpyr_origin[0].clone();//用于可视化检查
    cv::Mat img_right = vrightpyr_origin[0].clone();
    cv::Mat img_left_s = vleftpyr_s[0].clone();//用于可视化检查
    cv::Mat img_right_s = vrightpyr_s[0].clone();
    // 如果图像是灰度图，转换为彩色图
    if (img_left.channels() == 1)
        cv::cvtColor(img_left, img_left, cv::COLOR_GRAY2BGR);
    if (img_right.channels() == 1)
        cv::cvtColor(img_right, img_right, cv::COLOR_GRAY2BGR);
    if (img_left_s.channels() == 1)
        cv::cvtColor(img_left_s, img_left_s, cv::COLOR_GRAY2BGR);
    if (img_right_s.channels() == 1)
        cv::cvtColor(img_right_s, img_right_s, cv::COLOR_GRAY2BGR);
    cv::Mat combined_img, combined_img_s;
    cv::hconcat(img_left, img_right, combined_img);
    cv::hconcat(img_left_s, img_right_s, combined_img_s);
    //保存结果，用于在compute_depth.cpp程序单独测试深度求解
//    cv::imwrite("/home/hl/project/ov2_diverg_ws/test/test_depth/img_left.jpg", img_left);
//    cv::imwrite("/home/hl/project/ov2_diverg_ws/test/test_depth/img_right.jpg", img_right);
//    cv::imwrite("/home/hl/project/ov2_diverg_ws/test/test_depth/img_left_s.jpg", img_left_s);
//    cv::imwrite("/home/hl/project/ov2_diverg_ws/test/test_depth/img_right_s.jpg", img_right_s);
//    std::ofstream origin_points_file("/home/hl/project/ov2_diverg_ws/test/test_depth/origin_points.txt");
//    std::ofstream stereo_points_file("/home/hl/project/ov2_diverg_ws/test/test_depth/stereo_points.txt");

    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        if( !vkps.at(i).is3d_ && vkps.at(i).is_stereo_ )
        {
//            vstereoidx.push_back(i);
//            vrightbvs.push_back(vkps.at(i).bv_);//原图相机坐标
//
//            cv::Point2f pt_r = vkps.at(i).unpx_;// 右目点
//            Eigen::Vector3d pt_r_point3D((pt_r.x - pslamstate_->cxl_)/pslamstate_->fxl_, (pt_r.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化平面3d坐标
//
//            Eigen::Vector3d pt_l_point3D1 = pslamstate_->T_left_right_ * pt_r_point3D;//右目到左目
//            vleftbvs.push_back(pt_l_point3D1.normalized());//原图相机坐标
//
//            Eigen::Vector3d pt_rs_point3D1 = pslamstate_->R_sr.inverse() * pt_r_point3D;//右目到右双目
//            pt_rs_point3D1.normalize(); // 归一化后 pt_l_point3D1 的模长为 1
//            vrightbvs_s.push_back(pt_rs_point3D1);// 双目区坐标
//
//            cv::Point2f pt_l = frame.projCamToImage(pt_l_point3D1);// 左目点
//            Eigen::Vector3d pt_l_point3D((pt_l.x - pslamstate_->cxl_)/pslamstate_->fxl_, (pt_l.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化平面3d坐标
//            Eigen::Vector3d pt_ls_point3D1 = pslamstate_->R_sl.inverse() * pt_l_point3D;//左目到左双目
//            pt_ls_point3D1.normalize(); // 归一化后 pt_r_point3D1 的模长为 1
//            vleftbvs_s.push_back(pt_ls_point3D1);//双目区坐标

            vstereoidx.push_back(i);
            vrightbvs.push_back(vkps.at(i).bv_);//原图相机坐标
            vleftbvs.push_back(vkps.at(i).rbv_);//原图相机坐标

            cv::Point2f pt_r = vkps.at(i).unpx_;// 左目点
            Eigen::Vector3d pt_r_point3D((pt_r.x - pslamstate_->cxl_)/pslamstate_->fxl_, (pt_r.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化平面3d坐标
            Eigen::Vector3d pt_r_point3D1 = pslamstate_->R_sr.inverse() * pt_r_point3D;//左目到左双目
            pt_r_point3D1.normalize(); // 归一化后 pt_l_point3D1 的模长为 1
            vrightbvs_s.push_back(pt_r_point3D1);// 双目区坐标

            cv::Point2f pt_l = vkps.at(i).runpx_;// 右目点
            Eigen::Vector3d pt_l_point3D((pt_l.x - pslamstate_->cxl_)/pslamstate_->fxl_, (pt_l.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化平面3d坐标
            Eigen::Vector3d pt_l_point3D1 = pslamstate_->R_sl.inverse() * pt_l_point3D;//右目到右双目
            pt_l_point3D1.normalize(); // 归一化后 pt_r_point3D1 的模长为 1
            vleftbvs_s.push_back(pt_l_point3D1);//双目区坐标

//            cv::Point2f test1 = frame.projCamToImage(vkps.at(i).bv_);//
//            cv::Point2f test2 = frame.projCamToImage(vkps.at(i).rbv_);//
//            std::cout<<"vkps.at(i).bv_ = "<<test1<<std::endl;
//            std::cout<<"vkps.at(i).rbv_ = "<<test2<<std::endl;
//            std::cout<<"vkps.at(i).unpx_ = "<<vkps.at(i).unpx_<<std::endl;
//            std::cout<<"vkps.at(i).runpx_ = "<<vkps.at(i).runpx_<<std::endl;//验证原图点坐标没问题
            //检查 验证左右目匹配点坐标是对的
//            cv::Scalar random_color(rand() % 256, rand() % 256, rand() % 256);
//            cv::circle(combined_img, pt_l, 3, random_color, -1); // 红色圆点
//            cv::circle(combined_img, cv::Point(pt_r.x + img_left.cols, pt_r.y), 3, random_color, -1); // 红色圆点
//            cv::line(combined_img, pt_l, cv::Point(pt_r.x + img_left.cols, pt_r.y), random_color, 1);
//            cv::imwrite("/home/hl/project/ov2_diverg_ws/test2/match_origin1.jpg", combined_img);//可视化检查左右目原图匹配点，验证没问题
//            origin_points_file << pt_l.x << " " << pt_l.y << " " << pt_r.x << " " << pt_r.y << "\n";

//            cv::circle(combined_img_s, frame_s.projCamToImage_s(pt_ls_point3D1), 3, random_color, -1); // 红色圆点
//            cv::circle(combined_img_s, cv::Point(frame_s.projCamToImage_s(pt_rs_point3D1).x + img_left_s.cols, frame_s.projCamToImage_s(pt_rs_point3D1).y), 3, random_color, -1); // 红色圆点
//            cv::line(combined_img_s, frame_s.projCamToImage_s(pt_l_point3D1), cv::Point(frame_s.projCamToImage_s(pt_rs_point3D1).x + img_left_s.cols, frame_s.projCamToImage_s(pt_rs_point3D1).y), random_color, 1);
//            cv::imwrite("/home/hl/project/ov2_diverg_ws/test2/match_s1.jpg", combined_img_s);//可视化检查左右目原图匹配点，验证没问题
//            stereo_points_file << frame_s.projCamToImage_s(pt_l_point3D1).x << " " << frame_s.projCamToImage_s(pt_l_point3D1).y << " " << frame_s.projCamToImage_s(pt_r_point3D1).x << " " << frame_s.projCamToImage_s(pt_r_point3D1).y << "\n";*/
        }
    }
//    origin_points_file.close();
//    stereo_points_file.close();
//    cv::imwrite("/home/hl/project/ov2_diverg_ws/test/match_origin_all.jpg", combined_img);//可视化检查左右目原图匹配点，验证没问题
//    cv::imwrite("/home/hl/project/ov2_diverg_ws/test/match_s_all.jpg", combined_img_s);//可视化检查左右目原图匹配点，验证没问题

    if( vstereoidx.empty() ) {
        return;
    }

    size_t nbstereo = vstereoidx.size();

    cv::Point2f left_px_proj, right_px_proj;
    float ldist, rdist;
    Eigen::Vector3d left_pt, left_pt2, right_pt2, right_pt, wpt, left_pt_1;

    int kpidx;

    int good = 0;

    // For each stereo kp
    for( size_t i = 0 ; i < nbstereo ; i++ )
    {
        kpidx = vstereoidx.at(i);

//        cv::Point2f pt_r = vkps.at(kpidx).unpx_;// 右目点
//        Eigen::Vector3d pt_r_point3D((pt_r.x - pslamstate_->cxl_)/pslamstate_->fxl_, (pt_r.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化3d坐标
//        Eigen::Vector3d pt_rs_point3D1 = pslamstate_->R_sr.inverse() * pt_r_point3D;
//        cv::Point2f pt_rs = frame_s.projCamToImage_s(pt_rs_point3D1);//右目到右双目
//
//        Eigen::Vector3d pt_l_point3D1 = pslamstate_->T_left_right_ * pt_r_point3D;//右目到左目
//        cv::Point2f pt_l = frame.projCamToImage(pt_l_point3D1);// 左目点
//        Eigen::Vector3d pt_l_point3D((pt_l.x - pslamstate_->cxl_)/pslamstate_->fxl_, (pt_l.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化3d坐标
//        Eigen::Vector3d pt_ls_point3D1 = pslamstate_->R_sl.inverse() * pt_l_point3D;
//        cv::Point2f pt_ls = frame_s.projCamToImage_s(pt_ls_point3D1);//

//        if( pslamstate_->bdo_stereo_rect_ ) {   //用视差求深度，暂时没用但应该也可以用
//        float disp = vkps.at(kpidx).unpx_.x - vkps.at(kpidx).runpx_.x;//应该改为左双目区坐标视差
        cv::Point2f pt_r = vkps.at(kpidx).unpx_;// 左目点
        Eigen::Vector3d pt_r_point3D((pt_r.x - pslamstate_->cxl_)/pslamstate_->fxl_, (pt_r.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化3d坐标
        Eigen::Vector3d pt_r_point3D1 = pslamstate_->R_sr.inverse() * pt_r_point3D;
        cv::Point2f pt_rs = frame_s.projCamToImage_s(pt_r_point3D1);//左目到左双目
        cv::Point2f pt_l = vkps.at(kpidx).runpx_;// 右目点 右目到右双目
        Eigen::Vector3d pt_l_point3D((pt_l.x - pslamstate_->cxl_)/pslamstate_->fxl_, (pt_l.y - pslamstate_->cyl_)/pslamstate_->fyl_, 1.0);//归一化3d坐标
        Eigen::Vector3d pt_l_point3D1 = pslamstate_->R_sl.inverse() * pt_l_point3D;
        cv::Point2f pt_ls = frame_s.projCamToImage_s(pt_l_point3D1);//

        float disp = pt_ls.x - pt_rs.x;//改为左双目区坐标视差

        if( disp < 0. ) {
//            frame.removeStereoKeypointById(vkps.at(kpidx).lmid_);
            frame_r.removeStereoKeypointById(vkps.at(kpidx).lmid_);//应该只需要去除frame_r的
            continue;
        }

//            float z = frame.pcalib_leftcam_->fx_ * frame.pcalib_rightcam_->Tcic0_.translation().norm() / fabs(disp);
        float z = frame.pcalib_leftcam_s_->fx_ * frame.pcalib_rightcam_->Tcic0_.translation().norm() / fabs(disp);//mono_stereo 视差求深度：z=f*b/d  “左右目”和“左双目右双目”的基线长度一样，所以就用frame.pcalib_rightcam_->Tcic0_可以

//        left_pt_1 << vkps.at(kpidx).unpx_.x, vkps.at(kpidx).unpx_.y, 1.;
//        left_pt_1 = z * frame.pcalib_leftcam_s_->iK_ * left_pt_1.eval();//应该改为双目区坐标
//        left_pt_1 << pt_l.x, pt_l.y, 1.;
        left_pt_1 << pt_ls.x, pt_ls.y, 1.;
        left_pt_1 = z * frame.pcalib_leftcam_s_->iK_ * left_pt_1.eval();//改为双目区相机坐标   从像素坐标转到归一化相机坐标，再乘深度
//        } else {
        // Triangulate in left cam frame
//        std::cout<<"vleftbvs.at(i) = "<<vleftbvs.at(i).transpose()<<std::endl;
//        std::cout<<"vrightbvs.at(i) = "<<vrightbvs.at(i).transpose()<<std::endl;
        left_pt = computeTriangulation(Tlr, vleftbvs.at(i), vrightbvs.at(i));//用原图坐标
        //将left_pt转换到左双目区相机坐标
//        left_pt = pslamstate_->R_sl * left_pt;  //左目到左双目= 世界到左双目*左目到世界
        left_pt = pslamstate_->T_left_lefts_.inverse() * left_pt;  //左目到左双目= 世界到左双目*左目到世界

//        std::cout<<"Tlr2 = "<< Tlr2.matrix()<<std::endl;
//        std::cout<<"vleftbvs_s.at(i) = "<<vleftbvs_s.at(i).transpose()<<std::endl;
//        std::cout<<"vrightbvs_s.at(i) = "<<vrightbvs_s.at(i).transpose()<<std::endl;
        left_pt2 = computeTriangulation(Tlr2, vleftbvs_s.at(i), vrightbvs_s.at(i));//用双目区坐标
//        }
        // 检查深度结果
//        std::cout<<"right info: "<<std::endl;
//        std::cout<<"left_pt_1.z() (depth by disparity) = "<<left_pt_1.z()<<std::endl;
//        std::cout<<"left_pt.z() (depth by triangulation) = "<<left_pt.z()<<std::endl;//有问题，数值很小
//        std::cout<<"left_pt.z() (depth by triangulation_s) = "<<left_pt2.z()<<std::endl;
//        std::cout<<"depth diff between 2 methods = "<<left_pt2.z() - left_pt_1.z()<<std::endl;//用双目求的结果：视差和三角化结果一致

        // Project into right cam frame
        right_pt = Trl * left_pt;//原图坐标
        right_pt2 = Trl2 * left_pt2;//双目区坐标

//        if( left_pt2.z() < 0.1 || right_pt2.z() < 0.1 ) {
        if( left_pt2.z() < 0.1 || right_pt2.z() < 0.1 || left_pt2.z() >15 || right_pt2.z() >15) { // todo add 太远的点也删掉
//            frame.removeStereoKeypointById(vkps.at(kpidx).lmid_);
            frame_r.removeStereoKeypointById(vkps.at(kpidx).lmid_);//应该只需要去除frame_r的
            continue;
        }

        // Remove MP with high reprojection error
//        left_px_proj = frame.projCamToImage(left_pt);
//        right_px_proj = frame.projCamToRightImage(left_pt);
        left_px_proj = frame.projCamToImage_s(left_pt2);//mono_stereo
        right_px_proj = frame.projCamToRightImage_s(left_pt2);//mono_stereo
//        ldist = cv::norm(left_px_proj - vkps.at(kpidx).unpx_);//应该换成双目区坐标
//        rdist = cv::norm(right_px_proj - vkps.at(kpidx).runpx_);
        ldist = cv::norm(left_px_proj - pt_ls);//换成双目区坐标
        rdist = cv::norm(right_px_proj - pt_rs);

        if( ldist > pslamstate_->fmax_reproj_err_
            || rdist > pslamstate_->fmax_reproj_err_ ) {
//            frame.removeStereoKeypointById(vkps.at(kpidx).lmid_);
            frame_r.removeStereoKeypointById(vkps.at(kpidx).lmid_);//应该只需要去除frame_r的
//            std::cout<<"reproj_err filter------------------------"<<std::endl;
            continue;
        }

        // Project MP in world frame
        //原算法：存到左目地图
//        wpt = frame.projCamToWorld(left_pt);//左目到世界
//        pmap_->updateMapPoint(vkps.at(kpidx).lmid_, wpt, 1./left_pt.z());//存到左目地图pmap里
        wpt = frame.projCamToWorld_s(left_pt2);// mono_stereo 左双目到世界
//        wpt = left_pt2;// todo test
//        pmap_->updateMapPoint(vkps.at(kpidx).lmid_, wpt, 1./left_pt2.z());// mono_stereo 存到左目地图pmap_l里
//        pmap_l_->updateMapPoint(vkps.at(kpidx).lmid_, wpt, true, 1./left_pt2.z());// mono_stereo 存到左目地图pmap_l里
        pmap_r_->updateMapPoint(vkps.at(kpidx).lmid_, wpt, false, 1./left_pt2.z());// mono_stereo 存到右目地图pmap_r里， todo 问题：结果会不会不一样？
        //建图也要用到3d点信息，所以双目区也要地图？ (BEFORE STEREO TRIANGULATION mono_stereo) New KF nb 2d kps / 3d kps / stereokps
//        pmap_ls_->updateMapPoint_s(vkps.at(kpidx).lmid_, wpt, true, 1./left_pt.z());// mono_stereo 存到左双目地图pmap_ls里
//        pmap_rs_->updateMapPoint_s(vkps.at(kpidx).lmid_, wpt, false, 1./left_pt.z());// mono_stereo 存到左双目地图pmap_ls里
        //要存的点都是wpt吗

        good++;
    }

    if( pslamstate_->debug_ )
        std::cout << "\n \t >>> Stereo Mapping : " << good << " 3D MPs out of "
                  << nbstereo << " kps !\n";

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_TriangulateStereo");
}

inline Eigen::Vector3d Mapper::computeTriangulation(const Sophus::SE3d &T, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr)
{
    // OpenGV Triangulate
    return MultiViewGeometry::triangulate(T, bvl, bvr);
}

bool Mapper::matchingToLocalMap(Frame &frame)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_MatchingToLocalMap");

    // Maximum number of MPs to track
    const size_t nmax_localplms = pslamstate_->nbmaxkps_ * 10;

    // If room for more kps, get the local map  of the oldest co KF
    // and add it to the set of MPs to search for
    auto cov_map = frame.getCovisibleKfMap();

    if( frame.set_local_mapids_.size() < nmax_localplms ) 
    {
        int kfid = cov_map.begin()->first;
        auto pkf = pmap_->getKeyframe(kfid);
        while( pkf == nullptr  && kfid > 0 && !bnewkfavailable_ ) {
            kfid--;
            pkf = pmap_->getKeyframe(kfid);
        }

        // Skip if no time
        if( bnewkfavailable_ ) {
            return false;
        }
        
        if( pkf != nullptr ) {
            frame.set_local_mapids_.insert( pkf->set_local_mapids_.begin(), pkf->set_local_mapids_.end() );
        }

        // If still far not enough, go for another round
        if( pkf->kfid_ > 0 && frame.set_local_mapids_.size() < 0.5 * nmax_localplms )
        {
            pkf = pmap_->getKeyframe(pkf->kfid_);
            while( pkf == nullptr  && kfid > 0 && !bnewkfavailable_ ) {
                kfid--;
                pkf = pmap_->getKeyframe(kfid);
            }

            // Skip if no time
            if( bnewkfavailable_ ) {
                return false;
            }
            
            if( pkf != nullptr ) {
                frame.set_local_mapids_.insert( pkf->set_local_mapids_.begin(), pkf->set_local_mapids_.end() );
            }
        }
    }

    // Skip if no time
    if( bnewkfavailable_ ) {
        return false;
    }

    if( pslamstate_->debug_ )
        std::cout << "\n \t>>> matchToLocalMap() --> Number of local MPs selected : " 
            << frame.set_local_mapids_.size() << "\n";

    // Track local map
    std::map<int,int> map_previd_newid = matchToMap(
                                            frame, pslamstate_->fmax_proj_pxdist_, 
                                            pslamstate_->fmax_desc_dist_, frame.set_local_mapids_
                                            );

    size_t nbmatches = map_previd_newid.size();

    if( pslamstate_->debug_ )
        std::cout << "\n \t>>> matchToLocalMap() --> Match To Local Map found #" 
            << nbmatches << " matches \n"; 

    // Return if no matches
    if( nbmatches == 0 ) {
        return false;
    }

    // Merge in a thread to avoid waiting for BA to finish
    // mergeMatches(frame, map_previd_newid);
    std::thread thread(&Mapper::mergeMatches, this, std::ref(frame), map_previd_newid);
    thread.detach();

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_MatchingToLocalMap");
        
    return true;
}

bool Mapper::matchingToLocalMap(Frame &frame, std::shared_ptr<MapManager>& pmap, std::shared_ptr<Frame> pframe, bool isleft)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_MatchingToLocalMap");

    // Maximum number of MPs to track
    const size_t nmax_localplms = pslamstate_->nbmaxkps_ * 10;

    // If room for more kps, get the local map  of the oldest co KF
    // and add it to the set of MPs to search for
    auto cov_map = frame.getCovisibleKfMap();

    if( frame.set_local_mapids_.size() < nmax_localplms )
    {
        int kfid = cov_map.begin()->first;
        auto pkf = pmap->getKeyframe(kfid);
        while( pkf == nullptr  && kfid > 0 && !bnewkfavailable_ ) {
            kfid--;
            pkf = pmap->getKeyframe(kfid);
        }

        // Skip if no time
        if( bnewkfavailable_ ) {
            return false;
        }

        if( pkf != nullptr ) {
            frame.set_local_mapids_.insert( pkf->set_local_mapids_.begin(), pkf->set_local_mapids_.end() );
        }

        // If still far not enough, go for another round
        if( pkf->kfid_ > 0 && frame.set_local_mapids_.size() < 0.5 * nmax_localplms )
        {
            pkf = pmap->getKeyframe(pkf->kfid_);
            while( pkf == nullptr  && kfid > 0 && !bnewkfavailable_ ) {
                kfid--;
                pkf = pmap->getKeyframe(kfid);
            }

            // Skip if no time
            if( bnewkfavailable_ ) {
                return false;
            }

            if( pkf != nullptr ) {
                frame.set_local_mapids_.insert( pkf->set_local_mapids_.begin(), pkf->set_local_mapids_.end() );
            }
        }
    }

    // Skip if no time
    if( bnewkfavailable_ ) {
        return false;
    }

    if( pslamstate_->debug_ )
        std::cout << "\n \t>>> matchToLocalMap() --> Number of local MPs selected : "
                  << frame.set_local_mapids_.size() << "\n";

    // Track local map
//    std::map<int,int> map_previd_newid = matchToMap(
//            frame, pslamstate_->fmax_proj_pxdist_,
//            pslamstate_->fmax_desc_dist_, frame.set_local_mapids_
//    );
    std::map<int,int> map_previd_newid = matchToMap(    //mono_stereo
            frame, pslamstate_->fmax_proj_pxdist_,
            pslamstate_->fmax_desc_dist_, frame.set_local_mapids_, pmap, isleft
    );

    size_t nbmatches = map_previd_newid.size();

    if( pslamstate_->debug_ )
        std::cout << "\n \t>>> matchToLocalMap() --> Match To Local Map found #"
                  << nbmatches << " matches \n";

    // Return if no matches
    if( nbmatches == 0 ) {
        return false;
    }

    // Merge in a thread to avoid waiting for BA to finish
    // mergeMatches(frame, map_previd_newid);
//    std::thread thread(&Mapper::mergeMatches, this, std::ref(frame), map_previd_newid);//不报错
//    std::thread thread(&Mapper::mergeMatches2, this, std::ref(frame), map_previd_newid, std::ref(pmap), std::ref(pframe));//mono_stereo 报错
    std::thread thread(&Mapper::mergeMatches2, this, std::ref(frame), map_previd_newid, std::ref(pmap), isleft);//mono_stereo
    thread.detach();

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_MatchingToLocalMap");

    return true;
}

void Mapper::mergeMatches(const Frame &frame, const std::map<int,int> &map_kpids_lmids)
{
    std::lock_guard<std::mutex> lock2(pmap_->optim_mutex_);

    std::lock_guard<std::mutex> lock(pmap_->map_mutex_);

    // Merge the matches
    for( const auto &ids : map_kpids_lmids )
    {
        int prevlmid = ids.first;
        int newlmid = ids.second;

        pmap_->mergeMapPoints(prevlmid, newlmid);
    }

    if( pslamstate_->debug_ )
        std::cout << "\n >>> matchToLocalMap() / mergeMatches() --> Number of merges : " 
            << map_kpids_lmids.size() << "\n";
}

void Mapper::mergeMatches2(const Frame &frame, const std::map<int,int> &map_kpids_lmids, std::shared_ptr<MapManager>& pmap, bool isleft)
{
    std::lock_guard<std::mutex> lock2(pmap->optim_mutex_);

    std::lock_guard<std::mutex> lock(pmap->map_mutex_);

    // Merge the matches
    for( const auto &ids : map_kpids_lmids )
    {
        int prevlmid = ids.first;
        int newlmid = ids.second;

//        pmap->mergeMapPoints(prevlmid, newlmid);
//        pmap->mergeMapPoints(prevlmid, newlmid, pframe);// 问题在这，使用pframe就报线程错
        pmap->mergeMapPoints(prevlmid, newlmid, isleft);// 改成这种方式，不报错

    }

    if( pslamstate_->debug_ )
        std::cout << "\n >>> matchToLocalMap() / mergeMatches() --> Number of merges : "
                  << map_kpids_lmids.size() << "\n";
}

std::map<int,int> Mapper::matchToMap(const Frame &frame, const float fmaxprojerr, const float fdistratio, std::unordered_set<int> &set_local_lmids)
{
    std::map<int,int> map_previd_newid;

    // Leave if local map is empty
    if( set_local_lmids.empty() ) {
        return map_previd_newid;
    }

    // Compute max field of view
    const float vfov = 0.5 * frame.pcalib_leftcam_->img_h_ / frame.pcalib_leftcam_->fy_;
    const float hfov = 0.5 * frame.pcalib_leftcam_->img_w_ / frame.pcalib_leftcam_->fx_;

    float maxradfov = 0.;
    if( hfov > vfov ) {
        maxradfov = std::atan(hfov);
    } else {
        maxradfov = std::atan(vfov);
    }

    const float view_th = std::cos(maxradfov);

    // Define max distance from projection
    float dmaxpxdist = fmaxprojerr;
    if( frame.nb3dkps_ < 30 ) {
        dmaxpxdist *= 2.;
    }

    std::map<int, std::vector<std::pair<int, float>>> map_kpids_vlmidsdist;

    // Go through all MP from the local map
    for( const int lmid : set_local_lmids )
    {
        if( bnewkfavailable_ ) {
            break;
        }

        if( frame.isObservingKp(lmid) ) {
            continue;
        }

        auto plm = pmap_->getMapPoint(lmid);

        if( plm == nullptr ) {
            continue;
        } else if( !plm->is3d_ || plm->desc_.empty() ) {
            continue;
        }

        Eigen::Vector3d wpt = plm->getPoint();

        //Project 3D MP into KF's image
        Eigen::Vector3d campt = frame.projWorldToCam(wpt);

        if( campt.z() < 0.1 ) {
            continue;
        }

        float view_angle = campt.z() / campt.norm();

        if( fabs(view_angle) < view_th ) {
            continue;
        }

        cv::Point2f projpx = frame.projCamToImageDist(campt);

        if( !frame.isInImage(projpx) ) {
            continue;
        }

        // Get all the kps around the MP's projection
        auto vnearkps = frame.getSurroundingKeypoints(projpx);

        // Find two best matches
        float mindist = plm->desc_.cols * fdistratio * 8.; // * 8 to get bits size
        int bestid = -1;
        int secid = -1;

        float bestdist = mindist;
        float secdist = mindist;

        std::vector<int> vkpids;
        std::vector<float> vpxdist;
        cv::Mat descs;

        for( const auto &kp : vnearkps )
        {
            if( kp.lmid_ < 0 ) {
                continue;
            }

            float pxdist = cv::norm(projpx - kp.px_);

            if( pxdist > dmaxpxdist ) {
                continue;
            }

            // Check that this kp and the MP are indeed
            // candidates for matching (by ensuring that they
            // are never both observed in a given KF)
            auto pkplm = pmap_->getMapPoint(kp.lmid_);

            if( pkplm == nullptr ) {
                pmap_->removeMapPointObs(kp.lmid_,frame.kfid_);
                continue;
            }

            if( pkplm->desc_.empty() ) {
                continue;
            }
            bool is_candidate = true;
            auto set_plmkfs = plm->getKfObsSet();
            for( const auto &kfid : pkplm->getKfObsSet() ) {
                if( set_plmkfs.count(kfid) ) {
                    is_candidate = false;
                    break;
                }
            }
            if( !is_candidate ) {
                continue;
            }

            float coprojpx = 0.;
            size_t nbcokp = 0;

            for( const auto &kfid : pkplm->getKfObsSet() ) {
                auto pcokf = pmap_->getKeyframe(kfid);
                if( pcokf != nullptr ) {
                    auto cokp = pcokf->getKeypointById(kp.lmid_);
                    if( cokp.lmid_ == kp.lmid_ ) {
                        coprojpx += cv::norm(cokp.px_ - pcokf->projWorldToImageDist(wpt));
                        nbcokp++;
                    } else {
                        pmap_->removeMapPointObs(kp.lmid_, kfid);
                    }
                } else {
                    pmap_->removeMapPointObs(kp.lmid_, kfid);
                }
            }

            if( coprojpx / nbcokp > dmaxpxdist ) {
                continue;
            }
            
            float dist = plm->computeMinDescDist(*pkplm);

            if( dist <= bestdist ) {
                secdist = bestdist; // Will stay at mindist 1st time
                secid = bestid; // Will stay at -1 1st time

                bestdist = dist;
                bestid = kp.lmid_;
            }
            else if( dist <= secdist ) {
                secdist = dist;
                secid = kp.lmid_;
            }
        }

        if( bestid != -1 && secid != -1 ) {
            if( 0.9 * secdist < bestdist ) {
                bestid = -1;
            }
        }

        if( bestid < 0 ) {
            continue;
        }

        std::pair<int, float> lmid_dist(lmid, bestdist);
        if( !map_kpids_vlmidsdist.count(bestid) ) {
            std::vector<std::pair<int, float>> v(1,lmid_dist);
            map_kpids_vlmidsdist.emplace(bestid, v);
        } else {
            map_kpids_vlmidsdist.at(bestid).push_back(lmid_dist);
        }
    }

    for( const auto &kpid_vlmidsdist : map_kpids_vlmidsdist )
    {
        int kpid = kpid_vlmidsdist.first;

        float bestdist = 1024;
        int bestlmid = -1;

        for( const auto &lmid_dist : kpid_vlmidsdist.second ) {
            if( lmid_dist.second <= bestdist ) {
                bestdist = lmid_dist.second;
                bestlmid = lmid_dist.first;
            }
        }

        if( bestlmid >= 0 ) {
            map_previd_newid.emplace(kpid, bestlmid);
        }
    }

    return map_previd_newid;
}

std::map<int,int> Mapper::matchToMap(const Frame &frame, const float fmaxprojerr, const float fdistratio, std::unordered_set<int> &set_local_lmids, std::shared_ptr<MapManager>& pmap, bool isleft)
{
    std::map<int,int> map_previd_newid;

    // Leave if local map is empty
    if( set_local_lmids.empty() ) {
        return map_previd_newid;
    }

    // Compute max field of view
    const float vfov = 0.5 * frame.pcalib_leftcam_->img_h_ / frame.pcalib_leftcam_->fy_;
    const float hfov = 0.5 * frame.pcalib_leftcam_->img_w_ / frame.pcalib_leftcam_->fx_;

    float maxradfov = 0.;
    if( hfov > vfov ) {
        maxradfov = std::atan(hfov);
    } else {
        maxradfov = std::atan(vfov);
    }

    const float view_th = std::cos(maxradfov);

    // Define max distance from projection
    float dmaxpxdist = fmaxprojerr;
    if( frame.nb3dkps_ < 30 ) {
        dmaxpxdist *= 2.;
    }

    std::map<int, std::vector<std::pair<int, float>>> map_kpids_vlmidsdist;

    // Go through all MP from the local map
    for( const int lmid : set_local_lmids )
    {
        if( bnewkfavailable_ ) {
            break;
        }

        if( frame.isObservingKp(lmid) ) {
            continue;
        }

        auto plm = pmap->getMapPoint(lmid);//

        if( plm == nullptr ) {
            continue;
        } else if( !plm->is3d_ || plm->desc_.empty() ) {
            continue;
        }

        Eigen::Vector3d wpt = plm->getPoint();

        //Project 3D MP into KF's image
//        Eigen::Vector3d campt = frame.projWorldToCam(wpt);
        Eigen::Vector3d campt;  //mono_stereo
        if(isleft){
            campt = frame.projWorldToCam(wpt);//世界到左目
        } else {
            campt = pslamstate_->T_right_left_ * frame.projWorldToCam(wpt);//世界到右目=左目到右目*世界到左目
        }

        if( campt.z() < 0.1 ) {
            continue;
        }

        float view_angle = campt.z() / campt.norm();

        if( fabs(view_angle) < view_th ) {
            continue;
        }

        cv::Point2f projpx = frame.projCamToImageDist(campt);

        if( !frame.isInImage(projpx) ) {
            continue;
        }

        // Get all the kps around the MP's projection
        auto vnearkps = frame.getSurroundingKeypoints(projpx);

        // Find two best matches
        float mindist = plm->desc_.cols * fdistratio * 8.; // * 8 to get bits size
        int bestid = -1;
        int secid = -1;

        float bestdist = mindist;
        float secdist = mindist;

        std::vector<int> vkpids;
        std::vector<float> vpxdist;
        cv::Mat descs;

        for( const auto &kp : vnearkps )
        {
            if( kp.lmid_ < 0 ) {
                continue;
            }

            float pxdist = cv::norm(projpx - kp.px_);

            if( pxdist > dmaxpxdist ) {
                continue;
            }

            // Check that this kp and the MP are indeed
            // candidates for matching (by ensuring that they
            // are never both observed in a given KF)
            auto pkplm = pmap->getMapPoint(kp.lmid_);//

            if( pkplm == nullptr ) {
                pmap->removeMapPointObs(kp.lmid_,frame.kfid_);//
                continue;
            }

            if( pkplm->desc_.empty() ) {
                continue;
            }
            bool is_candidate = true;
            auto set_plmkfs = plm->getKfObsSet();
            for( const auto &kfid : pkplm->getKfObsSet() ) {
                if( set_plmkfs.count(kfid) ) {
                    is_candidate = false;
                    break;
                }
            }
            if( !is_candidate ) {
                continue;
            }

            float coprojpx = 0.;
            size_t nbcokp = 0;

            for( const auto &kfid : pkplm->getKfObsSet() ) {
                auto pcokf = pmap->getKeyframe(kfid);//
                if( pcokf != nullptr ) {
                    auto cokp = pcokf->getKeypointById(kp.lmid_);
                    if( cokp.lmid_ == kp.lmid_ ) {
                        if(isleft){ // mono_stereo
                            coprojpx += cv::norm(cokp.px_ - pcokf->projWorldToImageDist(wpt));
                        } else {
                            coprojpx += cv::norm(cokp.px_ - pcokf->projWorldToImageDist_right(wpt));
                        }
                        nbcokp++;
                    } else {
                        pmap->removeMapPointObs(kp.lmid_, kfid);//
                    }
                } else {
                    pmap->removeMapPointObs(kp.lmid_, kfid);
                }
            }

            if( coprojpx / nbcokp > dmaxpxdist ) {
                continue;
            }

            float dist = plm->computeMinDescDist(*pkplm);

            if( dist <= bestdist ) {
                secdist = bestdist; // Will stay at mindist 1st time
                secid = bestid; // Will stay at -1 1st time

                bestdist = dist;
                bestid = kp.lmid_;
            }
            else if( dist <= secdist ) {
                secdist = dist;
                secid = kp.lmid_;
            }
        }

        if( bestid != -1 && secid != -1 ) {
            if( 0.9 * secdist < bestdist ) {
                bestid = -1;
            }
        }

        if( bestid < 0 ) {
            continue;
        }

        std::pair<int, float> lmid_dist(lmid, bestdist);
        if( !map_kpids_vlmidsdist.count(bestid) ) {
            std::vector<std::pair<int, float>> v(1,lmid_dist);
            map_kpids_vlmidsdist.emplace(bestid, v);
        } else {
            map_kpids_vlmidsdist.at(bestid).push_back(lmid_dist);
        }
    }

    for( const auto &kpid_vlmidsdist : map_kpids_vlmidsdist )
    {
        int kpid = kpid_vlmidsdist.first;

        float bestdist = 1024;
        int bestlmid = -1;

        for( const auto &lmid_dist : kpid_vlmidsdist.second ) {
            if( lmid_dist.second <= bestdist ) {
                bestdist = lmid_dist.second;
                bestlmid = lmid_dist.first;
            }
        }

        if( bestlmid >= 0 ) {
            map_previd_newid.emplace(kpid, bestlmid);
        }
    }

    return map_previd_newid;
}


void Mapper::runFullBA()
{
    bool use_robust_cost = true;
    pestimator_->poptimizer_->fullBA(use_robust_cost);
}


bool Mapper::getNewKf(Keyframe &kf)
{
    std::lock_guard<std::mutex> lock(qkf_mutex_);

    // Check if new KF is available
    if( qkfs_.empty() ) {
//        std::cout << "No new KF available ...........!\n";
        bnewkfavailable_ = false;
        return false;
    }

    // Get new KF and signal BA to stop if
    // it is still processing the previous KF
    kf = qkfs_.front();
    qkfs_.pop();

    // Setting bnewkfavailable_ to true will limit
    // the processing of the KF to triangulation and postpone
    // other costly tasks to next KF as we are running late!
    if( qkfs_.empty() ) {
        bnewkfavailable_ = false;
    } else {
        bnewkfavailable_ = true;
    }

    return true;
}

bool Mapper::getNewKf(Keyframe &kf, bool isleft)
{
    std::lock_guard<std::mutex> lock(qkf_mutex_);

//    auto &qkfs = isleft ? qkfs_l_ : qkfs_r_;// todo 需要创建两个吗？

    // Check if new KF is available
    if( qkfs_.empty() ) {
//    if( qkfs.empty() ) {
        bnewkfavailable_ = false;
        return false;
    }

    // Get new KF and signal BA to stop if
    // it is still processing the previous KF
    kf = qkfs_.front();
//    kf = qkfs.front();
    qkfs_.pop();
//    qkfs.pop();

    // Setting bnewkfavailable_ to true will limit
    // the processing of the KF to triangulation and postpone
    // other costly tasks to next KF as we are running late!
    if( qkfs_.empty() ) {
//    if( qkfs.empty() ) {
        bnewkfavailable_ = false;
    } else {
        bnewkfavailable_ = true;
    }

    return true;
}


void Mapper::addNewKf(const Keyframe &kf)
{
    std::lock_guard<std::mutex> lock(qkf_mutex_);

    qkfs_.push(kf);// todo 可以创建多个keyframe kf，但是现在选择只用一个，所有东西都塞到里面。
//    qkfs_l_.push(kf);
//    qkfs_r_.push(kf);
//    qkfs_lm_.push(kf);
//    qkfs_ls_.push(kf);
//    qkfs_rm_.push(kf);
//    qkfs_rs_.push(kf);

    bnewkfavailable_ = true;
}

void Mapper::reset()
{
    std::lock_guard<std::mutex> lock2(pmap_->optim_mutex_);// todo 只用了pmap_的锁，但是还有pmap_l_, pmap_r_等等，需不需要？
    std::lock_guard<std::mutex> lock_left(pmap_l_->optim_mutex_);
    std::lock_guard<std::mutex> lock_right(pmap_r_->optim_mutex_);
    std::lock_guard<std::mutex> lock_left_s(pmap_ls_->optim_mutex_);
    std::lock_guard<std::mutex> lock_right_s(pmap_rs_->optim_mutex_);

    bnewkfavailable_ = false;
    bwaiting_for_lc_ = false;
    bexit_required_ = false; 

    std::queue<Keyframe> empty;
    std::swap(qkfs_, empty);
}
