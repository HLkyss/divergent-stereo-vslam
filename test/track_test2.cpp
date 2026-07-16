//多数代码尽量从ov2源代码里提取，保证代码一致性

//跟踪：visualTracking——trackMonoStereo——kltTracking——fbKltTracking


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // std::sort
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace cv;

//cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> pbrief_;
cv::Ptr<cv::DescriptorExtractor> pbrief_;

struct Keypoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int lmid_;

    cv::Point2f px_;
    cv::Point2f unpx_;
    Eigen::Vector3d bv_;

    int scale_;
    float angle_;
    cv::Mat desc_;

    bool is3d_;

    bool is_stereo_;
    cv::Point2f rpx_;
    cv::Point2f runpx_;
    Eigen::Vector3d rbv_;

    bool is_retracked_;

    Keypoint() : lmid_(-1), scale_(0), angle_(-1.), is3d_(false), is_stereo_(false), is_retracked_(false)
    {}

    // For using kps in ordered containers
    bool operator< (const Keypoint &kp) const
    {
        return lmid_ < kp.lmid_;
    }
};

std::vector<cv::Mat> describeBRIEF(const cv::Mat &im, const std::vector<cv::Point2f> &vpts)
{
    if( vpts.empty() ) {
        // std::cout << "\nNo kps provided to function describeBRIEF() \n";
        return std::vector<cv::Mat>();
    }

    std::vector<cv::KeyPoint> vkps;
    size_t nbkps = vpts.size();
    vkps.reserve(nbkps);
    std::vector<cv::Mat> vdescs;
    vdescs.reserve(nbkps);

    cv::KeyPoint::convert(vpts, vkps);

    cv::Mat descs;

    if( pbrief_ == nullptr ) {
        pbrief_  = cv::ORB::create(500, 1., 0);
        std::cout << "\n\n=======================================================================\n";
        std::cout << " BRIEF CANNOT BE USED ACCORDING TO CMAKELISTS (Opencv Contrib not enabled) \n";
        std::cout << " ORB WILL BE USED INSTEAD!  (BUT NO ROTATION  OR SCALE INVARIANCE ENABLED) \n";
        std::cout << "\n\n=======================================================================\n\n";
    }

    // std::cout << "\nCOmputing desc for #" << vkps.size() << " kps\n";

    pbrief_->compute(im, vkps, descs);

    // std::cout << "\nDesc computed for #" << vkps.size() << " kps\n";

    if( vkps.empty() ) {
        return std::vector<cv::Mat>(nbkps, cv::Mat());
    }

    size_t k = 0;
    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        if( k < vkps.size() ) {
            if( vkps[k].pt == vpts[i] ) {
                // vdescs.push_back(descs.row(k).clone());
                vdescs.push_back(descs.row(k));
                k++;
            }
            else {
                vdescs.push_back(cv::Mat());
            }
        } else {
            vdescs.push_back(cv::Mat());
        }
    }

    assert( vdescs.size() == vpts.size() );

    // std::cout << "\n \t >>> describeBRIEF : " << vkps.size() << " kps described!\n";

    return vdescs;
}

std::vector<cv::Point2f> detectSingleScale(const cv::Mat &im, const int ncellsize,
                                           const std::vector<cv::Point2f> &vcurkps, const cv::Rect &roi)
{
    double dmaxquality_ = 0.001;

    if( im.empty() ) {
        // std::cerr << "\n No image provided to detectSingleScale() !\n";
        return std::vector<cv::Point2f>();
    }

    size_t ncols = im.cols;
    size_t nrows = im.rows;

    size_t nhalfcell = ncellsize / 4;

    size_t nhcells = nrows / ncellsize;
    size_t nwcells = ncols / ncellsize;

    size_t nbcells = nhcells * nwcells;

    std::vector<cv::Point2f> vdetectedpx;
    vdetectedpx.reserve(nbcells);

    std::vector<std::vector<bool>> voccupcells(
            nhcells+1,
            std::vector<bool>(nwcells+1, false)
    );

    cv::Mat mask = cv::Mat::ones(im.rows, im.cols, CV_32F);

    for( const auto &px : vcurkps ) {
        voccupcells[px.y / ncellsize][px.x / ncellsize] = true;
        cv::circle(mask, px, nhalfcell, cv::Scalar(0.), -1);
    }

    // std::cout << "\n Single Scale detection \n";
    // std::cout << "\n nhcells : " << nhcells << " / nwcells : " << nwcells;
    // std::cout << " / nbcells : " << nhcells * nwcells;
    // std::cout << "\n cellsize : " << ncellsize;

    size_t nboccup = 0;

    std::vector<std::vector<cv::Point2f>> vvdetectedpx(nbcells);

    std::vector<std::vector<cv::Point2f>> vvsecdetectionspx(nbcells);

    auto cvrange = cv::Range(0, nbcells);

    parallel_for_(cvrange, [&](const cv::Range& range) {
        for( int i = range.start ; i < range.end ; i++ ) {

            size_t r = floor(i / nwcells);
            size_t c = i % nwcells;

            if( voccupcells[r][c] ) {
                nboccup++;
                continue;
            }

            size_t x = c*ncellsize;
            size_t y = r*ncellsize;

            cv::Rect hroi(x,y,ncellsize,ncellsize);

            if( x+ncellsize < ncols-1 && y+ncellsize < nrows-1 ) {
                cv::Mat hmap;
                cv::Mat filtered_im;
                cv::GaussianBlur(im(hroi), filtered_im, cv::Size(3,3), 0.);
                cv::cornerMinEigenVal(filtered_im, hmap, 3, 3);

                double dminval, dmaxval;
                cv::Point minpx, maxpx;

                cv::minMaxLoc(hmap.mul(mask(hroi)), &dminval, &dmaxval, &minpx, &maxpx);
                maxpx.x += x;
                maxpx.y += y;

                if( maxpx.x < roi.x || maxpx.y < roi.y
                    || maxpx.x >= roi.x+roi.width
                    || maxpx.y >= roi.y+roi.height )
                {
                    continue;
                }

                if( dmaxval >= dmaxquality_ ) {
                    vvdetectedpx.at(i).push_back(maxpx);
                    cv::circle(mask, maxpx, nhalfcell, cv::Scalar(0.), -1);
                }

                cv::minMaxLoc(hmap.mul(mask(hroi)), &dminval, &dmaxval, &minpx, &maxpx);
                maxpx.x += x;
                maxpx.y += y;

                if( maxpx.x < roi.x || maxpx.y < roi.y
                    || maxpx.x >= roi.x+roi.width
                    || maxpx.y >= roi.y+roi.height )
                {
                    continue;
                }

                if( dmaxval >= dmaxquality_ ) {
                    vvsecdetectionspx.at(i).push_back(maxpx);
                    cv::circle(mask, maxpx, nhalfcell, cv::Scalar(0.), -1);
                }
            }
        }
    });

    for( const auto &vpx : vvdetectedpx ) {
        if( !vpx.empty() ) {
            vdetectedpx.insert(vdetectedpx.end(), vpx.begin(), vpx.end());
        }
    }

    size_t nbkps = vdetectedpx.size();

    if( nbkps+nboccup < nbcells ) {
        size_t nbsec = nbcells - (nbkps+nboccup);
        size_t k = 0;
        for( const auto &vseckp : vvsecdetectionspx ) {
            if( !vseckp.empty() ) {
                vdetectedpx.push_back(vseckp.back());
                k++;
                if( k == nbsec ) {
                    break;
                }
            }
        }
    }

    nbkps = vdetectedpx.size();

    if( nbkps < 0.33 * (nbcells - nboccup) ) {
        dmaxquality_ /= 2.;
    }
    else if( nbkps > 0.9 * (nbcells - nboccup) ) {
        dmaxquality_ *= 1.5;
    }

    // Compute Corners with Sub-Pixel Accuracy
    if( !vdetectedpx.empty() )
    {
        /// Set the need parameters to find the refined corners
        cv::Size winSize = cv::Size(3,3);
        cv::Size zeroZone = cv::Size(-1,-1);
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS +
                                                     cv::TermCriteria::MAX_ITER, 30, 0.01);

        cv::cornerSubPix(im, vdetectedpx, winSize, zeroZone, criteria);
    }

    // std::cout << "\n \t>>> Found : " << nbkps;

    return vdetectedpx;
}

int main() {

    //读取图像文件
    std::string folder_left = "/media/hl/Stuff/ubuntu_share_2/Dataset/ue_pin_fov100/theta30/cam0/";
    std::string folder_right = "/media/hl/Stuff/ubuntu_share_2/Dataset/ue_pin_fov100/theta30/cam1/";
    // 读取文件夹中所有图像文件路径
    std::vector<cv::String> files_left, files_right; // 使用 cv::String 类型的容器
    cv::glob(folder_left, files_left, false);  // 获取左图文件列表
    cv::glob(folder_right, files_right, false); // 获取右图文件列表
    // 对文件列表按文件名进行排序（如果文件名中包含数字，这样做可以确保按顺序读取）
    std::sort(files_left.begin(), files_left.end());
    std::sort(files_right.begin(), files_right.end());

    cv::Mat img_left = cv::imread( "/media/hl/Stuff/ubuntu_share_2/Dataset/ue_pin_fov100/theta30/cam0/1701691071000000000.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_right = cv::imread("/media/hl/Stuff/ubuntu_share_2/Dataset/ue_pin_fov100/theta30/cam1/1701691071000000000.png", cv::IMREAD_GRAYSCALE);
    std::vector<cv::Mat> img_left_pyr_, img_right_pyr_;

    // 遍历每一对图像
    for (size_t i = 0; i < files_left.size(); ++i) {
        // 读取图像
        cv::Mat img_left = cv::imread(files_left[i], cv::IMREAD_GRAYSCALE);
        cv::Mat img_right = cv::imread(files_right[i], cv::IMREAD_GRAYSCALE);

        if (img_left.empty() || img_right.empty()) {
            std::cerr << "图像读取失败：" << files_left[i] << " 或 " << files_right[i] << std::endl;
            continue;
        }

///////////////////////////////////////////////////

        //is_kf_req = pvisualfrontend_->visualTracking(img_left, img_right, img_left_m, img_left_s, img_right_m, img_right_s, time);
        //bool iskfreq = trackMonoStereo(iml, imr, imlm, imls, imrm, imrs, time);
        //1. preprocessImage
        //clahe
        int tilesize = 50;
        cv::Size clahe_tiles(640 / tilesize, 480 / tilesize);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, clahe_tiles);
        clahe->apply(img_left, img_left);
        clahe->apply(img_right, img_right);

        cv::buildOpticalFlowPyramid(img_left, img_left_pyr_, cv::Size(9, 9), 3);
        cv::buildOpticalFlowPyramid(img_right, img_right_pyr_, cv::Size(9, 9), 3);


        //0. createKeyframe
        //prepareFrame 可能会去除一部分点，暂时省去
        //extractKeypoints
        std::vector<cv::Point2f> vpts_l, vpts_r;
//        std::vector<Keypoint> vkps = getKeypoints();

        //describeKeypoints
        std::vector<cv::Mat> vdescs_l, vdescs_r;
        vdescs_l = describeBRIEF(img_left, vpts_l);
        vdescs_r = describeBRIEF(img_right, vpts_r);

        std::vector<cv::Point2f> vnewpts_l, vnewpts_r;
        int nmaxdist = 35;
        const int nborder = 5;
        cv::Rect roi_rect_;
        roi_rect_ = cv::Rect(cv::Point2i(nborder,nborder), cv::Point2i(640-nborder,480-nborder));
        vnewpts_l = detectSingleScale(img_left, nmaxdist, vpts_l, roi_rect_);
        vnewpts_r = detectSingleScale(img_right, nmaxdist, vpts_r, roi_rect_);

        if( !vnewpts_l.empty() ) {
            vdescs_l = describeBRIEF(img_left, vnewpts_l);
        }
        if( !vnewpts_r.empty() ) {
            vdescs_r = describeBRIEF(img_right, vnewpts_r);
        }

        //可视化出特征点
        cv::Mat img_l = img_left.clone();
        cv::Mat img_r = img_right.clone();
        if (img_l.channels() == 1)
            cv::cvtColor(img_l, img_l, cv::COLOR_GRAY2BGR);
        if (img_r.channels() == 1)
            cv::cvtColor(img_r, img_r, cv::COLOR_GRAY2BGR);
        for(auto &kp : vnewpts_l)//绿色：来自新提取
        {
            cv::circle(img_l, kp, 2, cv::Scalar(0, 255, 0), 2);
        }
        for(auto &kp : vnewpts_r)//绿色：来自新提取
        {
            cv::circle(img_r, kp, 2, cv::Scalar(0, 255, 0), 2);
        }
        for(auto &kp : vpts_l)//红色：来自pframe->getKeypoints
        {
            cv::circle(img_l, kp, 2, cv::Scalar(0, 0, 255), 2);
        }
        for(auto &kp : vpts_r)//红色：来自pframe->getKeypoints
        {
            cv::circle(img_r, kp, 2, cv::Scalar(0, 0, 255), 2);
        }
        cv::imshow("new_kps left 1", img_l);
        cv::imshow("new_kps right 1", img_r);
        cv::waitKey(1);
//到这里，每张图都提取特征点，经测试，都没有天空点！！！！！！！！！！继续参考ov2代码，一步步补充后续代码，看是哪里导致提取到了天空点


        //2. kltTracking




    }







    return 0;
}