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

#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <queue>

#include <ros/ros.h>
#include <ros/console.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Imu.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>

#include "ov2slam.hpp"
#include "slam_params.hpp"

std::shared_ptr<ros::Publisher> g_pub_image;

class SensorsGrabber {

public:
    SensorsGrabber(SlamManager *slam): pslam_(slam) {
        std::cout << "\nSensors Grabber is created...\n";
    }

    void subLeftImage(const sensor_msgs::ImageConstPtr &image) {
        std::lock_guard<std::mutex> lock(img_mutex);
        img0_buf.push(image);
    }

    void subRightImage(const sensor_msgs::ImageConstPtr &image) {
        std::lock_guard<std::mutex> lock(img_mutex);
        img1_buf.push(image);
    }

    cv::Mat getGrayImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
    {
        // Get and prepare images
        cv_bridge::CvImageConstPtr ptr;
        try {
            ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        }
        catch(cv_bridge::Exception &e)
        {
            ROS_ERROR("\n\n\ncv_bridge exeception: %s\n\n\n", e.what());
            return cv::Mat();
        }

        return ptr->image;
    }

    // extract images with same timestamp from two topics
    // (mostly derived from Vins-Fusion: https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)
    void sync_process()
    {
        std::cout << "\nStarting the measurements reader thread!\n";
        
        while( !pslam_->bexit_required_ )
        {
            if( pslam_->pslamstate_->stereo_ || pslam_->pslamstate_->mono_stereo_)
            {
                cv::Mat image0, image1;

                std::lock_guard<std::mutex> lock(img_mutex);

                if (!img0_buf.empty() && !img1_buf.empty())
                {
                    double time0 = img0_buf.front()->header.stamp.toSec();
                    double time1 = img1_buf.front()->header.stamp.toSec();
//                    std::cout<< "time0 = " << time0 << " - time1 = " << time1 << "\n";//全是0

                    // sync tolerance
                    if(time0 < time1 - 0.015)
                    {
                        img0_buf.pop();
//                        std::cout << "\n Throw img0 -- Sync error : " << (time0 - time1) << "\n";
                    }
                    else if(time0 > time1 + 0.015)
                    {
                        img1_buf.pop();
//                        std::cout << "\n Throw img1 -- Sync error : " << (time0 - time1) << "\n";
                    }
                    else
                    {
                        image0 = getGrayImageFromMsg(img0_buf.front());
                        image1 = getGrayImageFromMsg(img1_buf.front());
                        img0_buf.pop();
                        img1_buf.pop();

                        if( !image0.empty() && !image1.empty() ) {
                            pslam_->addNewStereoImages(time0, image0, image1);
                        }
                    }
                }
            } 
            else if( pslam_->pslamstate_->mono_ ) 
            {
                cv::Mat image0;

                std::lock_guard<std::mutex> lock(img_mutex);

                if ( !img0_buf.empty() )
                {
                    double time = img0_buf.front()->header.stamp.toSec();
                    image0 = getGrayImageFromMsg(img0_buf.front());
                    img0_buf.pop();

                    if( !image0.empty()) {
                        pslam_->addNewMonoImage(time, image0);
                    }
                }
            }

            std::chrono::milliseconds dura(1);
            std::this_thread::sleep_for(dura);
        }

        std::cout << "\n Bag reader SyncProcess thread is terminating!\n";
    }

    std::queue<sensor_msgs::ImageConstPtr> img0_buf;
    std::queue<sensor_msgs::ImageConstPtr> img1_buf;
    std::mutex img_mutex;
    
    SlamManager *pslam_;
};

class KITTIGrabber {
public:
    KITTIGrabber(SlamManager *slam): pslam_(slam) {
        std::cout << "\nSensors Grabber is created...\n";
    }
    void sync_process() {
        std::cout << "\nStarting the measurements reader thread!\n";
//        std::string root_path = pslam_->pslamstate_->dataset_path_;
        std::string root_path = "/media/hl/Stuff/ubuntu_share_2/Dataset/ue_indoor3_fov100";

        load_images(root_path + "/theta10/cam0",
                    root_path + "/theta10/cam1",
                    root_path + "/time_1.txt",
                    root_path + "/time_15hz.txt",
                    image_left_,
                    image_right_,
                    image_timestamps_);
        // sleep 5s to wait for the initialization of the system
        std::this_thread::sleep_for(std::chrono::seconds(10));

        std::chrono::steady_clock::time_point start_image_time = std::chrono::steady_clock::now();
        int image_index = 0;
        while( !pslam_->bexit_required_ && image_index < image_timestamps_.size() - 1)
        {
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            double elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_image_time).count();
            double next_image_time_ms = image_timestamps_[image_index+1] * 1000 - image_timestamps_[0] * 1000;
            if (image_index > 0 && elapsed_time_ms < next_image_time_ms) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            if( pslam_->pslamstate_->stereo_ || pslam_->pslamstate_->mono_stereo_)
            {
                std::lock_guard<std::mutex> lock(img_mutex);
                // to sec
                double time = (image_timestamps_[image_index] - image_timestamps_[0]);

                cv::Mat image0 = cv::imread(image_left_[image_index], cv::IMREAD_GRAYSCALE);
                cv::Mat image1 = cv::imread(image_right_[image_index], cv::IMREAD_GRAYSCALE);
                if( !image0.empty() && !image1.empty()) {
                    pslam_->addNewStereoImages(time, image0, image1);
                } else {
                    std::cout << "Empty image at index: " << image_index << std::endl;
                }
            }
            else if( pslam_->pslamstate_->mono_ )
            {
                std::lock_guard<std::mutex> lock(img_mutex);
                // to sec
                double time = (image_timestamps_[image_index] - image_timestamps_[0]);
                cv::Mat image0 = cv::imread(image_left_[image_index], cv::IMREAD_GRAYSCALE);
                if( !image0.empty()) {
                    pslam_->addNewMonoImage(time, image0);
                } else {
                    std::cout << "Empty image at index: " << image_index << std::endl;
                }
                // publish the image
                g_pub_image->publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", image0).toImageMsg());
            }
            image_index += 1;
            std::chrono::milliseconds dura(1);
            std::this_thread::sleep_for(dura);
        }
    }

    void load_images(const std::string &path_left,
                     const std::string &path_right,
                     const std::string &path_name,
                     const std::string &path_time,
                     std::vector<std::string> &image_left,
                     std::vector<std::string> &image_right,
                     std::vector<double> &time_stamp)
    {
        std::ifstream fNames(path_name);
        std::ifstream fTimes(path_time);
        if (!fNames.is_open()) {
            std::cerr << "Failed to open the names file: " << path_name << std::endl;
            return;
        }
        if (!fTimes.is_open()) {
            std::cerr << "Failed to open the times file: " << path_time << std::endl;
            return;
        }
        time_stamp.reserve(5000);
        image_left.reserve(5000);
        image_right.reserve(5000);
        std::string name_line;
        std::string time_line;
        while (std::getline(fNames, name_line) )
        {
            if (!name_line.empty())
            {
                std::stringstream ss(name_line);
                std::string name;

                // 读取图像文件名（左目和右目的文件名相同）
                ss >> name;

                // 拼接图像路径
                std::string left_image_path = path_left + "/" + name.c_str() + ".png";
                std::string right_image_path = path_right + "/" + name.c_str() + ".png";
                image_left.push_back(left_image_path);
                image_right.push_back(right_image_path);

//                 std::cout << "左路径：" << left_image_path << std::endl;
//                 std::cout << "右路径：" << right_image_path << std::endl;
            }
        }
        while (std::getline(fTimes, time_line)){
            std::stringstream ss(time_line);
            double t;
            std::string str_t;

            // 读取图像文件名（左目和右目的文件名相同）
            ss >> str_t;

            t = std::stod(str_t);

            // 将时间戳转化为秒
            time_stamp.push_back(t / 1e9);
        }
    }

    SlamManager *pslam_;
    std::mutex img_mutex;
    std::vector<std::string> image_left_;
    std::vector<std::string> image_right_;
    std::vector<double> image_timestamps_;
};

int main(int argc, char** argv)
{
    // Init the node
    ros::init(argc, argv, "ov2slam_node");

    if(argc < 2)
    {
       std::cout << "\nUsage: rosrun ov2slam ov2slam_node parameters_files/params.yaml\n";
       return 1;
    }

    std::cout << "\nLaunching OV²SLAM...\n\n";

    ros::NodeHandle nh("~");

    // Load the parameters
    std::string parameters_file = argv[1];

    std::cout << "\nLoading parameters file : " << parameters_file << "...\n";

    const cv::FileStorage fsSettings(parameters_file.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened()) {
       std::cout << "Failed to open settings file...";
       return 1;
    } else {
        std::cout << "\nParameters file loaded...\n";
    }

    std::shared_ptr<SlamParams> pparams;
    pparams.reset( new SlamParams(fsSettings) );

    // Create the ROS Visualizer
    std::shared_ptr<RosVisualizer> prosviz;
    prosviz.reset( new RosVisualizer(nh) );

    // Setting up the SLAM Manager
    SlamManager slam(pparams, prosviz);

//    cv::namedWindow("Left-Right Concat", cv::WINDOW_NORMAL);
//    cv::namedWindow("Multi Concat", cv::WINDOW_NORMAL);
//    cv::namedWindow("Optical Flow Tracking left", cv::WINDOW_NORMAL);
//    cv::namedWindow("Optical Flow Tracking right", cv::WINDOW_NORMAL);

    // Start the SLAM thread
    std::thread slamthread(&SlamManager::run, &slam);//左右目跟踪在一起 分别跟踪左右目
//    std::thread slamthread_left(&SlamManager::run2, &slam,  true);//左右目跟踪分开，多个线程 todo 2.0 没做完，后面再做，做到这个可能可以单独用一个相机。应该不难，拆开到两个线程里，注意锁
//    std::thread slamthread_right(&SlamManager::run2, &slam,  false);

    // Create the Bag file reader & callback functions
    SensorsGrabber sb(&slam);
    // Create callbacks according to the topics set in the parameters file
    ros::Subscriber subleft = nh.subscribe(fsSettings["Camera.topic_left"], 2, &SensorsGrabber::subLeftImage, &sb);
    ros::Subscriber subright = nh.subscribe(fsSettings["Camera.topic_right"], 2, &SensorsGrabber::subRightImage, &sb);
    // Start a thread for providing new measurements to the SLAM
    std::thread sync_thread(&SensorsGrabber::sync_process, &sb);

//    KITTIGrabber sb(&slam);
//    g_pub_image = std::make_shared<ros::Publisher>(nh.advertise<sensor_msgs::Image>("image", 1000));
//    std::thread sync_thread(&KITTIGrabber::sync_process, &sb);

    // ROS Spin
    ros::spin();

    // Request Slam Manager thread to exit
    slam.bexit_required_ = true;

    // Waiting end of SLAM Manager
    while( slam.bis_on_ ) {
        std::chrono::seconds dura(1);
        std::this_thread::sleep_for(dura);
    }

    return 0;
}
