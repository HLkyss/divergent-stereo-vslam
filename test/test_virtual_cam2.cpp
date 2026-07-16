#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;



// 定义生成绕 y 轴旋转的矩阵函数
Mat getRotationMatrixY(double theta) {
    double cosTheta = cos(theta);
    double sinTheta = sin(theta);

    Mat R = (Mat_<double>(3, 3) << cosTheta, 0, sinTheta,
            0,       1, 0,
            -sinTheta, 0, cosTheta);
    return R;
}

// 计算3D点在真实相机图像上的像素坐标
Point2d projectToRealCamera(const Point3d& point3D, const Mat& realCameraK) {
    Mat point3DHomogeneous = (Mat_<double>(3, 1) << point3D.x, point3D.y, point3D.z);
    Mat pixel = realCameraK * point3DHomogeneous;
    return Point2d(pixel.at<double>(0, 0) / pixel.at<double>(2, 0),
                   pixel.at<double>(1, 0) / pixel.at<double>(2, 0));
}

// 反投影虚拟相机像素到归一化平面并转换为真实相机上的像素
Mat generateVirtualImage(const Mat& realImage, int virtualImageWidth, int virtualImageHeight, const Mat& R, const Mat& virtualCameraK, const Mat& realCameraK, const Mat& T) {
    //输出传入的参数
    std::cout<<"realImage:"<<realImage.size()<<std::endl;
    std::cout<<"virtualImageWidth:"<<virtualImageWidth<<std::endl;
    std::cout<<"virtualImageHeight:"<<virtualImageHeight<<std::endl;
    std::cout<<"R:"<<R<<std::endl;
    std::cout<<"virtualCameraK:"<<virtualCameraK<<std::endl;
    std::cout<<"realCameraK:"<<realCameraK<<std::endl;
    std::cout<<"T:"<<T<<std::endl;

    Mat virtualImage = Mat::zeros(virtualImageHeight, virtualImageWidth, realImage.type());

    int count_=0;
    for (int v_y = 0; v_y < virtualImageHeight; ++v_y) {
        for (int v_x = 0; v_x < virtualImageWidth; ++v_x) {
            if(count_ > 148390){
                cout<<"ok"<<endl;
            }

            // 虚拟相机的像素坐标归一化
            Point3d point3D((v_x - virtualCameraK.at<double>(0, 2)) / virtualCameraK.at<double>(0, 0),
                            (v_y - virtualCameraK.at<double>(1, 2)) / virtualCameraK.at<double>(1, 1),
                            1.0);
            cout<<"point3D:"<<point3D<<endl;

            // 转换到真实相机的坐标系
//            Mat point3DReal = R.t() * (Mat(point3D) - T);//R 左双目到左目
            Mat point3DReal = R * (Mat(point3D) - T);//R 左双目到左目
            cout<<"point3DReal:"<<point3DReal<<endl;

            // 投影到真实相机的像素坐标
            Point2d realPixel = projectToRealCamera(Point3d(point3DReal), realCameraK);
            cout<<"realPixel:"<<realPixel<<endl;

            // 检查坐标是否在真实图像范围内
            if (realPixel.x >= 0 && realPixel.x < realImage.cols &&
                realPixel.y >= 0 && realPixel.y < realImage.rows) {
                virtualImage.at<Vec3b>(v_y, v_x) = realImage.at<Vec3b>(realPixel);
            }
            count_++;
            std::cout<<"count = "<<count_<<std::endl;
        }
    }
    return virtualImage;
}

int main() {
    // 加载真实相机图像
    Mat realImage1 = imread("/home/hl/project/ov2_diverg_ws/src/ov2slam/test/fov100_theta30_left.png");
    Mat realImage2 = imread("/home/hl/project/ov2_diverg_ws/src/ov2slam/test/fov100_theta30_right.png");

    int virtualImageWidth0 = 640;
    int virtualImageHeight0 = 480;
    double theta = 30.0; // 偏转角度
    double fov = 100;//看导出的仿真数据集设置的参数，其实不是正好100度，有点误差的。因此后面的计算也有点误差

    float fov_rad = fov * CV_PI / 180.0;
//    float fx = virtualImageWidth0/2/tan(fov_rad/2);
    float fx =  266.666667;
    float fy =  266.666667;
    //单双目区大小
    float angle_s = 2*(fov/2-theta);
    float angle_m = fov-angle_s;
    float angle_s_rad = angle_s * CV_PI / 180.0;
    float angle_m_rad = angle_m * CV_PI / 180.0;
    // 定义虚拟相机图像的尺寸
    int virtualImageWidth_s = int(virtualImageWidth0*tan(angle_s_rad/2)/tan(fov_rad/2));//不是640/2，而是保证fov从100变成40求出的大小：640*tan(40/2)/tan(100/2)
//    int virtualImageWidth_s = 195;//不是640/2，而是保证fov从100变成40求出的大小：640*tan(40/2)/tan(100/2)
    int virtualImageHeight_s = 480;
    int virtualImageWidth_m = int(virtualImageWidth0*tan(angle_m_rad/2)/tan(fov_rad/2));//不是640/2，而是保证fov从100变成60求出的大小：640*tan(60/2)/tan(100/2)
//    int virtualImageWidth_m = 310;//不是640/2，而是保证fov从100变成60求出的大小：640*tan(60/2)/tan(100/2)
    int virtualImageHeight_m = 480;

    // 定义相机内参和外参
    Mat realCameraK = (Mat_<double>(3, 3) << fx, 0, virtualImageWidth0/2,
            0, fy, virtualImageHeight0/2,
            0, 0, 1);
    Mat virtualCameraK_s = (Mat_<double>(3, 3) << fx, 0, virtualImageWidth_s/2,
            0, fy, virtualImageHeight_s/2,
            0, 0, 1);
    Mat virtualCameraK_m = (Mat_<double>(3, 3) << fx, 0, virtualImageWidth_m/2,
            0, fy, virtualImageHeight_m/2,
            0, 0, 1);
    Mat T = (Mat_<double>(3, 1) << 0, 0, 0);   // 真实相机到虚拟相机的平移向量

    //朝向(分别相对于真实左目和真实右目)
    double theta_s = theta * CV_PI / 180.0;//大小绝对值,转换为弧度
//    double theta_m = (angle_m/2-fov/2+angle_s) * CV_PI / 180.0;//不是最简式 但是懒得化简了
    double theta_m = (fov/2-theta) * CV_PI / 180.0;//最简式
//    Mat R_sl = getRotationMatrixY(-1*theta_s);//负 向右转
//    Mat R_sr = getRotationMatrixY(theta_s);
//    Mat R_ml = getRotationMatrixY(theta_m);
//    Mat R_mr = getRotationMatrixY(-1*theta_m);
    Mat R_sl = getRotationMatrixY(theta_s);//负 向右转
    Mat R_sr = getRotationMatrixY(-1*theta_s);
    Mat R_ml = getRotationMatrixY(-1*theta_m);
    Mat R_mr = getRotationMatrixY(theta_m);

    // 生成虚拟相机图像
    Mat virtualImage_ml = generateVirtualImage(realImage1, virtualImageWidth_m, virtualImageHeight_m, R_ml, virtualCameraK_m, realCameraK, T);
    Mat virtualImage_sl = generateVirtualImage(realImage1, virtualImageWidth_s, virtualImageHeight_s, R_sl, virtualCameraK_s, realCameraK, T);
    Mat virtualImage_sr = generateVirtualImage(realImage2, virtualImageWidth_s, virtualImageHeight_s, R_sr, virtualCameraK_s, realCameraK, T);
    Mat virtualImage_mr = generateVirtualImage(realImage2, virtualImageWidth_m, virtualImageHeight_m, R_mr, virtualCameraK_m, realCameraK, T);

    // 显示结果
//    imshow("Real Image left", realImage1);
//    imshow("Real Image right", realImage2);
//    imshow("virtualImage_ml", virtualImage_ml);
//    imshow("virtualImage_sl", virtualImage_sl);
//    imshow("virtualImage_sr", virtualImage_sr);
//    imshow("virtualImage_mr", virtualImage_mr);
//    waitKey(0);

    cv::Mat leftRightConcat;
    cv::hconcat(realImage1, realImage2, leftRightConcat);// 左右拼接 iml 和 imr

    std::vector<cv::Mat> imagesToConcat = {virtualImage_ml, virtualImage_sl, virtualImage_sr, virtualImage_mr};
    cv::Mat multiConcat;
    cv::hconcat(imagesToConcat, multiConcat);// 依次拼接 iml_m, iml_s, imr_s, imr_m

    // 可视化拼接结果
    cv::imshow("Left-Right Concat", leftRightConcat);
    cv::imshow("Multi Concat", multiConcat);
    cv::waitKey(0);
    return 0;
}
//测试 一定发散角度下的双目相机图像 拆分成4个虚拟相机