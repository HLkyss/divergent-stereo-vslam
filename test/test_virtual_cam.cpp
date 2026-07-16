#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// 定义相机内参和外参
Mat realCameraK = (Mat_<double>(3, 3) << 266.666667, 0, 320.0,
        0, 266.666667, 240,
        0, 0, 1);
Mat virtualCameraK1 = (Mat_<double>(3, 3) << 266.666667, 0, 125,
        0, 266.666667, 240,
        0, 0, 1);
Mat virtualCameraK2 = (Mat_<double>(3, 3) << 266.666667, 0, 125,
        0, 266.666667, 240,
        0, 0, 1);
//Mat R = (Mat_<double>(3, 3) << 1, 0, 0,
//        0, 1, 0,
//        0, 0, 1); // 真实相机到虚拟相机的旋转矩阵
Mat T = (Mat_<double>(3, 1) << 0, 0, 0);   // 真实相机到虚拟相机的平移向量

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
Mat generateVirtualImage(const Mat& realImage, int virtualImageWidth, int virtualImageHeight, const Mat& R, const Mat& virtualCameraK) {
    Mat virtualImage = Mat::zeros(virtualImageHeight, virtualImageWidth, realImage.type());

    for (int v_y = 0; v_y < virtualImageHeight; ++v_y) {
        for (int v_x = 0; v_x < virtualImageWidth; ++v_x) {
            // 虚拟相机的像素坐标归一化
            Point3d point3D((v_x - virtualCameraK.at<double>(0, 2)) / virtualCameraK.at<double>(0, 0),
                            (v_y - virtualCameraK.at<double>(1, 2)) / virtualCameraK.at<double>(1, 1),
                            1.0);

            // 转换到真实相机的坐标系
            Mat point3DReal = R.t() * (Mat(point3D) - T);

            // 投影到真实相机的像素坐标
            Point2d realPixel = projectToRealCamera(Point3d(point3DReal), realCameraK);

            // 检查坐标是否在真实图像范围内
            if (realPixel.x >= 0 && realPixel.x < realImage.cols &&
                realPixel.y >= 0 && realPixel.y < realImage.rows) {
                virtualImage.at<Vec3b>(v_y, v_x) = realImage.at<Vec3b>(realPixel);
            }
        }
    }
    return virtualImage;
}

int main() {
    // 加载真实相机图像
    Mat realImage = imread("/home/hl/project/ov2_diverg_ws/src/ov2slam/test/fov100_theta0_left.png");

    // 定义虚拟相机图像的尺寸
    int virtualImageWidth0 = 640;
    int virtualImageHeight0 = 480;
    int virtualImageWidth = 250;//不是640/2，而是保证fov从100变成50求出的大小：640*tan(50/2)/tan(100/2)
    int virtualImageHeight = 480;

    // 假设已知水平旋转角度（例如，30度）
    double theta0 = 0.0 * CV_PI / 180.0; // 转换为弧度
    double theta1 = 0.0 * CV_PI / 180.0; // 转换为弧度
    double theta2 = -25.0 * CV_PI / 180.0; // 转换为弧度
    double theta3 = 25.0 * CV_PI / 180.0; // 转换为弧度
    Mat R0 = getRotationMatrixY(theta0);
    Mat R1 = getRotationMatrixY(theta1);
    Mat R2 = getRotationMatrixY(theta2);
    Mat R3 = getRotationMatrixY(theta3);

    // 生成虚拟相机图像
    Mat virtualImage0 = generateVirtualImage(realImage, virtualImageWidth0, virtualImageHeight0, R0, realCameraK);
    Mat virtualImage1 = generateVirtualImage(realImage, virtualImageWidth, virtualImageHeight, R1, virtualCameraK1);
    Mat virtualImage2 = generateVirtualImage(realImage, virtualImageWidth, virtualImageHeight, R2, virtualCameraK2);
    Mat virtualImage3 = generateVirtualImage(realImage, virtualImageWidth, virtualImageHeight, R3, virtualCameraK2);

    // 显示结果
    imshow("Real Image", realImage);
    imshow("Virtual Image 0", virtualImage0);
    imshow("Virtual Image 1", virtualImage1);
    imshow("Virtual Image 2", virtualImage2);
    imshow("Virtual Image 3", virtualImage3);
    waitKey(0);
    return 0;
}
//测试单个相机图像拆分成两个虚拟相机