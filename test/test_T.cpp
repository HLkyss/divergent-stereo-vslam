#include <iostream>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

int main() {
    // 定义角度（单位：弧度）
//    double angle_left = 0;  // 0度测试
//    double angle_right = 0;  // 0度测试
//    double angle_left = M_PI / 6;  // 左相机旋转角度 30 度 = π/6
//    double angle_right = -M_PI / 6;  // 右相机旋转角度 -30 度 = -π/6
//    double angle_left = M_PI / 18;  // 左相机旋转角度 10 度
//    double angle_right = -M_PI / 18;  // 右相机旋转角度 -10 度
    double angle_left = M_PI / 18 * 2;  // 左相机旋转角度 20 度
    double angle_right = -M_PI / 18 * 2;  // 右相机旋转角度 -20 度

    // 定义左相机和右相机之间的平移向量（0.2m 的水平距离）
    Eigen::Vector3d translation_left_right(0.2, 0, 0);

    // 定义旋转矩阵
    Eigen::Matrix3d rotation_left, rotation_right;
    rotation_left << cos(angle_left), 0, sin(angle_left),
            0, 1, 0,
            -sin(angle_left), 0, cos(angle_left);

    rotation_right << cos(angle_right), 0, sin(angle_right),
            0, 1, 0,
            -sin(angle_right), 0, cos(angle_right);

    // 构造左相机的 4x4 齐次变换矩阵（假设左相机在原点）
    Eigen::Matrix4d Tbc0 = Eigen::Matrix4d::Identity();
    Tbc0.block<3, 3>(0, 0) = rotation_left;
    Tbc0.block<3, 1>(0, 3) = Eigen::Vector3d(0, 0, 0);  // 左相机在原点

    // 构造右相机的 4x4 齐次变换矩阵（右相机相对于 body 坐标系）
    Eigen::Matrix4d Tbc1 = Eigen::Matrix4d::Identity();
    Tbc1.block<3, 3>(0, 0) = rotation_right;
    Tbc1.block<3, 1>(0, 3) = translation_left_right;  // 右相机相对于 body 的平移

    // 输出 body_T_cam0 和 body_T_cam1
    std::cout << "Transform from body to left camera (body_T_cam0):\n" << Tbc0 << std::endl;
    std::cout << "Transform from body to right camera (body_T_cam1):\n" << Tbc1 << std::endl;

    // 计算左目到右目的变换矩阵
    Eigen::Matrix4d T_left_right = Tbc0.inverse() * Tbc1;//到这里没问题

    // 输出结果
    std::cout << "Transform from left to right camera (T_left_right):\n" << T_left_right << std::endl;


    // 定义旋转角度（单位：弧度）
//    double theta = 30.0; // 偏转角度
//    double theta = 10.0; // 偏转角度
    double theta = 20.0; // 偏转角度
    double fov = 100;
    double theta_s_rad = theta * M_PI / 180.0;//大小绝对值,转换为弧度
    double theta_m_rad = (fov/2-theta) * M_PI / 180.0;
    // 定义旋转矩阵
    Eigen::Matrix3d R_sl, R_sr, R_ml, R_mr;
    R_sl << cos(-theta_s_rad), 0, sin(-theta_s_rad),  // 左相机到左双目区
            0, 1, 0,
            -sin(-theta_s_rad), 0, cos(-theta_s_rad);

    R_sr << cos(theta_s_rad), 0, sin(theta_s_rad),  // 右相机到右双目区
            0, 1, 0,
            -sin(theta_s_rad), 0, cos(theta_s_rad);
    // 构建 Sophus::SE3d 变换矩阵
    Sophus::SE3d T_left_lefts(R_sl, Eigen::Vector3d(0, 0, 0));  // 左目到左双目区
    Sophus::SE3d T_right_rights(R_sr, Eigen::Vector3d(0, 0, 0));  // 右目到右双目区
    //输出R_sl R_sr T_left_lefts T_right_rights
    std::cout << "R_sl:\n" << R_sl << std::endl;
    std::cout << "R_sr:\n" << R_sr << std::endl;
    std::cout << "T_left_lefts:\n" << T_left_lefts.matrix() << std::endl;
    std::cout << "T_right_rights:\n" << T_right_rights.matrix() << std::endl;


    Sophus::SE3d T_left_right_sophus(T_left_right);
    // 计算左双目区到右双目区的变换
    Sophus::SE3d T_lefts_rights = T_right_rights * T_left_right_sophus * T_left_lefts.inverse();

    //输出T_right_rights T_left_right_sophus T_left_lefts
    std::cout << " T_right_rights:\n"
              << T_right_rights.matrix() << std::endl;//输出基本只有平移没有旋转，基本正确
    std::cout << "T_left_right_sophus:\n"
                << T_left_right_sophus.matrix() << std::endl;//输出基本只有平移没有旋转，基本正确
    std::cout << "T_left_lefts:\n"
                << T_left_lefts.matrix() << std::endl;//输出基本只有平移没有旋转，基本正确
    // 输出结果
    std::cout << "Transform from left stereo region to right stereo region (T_lefts_rights):\n"
              << T_lefts_rights.matrix() << std::endl;//输出基本只有平移没有旋转，基本正确

    return 0;
}

//能够根据输入的偏转角，求得body_T_cam0和body_T_cam1两组外参，并用0度偏转和配置文件结果对比以验证
//用两双目区的虚拟相机外参验证了一下，结果也是正确的