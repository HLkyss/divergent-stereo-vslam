#include <iostream>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

int main() {

    Eigen::Matrix4d T_cn_cnm1, T_cn_cnm1_no;
    T_cn_cnm1 << 0.8574242416690883, -0.001221710709984037, -0.5146087613139982, -0.3359033451935533,
            -0.0014848713480364012, 0.9999871456008115, -0.004848070772877612, 0.000772930806475098,
            0.514608069267541, 0.004920981211116322, 0.8574114059123842, -0.09003983591978837,
            0.0, 0.0, 0.0, 1.0;
//    T_cn_cnm1_no << 0.8574242416690883, -0.001221710709984037, -0.5146087613139982, -0.35,
//            -0.0014848713480364012, 0.9999871456008115, -0.004848070772877612, 0,
//            0.514608069267541, 0.004920981211116322, 0.8574114059123842, 0,
//            0.0, 0.0, 0.0, 1.0;

    // 提取平移向量（矩阵的最后一列前三个元素）
    Eigen::Matrix3d rotation = T_cn_cnm1.block<3, 3>(0, 0);
    Eigen::Vector3d translation = T_cn_cnm1.block<3, 1>(0, 3);
    Sophus::SE3d T_cn_cnm1_sophus(rotation, translation);
//    Eigen::Vector3d translation_no = T_cn_cnm1_no.block<3, 1>(0, 3);

//    Sophus::SE3d fixed_Twc_test(
////                Sophus::SO3d::rotY(30 * M_PI / 180), // 例如绕Z轴旋转30度
//            Sophus::SO3d::rotY(-1 * -30 * M_PI / 180), // 例如绕Z轴旋转30度
////            Eigen::Vector3d(0.4, 0.0, 0.0)       // 固定平移向量
//            -1 * translation_no       // 固定平移向量
//    );

//////////////////////////////
//    //验证固定变换矩阵：
//    // 定义一个点 (x, y, z)，这里假设点的坐标为 (1, 2, 3)
//    Eigen::Vector3d point(1.0, 2.0, 3.0);
//    // 将点转换为齐次坐标 (x, y, z, 1)
//    Eigen::Vector4d point_homogeneous;
//    point_homogeneous << point, 1.0;
//    // 将 fixed_Twc 转换为 4x4 矩阵
//    Eigen::Matrix4d T_wc = fixed_Twc_test.matrix();
//    // 合成两个变换矩阵 (T_cn_cnm1 * T_wc)
//    Eigen::Matrix4d T_combined = T_cn_cnm1_no * T_wc;
//    // 变换后的点 (T_combined * point_homogeneous)
//    Eigen::Vector4d transformed_point = T_combined * point_homogeneous;
//    // 输出变换后的点
//    std::cout << "Transformed Point: \n" << transformed_point.head<3>() << std::endl;
//////////////////////////////

    Sophus::SE3d T_b_cn(
            Sophus::SO3d::rotY(-1 * -15 * M_PI / 180), // 例如绕Z轴旋转30度
            -0.5 * translation       // 固定平移向量
//            Eigen::Vector3d(0.4, 0.0, 0.0)       // 固定平移向量
    );

//    Eigen::Matrix4d T_b_cn = fixed_Twc.matrix();

    Sophus::SE3d T_b_cnm1 = T_b_cn * T_cn_cnm1_sophus;

    std::cout << "T_b_cn matrix: \n" << T_b_cn.matrix() << std::endl;
    std::cout << "T_b_cnm1 matrix: \n" << T_b_cnm1.matrix() << std::endl;

    ////////


    return 0;
}
