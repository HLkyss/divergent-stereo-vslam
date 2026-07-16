//来自triangulateStereo_s2。已可视化验证左右目和左右双目匹配点正确，从triangulateStereo_s2函数导出第一帧图像的匹配点坐标，在这里只测试深度计算部分。

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

struct PointPair {
    cv::Point2f pt1; // 左目点
    cv::Point2f pt2; // 右目点
};

// 读取点对数据
std::vector<PointPair> loadPoints(const std::string &file_path) {
    std::vector<PointPair> points;
    std::ifstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return points;
    }

    float x1, y1, x2, y2;
    while (file >> x1 >> y1 >> x2 >> y2) {
        points.push_back({cv::Point2f(x1, y1), cv::Point2f(x2, y2)});
    }
    return points;
}

Eigen::Vector3d computeTriangulation(const Sophus::SE3d &Tlr, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr)   //原函数
{
    std::vector<cv::Point2f> lpt, rpt;
    lpt.push_back( cv::Point2f(bvl.x()/bvl.z(), bvl.y()/bvl.z()) );
    rpt.push_back( cv::Point2f(bvr.x()/bvr.z(), bvr.y()/bvr.z()) );

    cv::Matx34f P0 = cv::Matx34f(1, 0, 0, 0,
                                 0, 1, 0, 0,
                                 0, 0, 1, 0);

    Sophus::SE3d Tcw = Tlr.inverse();
    Eigen::Matrix3d R = Tcw.rotationMatrix();
    Eigen::Vector3d t = Tcw.translation();

    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));

    cv::Mat campt;
    cv::triangulatePoints(P0, P1, lpt, rpt, campt);

    if( campt.col(0).at<float>(3) != 1. ) {
        campt.col(0) /= campt.col(0).at<float>(3);
    }

    Eigen::Vector3d pt(
            campt.col(0).at<float>(0),
            campt.col(0).at<float>(1),
            campt.col(0).at<float>(2)
    );

    return pt;
}

// 输入：
// T_lr: 左相机到右相机的变换，包含旋转矩阵和平移向量
// pt_l: 左相机的归一化平面坐标 (x, y, 1)
// pt_r: 右相机的归一化平面坐标 (x, y, 1)
// 输出：3D坐标
Eigen::Vector3d triangulate(const Eigen::Vector3d& left_point, const Eigen::Vector3d& right_point, const Sophus::SE3d& Tlr) {
    // 获取左相机的内外参数
    Eigen::Matrix3d Rl = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tl(0, 0, 0); // 假设左相机的平移为零

    // 获取右相机的内外参数
    Eigen::Matrix3d Rr = Tlr.rotationMatrix();
    Eigen::Vector3d tr = Tlr.translation();

    // 构造左相机和右相机的投影矩阵
    Eigen::Matrix<double, 3, 4> P_left;
    P_left.block<3, 3>(0, 0) = Rl;
    P_left.col(3) = tl;

    Eigen::Matrix<double, 3, 4> P_right;
    P_right.block<3, 3>(0, 0) = Rr;
    P_right.col(3) = tr;

    // 构建矩阵 A 来解这个线性系统 A * X = 0
    Eigen::Matrix<double, 4, 4> A;
    A.row(0) = left_point.x() * P_left.row(2) - P_left.row(0);
    A.row(1) = left_point.y() * P_left.row(2) - P_left.row(1);
    A.row(2) = right_point.x() * P_right.row(2) - P_right.row(0);
    A.row(3) = right_point.y() * P_right.row(2) - P_right.row(1);

    // 通过 SVD 分解解这个线性方程组，得到三维坐标 X
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Vector4d X = svd.matrixV().col(3); // 最小奇异值对应的右奇异向量

    // 得到 3D 坐标（将齐次坐标除以 w）
    Eigen::Vector3d world_point = X.head<3>() / X(3);

    return world_point;
}

// 主程序
int main() {
    // 文件路径
    std::string left_image_path     = "/home/hl/project/ov2_diverg_ws/test/test_depth/img_left.jpg";
    std::string right_image_path    = "/home/hl/project/ov2_diverg_ws/test/test_depth/img_right.jpg";
    std::string left_s_image_path   = "/home/hl/project/ov2_diverg_ws/test/test_depth/img_left_s.jpg";
    std::string right_s_image_path  = "/home/hl/project/ov2_diverg_ws/test/test_depth/img_right_s.jpg";
    std::string origin_points_path  = "/home/hl/project/ov2_diverg_ws/test/test_depth/origin_points.txt";
    std::string stereo_points_path  = "/home/hl/project/ov2_diverg_ws/test/test_depth/stereo_points.txt";

    // 载入图像
    cv::Mat img_left = cv::imread(left_image_path);
    cv::Mat img_right = cv::imread(right_image_path);
    cv::Mat img_left_s = cv::imread(left_s_image_path);
    cv::Mat img_right_s = cv::imread(right_s_image_path);

    if (img_left.empty() || img_right.empty() || img_left_s.empty() || img_right_s.empty()) {
        std::cerr << "无法载入图像文件，请检查路径是否正确。" << std::endl;
        return -1;
    }

    // 拼接原图和双目区图像
    cv::Mat combined_img, combined_img_s;
    cv::hconcat(img_left, img_right, combined_img);
    cv::hconcat(img_left_s, img_right_s, combined_img_s);

    Eigen::Matrix3d K, iK, K_s, iK_s;
    K << 266.667,   0,      320,
            0,    266.667,  240,
            0,        0,      1;
    K_s << 266.667,   0,      97.5,
            0,    266.667,  240,
            0,        0,      1;
    iK = K.inverse();
    iK_s = K_s.inverse();

    Eigen::Matrix3d R, R_s, R_sl, R_sr, R_ls_l;
    Eigen::Vector3d t, t_s, t_ls_l;
    double theta = 30 * M_PI / 180.0;
//    R <<  0.5,       -0,        -0.866026,
//            0,         1,        -0,
//            0.866026,       0,         0.5;
//    R <<    sin(theta), 0, -1*cos(theta),
//            0,          1, 0,
//            cos(theta), 0, sin(theta);
//    t << 0.173205, 0, 0.1;
    R <<    cos(2*theta), 0., -sin(2*theta),
            0., 1., 0.,
            sin(2*theta), 0., cos(2*theta);
    t << 0.173205, 0, 0.1;
    Sophus::SE3d Tlr(R, t);
    std::cout<< "Tlr = " << Tlr.matrix() << std::endl;
    R_s = Eigen::Matrix3d::Identity();
    t_s << 0.2, 0, 0;
    Sophus::SE3d Tlr2(R_s, t_s);
    std::cout<< "Tlr2 = " << Tlr2.matrix() << std::endl;
    R_sl << cos(-1*theta), 0, sin(-1*theta),//左双目区虚拟相机到左相机
            0,          1, 0,
            -sin(-1*theta), 0, cos(-1*theta);
    R_sr << cos(theta), 0, sin(theta),//右双目区虚拟相机到右相机
            0,          1, 0,
            -sin(theta), 0, cos(theta);
//    R_ls_l << cos(theta), -sin(theta), 0,
//            sin(theta), cos(theta), 0,
//            0, 0, 1;
    t_ls_l << 0, 0, 0;
//    Sophus::SE3d Tls_l(R_ls_l, t_ls_l);
    Sophus::SE3d Tls_l(R_sl, t_ls_l);
    Tls_l = Tls_l.inverse();

    // 载入点对数据
    auto origin_points = loadPoints(origin_points_path);
    auto stereo_points = loadPoints(stereo_points_path);
    std::vector<float> disparity;
    std::vector<cv::Point2f> pt_l_vector, pt_r_vector, pt_ls_vector, pt_rs_vector;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vleftbvs, vrightbvs;
    std::vector<Eigen::Vector3d> vleftbvs2, vrightbvs2;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vleftbvs_s, vrightbvs_s;
    // 可视化原图匹配点
    for (const auto &point : origin_points) {
        cv::Scalar random_color(rand() % 256, rand() % 256, rand() % 256);
        cv::Point2f pt_l = point.pt1;
        cv::Point2f pt_r = point.pt2;
        cv::circle(combined_img, pt_l, 3, random_color, -1); // 左目点
        cv::circle(combined_img, cv::Point(pt_r.x + img_left.cols, pt_r.y), 3, random_color, -1); // 右目点
        cv::line(combined_img, pt_l, cv::Point(pt_r.x + img_left.cols, pt_r.y), random_color, 1);
        Eigen::Vector3d hunpx(pt_l.x, pt_l.y, 1.0);
        Eigen::Vector3d hrunpx(pt_r.x, pt_r.y, 1.0);
        Eigen::Vector3d pt_l_bv, pt_r_bv;
        pt_l_bv = K.inverse() * hunpx;
        pt_l_bv.normalize();
        pt_r_bv = K.inverse() * hrunpx;
        pt_r_bv.normalize();
        vleftbvs.push_back(pt_l_bv);
        vrightbvs.push_back(pt_r_bv);
        vleftbvs2.push_back(hunpx);
        vrightbvs2.push_back(hrunpx);

        Eigen::Vector3d pt_l_point3D((pt_l.x - K(0, 2))/K(0, 0), (pt_l.y - K(1, 2))/K(1, 1), 1.0);//归一化平面3d坐标
        Eigen::Vector3d pt_l_point3D1 = R_sl * pt_l_point3D;//左目到左双目
        pt_l_point3D.normalize(); // 归一化后 pt_l_point3D1 的模长为 1
        pt_l_point3D1.normalize(); // 归一化后 pt_l_point3D1 的模长为 1
//        std::cout<<"pt_l_bv = "<<pt_l_bv<<std::endl;
//        std::cout<<"pt_l_point3D = "<<pt_l_point3D<<std::endl;// same
        vleftbvs_s.push_back(pt_l_point3D1);

        Eigen::Vector3d pt_r_point3D((pt_r.x - K(0, 2))/K(0, 0), (pt_r.y - K(1, 2))/K(1, 1), 1.0);//归一化平面3d坐标
        Eigen::Vector3d pt_r_point3D1 = R_sr * pt_r_point3D;//右目到左相机
        pt_r_point3D1.normalize(); // 归一化后 pt_r_point3D1 的模长为 1
        vrightbvs_s.push_back(pt_r_point3D1);
    }

    // 可视化双目区匹配点
    for (const auto &point : stereo_points) {
        cv::Scalar random_color(rand() % 256, rand() % 256, rand() % 256);
        cv::Point2f pt_ls = point.pt1;
        cv::Point2f pt_rs = point.pt2;
        cv::circle(combined_img_s, pt_ls, 3, random_color, -1); // 左双目区点
        cv::circle(combined_img_s, cv::Point(pt_rs.x + img_left_s.cols, pt_rs.y), 3, random_color, -1); // 右双目区点
        cv::line(combined_img_s, pt_ls, cv::Point(pt_rs.x + img_left_s.cols, pt_rs.y), random_color, 1);
        disparity.push_back(pt_ls.x - pt_rs.x);//双目区坐标视差
        pt_ls_vector.push_back(pt_ls);
        pt_rs_vector.push_back(pt_rs);

//        Eigen::Vector3d pt_ls_point3D((pt_ls.x - K_s(0, 2))/K_s(0, 0), (pt_ls.y - K_s(1, 2))/K_s(1, 1), 1.0);//归一化平面3d坐标
//        Eigen::Vector3d pt_ls_point3D1 = R_sl * pt_ls_point3D;//左目到左双目
//        pt_ls_point3D1.normalize(); // 归一化后 pt_l_point3D1 的模长为 1
//        vleftbvs_s.push_back(pt_ls_point3D1);
//
//        Eigen::Vector3d pt_rs_point3D((pt_rs.x - K_s(0, 2))/K_s(0, 0), (pt_rs.y - K_s(1, 2))/K_s(1, 1), 1.0);//归一化平面3d坐标
//        Eigen::Vector3d pt_rs_point3D1 = R_sr * pt_rs_point3D;//右目到左相机
//        pt_rs_point3D1.normalize(); // 归一化后 pt_r_point3D1 的模长为 1
//        vrightbvs_s.push_back(pt_rs_point3D1);
    }

    // 显示结果
    cv::imshow("Original Image Matches", combined_img);
    cv::imshow("Stereo Image Matches", combined_img_s);
//    cv::imwrite("/home/hl/project/ov2_diverg_ws/test/match_visualization_origin.jpg", combined_img);
//    cv::imwrite("/home/hl/project/ov2_diverg_ws/test/match_visualization_stereo.jpg", combined_img_s);



    // 深度测试
    for (int i = 0; i < disparity.size(); i++) {
        // 1. 双目区视差
        float z = 266.666667 * 0.2 / fabs(disparity[i]);
//        std::cout << "disparity[" << i << "] = " << disparity[i] << std::endl;
        Eigen::Vector3d left_pt_1, left_pt, left_pt2;
//        left_pt_1 << pt_l.x, pt_l.y, 1.;
        left_pt_1 << stereo_points[i].pt1.x, stereo_points[i].pt1.y, 1.;
        left_pt_1 = z * iK_s * left_pt_1.eval();
        std::cout<<"left_pt_1.z() (depth by disparity) = "<<left_pt_1.z()<<", left_pt_1 = "<<left_pt_1.transpose()<<std::endl;

        // 2. 双目区三角化
        left_pt2 = computeTriangulation(Tlr2, vleftbvs_s.at(i), vrightbvs_s.at(i));
        std::cout<<"vleftbvs_s.at(i) = "<<vleftbvs_s.at(i).transpose()<<std::endl;
        std::cout<<"vrightbvs_s.at(i) = "<<vrightbvs_s.at(i).transpose()<<std::endl;
        std::cout<<"left_pt2.z() (depth by triangulation_s) = "<<left_pt2.z()<<", left_pt2 = "<<left_pt2.transpose()<<std::endl;

        // 3. 原图坐标三角化
//        left_pt = computeTriangulation(Tlr, vleftbvs.at(i), vrightbvs.at(i));
        left_pt = computeTriangulation(Tlr.inverse(), vleftbvs.at(i), vrightbvs.at(i));// todo 加了.inverse()
        std::cout<<"vleftbvs.at(i) = "<<vleftbvs.at(i).transpose()<<std::endl;
        std::cout<<"vrightbvs.at(i) = "<<vrightbvs.at(i).transpose()<<std::endl;
        std::cout<<"left_pt.z() (depth by triangulation) = "<<left_pt.z()<<", left_pt = "<<left_pt.transpose()<<std::endl;
//        std::cout<<"Tls_l = "<<Tls_l.matrix()<<std::endl;
//        left_pt = Tls_l * left_pt;    //左目到左双目= 世界到左双目*左目到世界
        left_pt = Tls_l.inverse() * left_pt;  //左目到左双目= 世界到左双目*左目到世界  todo 加了.inverse()
        std::cout<<"left_pt.z() (depth by triangulation) = "<<left_pt.z()<<", left_pt = "<<left_pt.transpose()<<std::endl;

        //4. 自己写一个三角化函数，用归一化坐标计算
//        left_pt = triangulate(vleftbvs2.at(i), vrightbvs2.at(i), Tlr);
//        left_pt = Tls_l * left_pt;  //左目到左双目= 世界到左双目*左目到世界
//        std::cout<<"left_pt.z() (depth by triangulation) = "<<left_pt.z()<<", left_pt = "<<left_pt.transpose()<<std::endl;


    }






    cv::waitKey(0);

    return 0;
}
