// switchMapKeypoint.cuh
#ifndef SWITCH_MAP_KEYPOINT_CUH
#define SWITCH_MAP_KEYPOINT_CUH

#include <vector>
#include <cuda_runtime.h>

// 定义MapPoint的数据结构
struct MapPointData {
    int lmid;       // MapPoint ID
    double x, y, z; // 3D坐标
};

// 定义Frame的数据结构
struct FrameData {
    double Twc[16];          // 4x4世界到相机的变换矩阵，按行优先存储
    double T_left_right[16]; // 左右相机之间的变换矩阵，按行优先存储
    double cos_half_fov;     // 视野的cos值
    int img_left_w;          // 图像宽度
    int img_left_h;          // 图像高度
    int img_lefts_w;         // 图像左半部分宽度
};

// 定义最大新关键点数量
#define MAX_NEW_POINTS 100000

// CUDA内核函数接口
void switchMapKeypointCUDA(
        const std::vector<MapPointData>& mapPoints,
        const FrameData& frame_r,
        const unsigned char* d_iml, // 图像数据已经在主机端上传到设备内存
        std::vector<float>& newPoints, // 输出的新关键点 (x, y) 浮点数组
        int maxNewPoints = MAX_NEW_POINTS // 预分配的最大关键点数量
);

#endif // SWITCH_MAP_KEYPOINT_CUH
