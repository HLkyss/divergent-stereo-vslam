// switchMapKeypoint.cu
#include "/home/hl/project/ov2_diverg_ws_2map1traj/src/ov2slam/include/switchMapKeypoint.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// 简化的4x4矩阵结构
struct Matrix4x4 {
    double m[16];
};

// CUDA设备端的投影函数（简化版）
__device__ float3 projectToImage(double x, double y, double z) {
    // 简单的透视投影，假设相机内参为单位矩阵
    // 实际应用中需要根据相机内参进行调整
    return make_float3(static_cast<float>(x / z), static_cast<float>(y / z), static_cast<float>(z));
}

// CUDA内核函数
__global__ void switchMapKeypointKernel(
        const MapPointData* mapPoints,
        int numPoints,
        Matrix4x4 Twc_r,
        Matrix4x4 T_left_right,
        double cos_half_fov,
        int img_left_w,
        int img_left_h,
        int img_lefts_w,
        const unsigned char* iml,
        int iml_stride,
        float* d_newPoints,
        int* d_newPointsCount
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    // 获取当前MapPoint
    MapPointData mp = mapPoints[idx];

    // 将MapPoint从世界坐标系转换到相机坐标系
    double point_world[4] = { mp.x, mp.y, mp.z, 1.0 };
    double point_cam[4] = {0.0, 0.0, 0.0, 0.0};
    for(int i = 0; i < 4; ++i){
        for(int j = 0; j < 4; ++j){
            point_cam[i] += Twc_r.m[i * 4 + j] * point_world[j];
        }
    }

    // 计算点到相机的向量
    double point_to_camera[3] = {
            mp.x - Twc_r.m[12],
            mp.y - Twc_r.m[13],
            mp.z - Twc_r.m[14]
    };

    // 相机方向（假设相机朝向Z轴正方向）
    double camera_dir[3] = { Twc_r.m[2], Twc_r.m[6], Twc_r.m[10] };

    // 计算点积
    double dot_product = camera_dir[0] * point_to_camera[0] +
                         camera_dir[1] * point_to_camera[1] +
                         camera_dir[2] * point_to_camera[2];

    // 计算距离
    double camera_distance = sqrt(point_to_camera[0]*point_to_camera[0] +
                                  point_to_camera[1]*point_to_camera[1] +
                                  point_to_camera[2]*point_to_camera[2]);

    // 判断是否在视野范围内
    if (dot_product >= cos_half_fov * camera_distance * camera_distance) {
        // 透视投影
        if(point_cam[2] <= 0) return; // 点在相机背后

        float3 pt = projectToImage(point_cam[0], point_cam[1], point_cam[2]);

        // 检查图像边界
        if (pt.x > (img_lefts_w / 2.0f) && pt.y > 0.0f && pt.x < img_left_w && pt.y < img_left_h) {
            int x = static_cast<int>(pt.x);
            int y = static_cast<int>(pt.y);

            // 获取图像灰度值（假设图像为单通道）
            unsigned char color = iml[y * iml_stride + x];

            // 使用原子操作添加新的关键点
            int pos = atomicAdd(d_newPointsCount, 1);
            if (pos < MAX_NEW_POINTS) { // 防止超过预分配的最大数量
                d_newPoints[2 * pos] = pt.x;
                d_newPoints[2 * pos + 1] = pt.y;
            }
        }
    }
}

// 主机端的函数实现
void switchMapKeypointCUDA(
        const std::vector<MapPointData>& mapPoints,
        const FrameData& frame_r,
        const unsigned char* d_iml, // 图像数据已经在设备内存中
        std::vector<float>& newPoints, // 输出的新关键点 (x, y) 浮点数组
        int maxNewPoints
) {
    int numPoints = mapPoints.size();

    // 分配设备内存并复制MapPoints
    MapPointData* d_mapPoints;
    cudaMalloc(&d_mapPoints, numPoints * sizeof(MapPointData));
    cudaMemcpy(d_mapPoints, mapPoints.data(), numPoints * sizeof(MapPointData), cudaMemcpyHostToDevice);

    // 转换FrameData到Matrix4x4
    Matrix4x4 Twc_r;
    Matrix4x4 T_left_right;
    for(int i = 0; i < 16; ++i){
        Twc_r.m[i] = frame_r.Twc[i];
        T_left_right.m[i] = frame_r.T_left_right[i];
    }

    // 分配输出关键点内存
    float* d_newPoints;
    cudaMalloc(&d_newPoints, maxNewPoints * 2 * sizeof(float)); // 每个关键点有x和y
    cudaMemset(d_newPoints, 0, maxNewPoints * 2 * sizeof(float));

    // 分配并初始化计数器
    int* d_newPointsCount;
    cudaMalloc(&d_newPointsCount, sizeof(int));
    cudaMemset(d_newPointsCount, 0, sizeof(int));

    // 定义CUDA线程和块的数量
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    // 启动CUDA核函数
    switchMapKeypointKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_mapPoints,
            numPoints,
            Twc_r,
            T_left_right,
            frame_r.cos_half_fov,
            frame_r.img_left_w,
            frame_r.img_left_h,
            frame_r.img_lefts_w,
            d_iml,
            frame_r.img_left_w, // 假设图像的stride为宽度
            d_newPoints,
            d_newPointsCount
    );

    // 同步并检查错误
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 获取新关键点的数量
    int h_newPointsCount;
    cudaMemcpy(&h_newPointsCount, d_newPointsCount, sizeof(int), cudaMemcpyDeviceToHost);

    // 限制最大关键点数量
    h_newPointsCount = (h_newPointsCount > maxNewPoints) ? maxNewPoints : h_newPointsCount;

    // 复制新关键点到主机
    std::vector<float> h_newPoints(2 * h_newPointsCount);
    cudaMemcpy(h_newPoints.data(), d_newPoints, 2 * h_newPointsCount * sizeof(float), cudaMemcpyDeviceToHost);

    // 添加到输出向量
    newPoints.insert(newPoints.end(), h_newPoints.begin(), h_newPoints.end());

    // 释放设备内存
    cudaFree(d_mapPoints);
    cudaFree(d_newPoints);
    cudaFree(d_newPointsCount);
}
