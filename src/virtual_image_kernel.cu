// virtual_image_kernel.cu
#include "/home/hl/project/ov2_diverg_ws_2map1traj/src/ov2slam/include/virtual_image_kernel.cuh"
#include <cuda_runtime.h>

// CUDA 核函数
__global__ void generateVirtualImageKernel(const unsigned char* realImage, unsigned char* virtualImage,
                                           int width, int height, int realWidth, int realHeight,
                                           const double* R, const double* virtualCameraK,
                                           const double* realCameraK, const double* T) {
    int v_x = blockIdx.x * blockDim.x + threadIdx.x;
    int v_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (v_x < width && v_y < height) {
        // 虚拟图像坐标归一化
        double point3D_virtual[3];
        point3D_virtual[0] = (v_x - virtualCameraK[2]) / virtualCameraK[0];
        point3D_virtual[1] = (v_y - virtualCameraK[5]) / virtualCameraK[4];
        point3D_virtual[2] = 1.0;

        // 将虚拟坐标转换到真实相机坐标
        double point3D_real[3];
        for (int i = 0; i < 3; i++) {
//            point3D_real[i] = R[0 * 3 + i] * point3D_virtual[0] +
//                              R[1 * 3 + i] * point3D_virtual[1] +
//                              R[2 * 3 + i] * point3D_virtual[2] - T[i];//使用转置的R矩阵
            point3D_real[i] = R[i * 3 + 0] * point3D_virtual[0] +
                              R[i * 3 + 1] * point3D_virtual[1] +
                              R[i * 3 + 2] * point3D_virtual[2] - T[i];//使用R
        }

        if (point3D_real[2] != 0) {
            // 将三维点投影到真实相机的像素坐标
            double realPixelX = (realCameraK[0] * point3D_real[0] + realCameraK[1] * point3D_real[1] + realCameraK[2] * point3D_real[2]) / point3D_real[2];
            double realPixelY = (realCameraK[3] * point3D_real[0] + realCameraK[4] * point3D_real[1] + realCameraK[5] * point3D_real[2]) / point3D_real[2];

            int pixelX = static_cast<int>(realPixelX);
            int pixelY = static_cast<int>(realPixelY);

            if (pixelX >= 0 && pixelX < realWidth && pixelY >= 0 && pixelY < realHeight) {
                virtualImage[v_y * width + v_x] = realImage[pixelY * realWidth + pixelX];
            } else {
                virtualImage[v_y * width + v_x] = 0;
            }
        } else {
            virtualImage[v_y * width + v_x] = 0;
        }
    }
}

// CUDA 接口函数
void generateVirtualImageCUDA(const unsigned char* realImage, unsigned char* virtualImage, int width, int height,
                              int realWidth, int realHeight, const double* R, const double* virtualCameraK,
                              const double* realCameraK, const double* T) {
    // 分配并复制矩阵数据到设备
    double *d_R, *d_virtualCameraK, *d_realCameraK, *d_T;
    cudaMalloc((void**)&d_R, 9 * sizeof(double));
    cudaMalloc((void**)&d_virtualCameraK, 9 * sizeof(double));
    cudaMalloc((void**)&d_realCameraK, 9 * sizeof(double));
    cudaMalloc((void**)&d_T, 3 * sizeof(double));

    cudaMemcpy(d_R, R, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_virtualCameraK, virtualCameraK, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_realCameraK, realCameraK, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T, T, 3 * sizeof(double), cudaMemcpyHostToDevice);

    // 分配并复制图像数据到设备
    unsigned char *d_realImage, *d_virtualImage;
    cudaMalloc(&d_realImage, realWidth * realHeight * sizeof(unsigned char));
    cudaMalloc(&d_virtualImage, width * height * sizeof(unsigned char));
    cudaMemcpy(d_realImage, realImage, realWidth * realHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // 配置 CUDA 网格和块大小
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 执行 CUDA 内核
    generateVirtualImageKernel<<<gridSize, blockSize>>>(d_realImage, d_virtualImage, width, height, realWidth, realHeight, d_R, d_virtualCameraK, d_realCameraK, d_T);

    // 复制结果回主机
    cudaMemcpy(virtualImage, d_virtualImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_R);
    cudaFree(d_virtualCameraK);
    cudaFree(d_realCameraK);
    cudaFree(d_T);
    cudaFree(d_realImage);
    cudaFree(d_virtualImage);
}
