// virtual_image_kernel.cuh
#ifndef VIRTUAL_IMAGE_KERNEL_CUH
#define VIRTUAL_IMAGE_KERNEL_CUH

// CUDA 加速函数声明
void generateVirtualImageCUDA(const unsigned char* realImage, unsigned char* virtualImage, int width, int height,
                              int realWidth, int realHeight, const double* R, const double* virtualCameraK,
                              const double* realCameraK, const double* T);

#endif // VIRTUAL_IMAGE_KERNEL_CUH
