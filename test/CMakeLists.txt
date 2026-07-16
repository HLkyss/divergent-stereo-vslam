# 设置最低版本和项目名称
cmake_minimum_required(VERSION 3.10)
project(VirtualCameraProjection)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# 查找 Sophus
find_package(Sophus NO_DEFAULT_PATH PATHS /home/hl/project/ov2_diverg_ws/src/ov2slam/Thirdparty/Sophus/build/)
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# 查找 OpenCV 库
set(OpenCV_DIR "/home/hl/Downloads/opencv-3.4.15/build")
find_package(OpenCV 3)

# 包含 OpenCV 头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(VirtualCameraProjection test_virtual_cam.cpp)
add_executable(VirtualCameraProjection2 test_virtual_cam2.cpp)
add_executable(test_T test_T.cpp)
add_executable(test_T2 test_T2.cpp)
add_executable(track_test1 track_test1.cpp)
add_executable(track_test2 track_test2.cpp)
add_executable(compute_depth compute_depth.cpp)

# 链接 OpenCV 库到可执行文件
target_link_libraries(VirtualCameraProjection ${OpenCV_LIBS})
target_link_libraries(VirtualCameraProjection2 ${OpenCV_LIBS})
target_link_libraries(test_T Eigen3::Eigen Sophus::Sophus)
target_link_libraries(test_T2 Eigen3::Eigen Sophus::Sophus)
target_link_libraries(track_test1 Eigen3::Eigen Sophus::Sophus ${OpenCV_LIBS})
target_link_libraries(track_test2 Eigen3::Eigen Sophus::Sophus ${OpenCV_LIBS})
target_link_libraries(compute_depth Eigen3::Eigen Sophus::Sophus ${OpenCV_LIBS})
#target_link_libraries(test_T2 Eigen3::Eigen Sophus::Sophus)

