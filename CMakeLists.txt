cmake_minimum_required(VERSION 3.16)
project(vision_training LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖库
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE) 
find_package(Ceres REQUIRED)            

# ================= 任务1 =================
add_executable(task1_image src/task1_image/main.cpp)
target_include_directories(task1_image PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(task1_image PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)

# ================= 任务2：视频拟合 =================
add_executable(task2_fit src/task2_fit/main.cpp)
target_include_directories(task2_fit PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(task2_fit PRIVATE ${OpenCV_LIBS} Eigen3::Eigen Ceres::ceres)

# ================= Ceres测试 =================
add_executable(task3_windmill src/task3_windmill/main.cpp) 
target_include_directories(task3_windmill PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(task3_windmill PRIVATE ${OpenCV_LIBS} Eigen3::Eigen Ceres::ceres)