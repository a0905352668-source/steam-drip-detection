# ========== 基础配置 ==========
cmake_minimum_required(VERSION 3.18)  # TensorRT 8.x 需要CMake 3.18+
project(cpp LANGUAGES CXX)

# 禁用源码内构建
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "禁止源码内构建！请创建build目录：mkdir build && cd build && cmake ..")
endif()

# 设置编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ========== 依赖配置 ==========
# 1. OpenCV配置（需要添加 video 模块，因为使用了 findTransformECC）
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio video)  # 添加 video
if(NOT OpenCV_FOUND)
    # 手动指定路径（如果自动查找失败）
    set(OpenCV_DIR "/path/to/opencv/build")
    find_package(OpenCV REQUIRED)
endif()
message(STATUS "OpenCV版本: ${OpenCV_VERSION}")

# 2. TensorRT配置（必须手动指定路径）
set(TENSORRT_ROOT "/opt/TensorRT-8.6.1.6" CACHE PATH "TensorRT installation")  # 根据您的安装路径修改
set(TensorRT_INCLUDE_DIRS ${TENSORRT_ROOT}/include)
set(TensorRT_LIBRARIES
    ${TENSORRT_ROOT}/lib/libnvinfer.so
    ${TENSORRT_ROOT}/lib/libnvinfer_plugin.so
    ${TENSORRT_ROOT}/lib/libnvonnxparser.so
)

# 3. CUDA运行时（TensorRT依赖）
find_package(CUDAToolkit REQUIRED)  # 自动查找CUDA

# ========== 路径验证 ==========
foreach(DEP_PATH IN LISTS OpenCV_INCLUDE_DIRS TensorRT_INCLUDE_DIRS)
    if(NOT EXISTS "${DEP_PATH}")
        message(FATAL_ERROR "路径不存在: ${DEP_PATH}")
    endif()
endforeach()

# ========== 目标定义 ==========
# 添加可执行文件
add_executable(${PROJECT_NAME}
    steam_infer.cpp
)

# 包含头文件目录
target_include_directories(${PROJECT_NAME} PRIVATE
    include/
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)

# 链接库文件 - 修复这里！
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
    CUDA::cudart  # CUDA运行时库
    pthread       # Linux线程库
    dl            # 动态加载库
    stdc++fs      # 添加这一行：C++ 实验性文件系统库
)

# ========== 构建配置 ==========
# 优化选项（Release模式）
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(${PROJECT_NAME} PRIVATE -O3 -march=native)
    target_link_options(${PROJECT_NAME} PRIVATE -s)  # 去除符号表
else()
    # Debug模式配置
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG=1)
    target_compile_options(${PROJECT_NAME} PRIVATE -g3 -O0)
endif()

# ========== 安装规则 ==========
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)
install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models OPTIONAL)  # 部署模型文件

# ========== 高级选项 ==========
option(ENABLE_TESTS "启用单元测试" OFF)
if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)  # 测试目录需包含CMakeLists.txt
endif()
