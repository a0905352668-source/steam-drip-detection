#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <map>
#include <chrono>
#include <unordered_map>

using namespace nvinfer1;
namespace fs = std::filesystem;

class MyLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

#define CHECK_CUDA(call)                                                               \
    do {                                                                               \
        cudaError_t err = call;                                                        \
        if (err != cudaSuccess) {                                                      \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)                     \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;           \
            std::exit(EXIT_FAILURE);                                                   \
        }                                                                              \
    } while (0)

// 时间格式化函数
std::string formatTime(double seconds) {
    if (seconds < 1e-3) {
        return std::to_string(seconds * 1e6) + " μs";
    } else if (seconds < 1) {
        return std::to_string(seconds * 1000) + " ms";
    } else {
        return std::to_string(seconds) + " s";
    }
}

// 结果结构体
struct InferenceResult {
    std::string videoPath;
    std::string videoName;
    float probNoSteam;
    float probSteam;
    bool isSteam;
    double inferenceTime;
    int processedFrames;
    std::string errorMsg;

    InferenceResult() : probNoSteam(0), probSteam(0), isSteam(false),
                       inferenceTime(0), processedFrames(0) {}
};

// 读取 .engine 文件
std::vector<char> loadEngineFile(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open engine file: " + enginePath);
    }
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    if (!file) {
        throw std::runtime_error("Failed to read engine file: " + enginePath);
    }
    return buffer;
}

// 获取目录中的所有视频文件
std::vector<std::string> getVideoFiles(const std::string& directoryPath) {
    std::vector<std::string> videoFiles;
    std::vector<std::string> videoExtensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"};

    try {
        if (!fs::exists(directoryPath)) {
            std::cerr << "❌ 目录不存在: " << directoryPath << std::endl;
            return videoFiles;
        }

        if (fs::is_regular_file(directoryPath)) {
            // 如果是单个文件，直接返回
            videoFiles.push_back(directoryPath);
            return videoFiles;
        }

        // 递归遍历目录
        for (const auto& entry : fs::recursive_directory_iterator(directoryPath)) {
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension().string();
                // 转换为小写比较
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

                if (std::find(videoExtensions.begin(), videoExtensions.end(), extension) != videoExtensions.end()) {
                    videoFiles.push_back(entry.path().string());
                }
            }
        }

        // 按文件名排序
        std::sort(videoFiles.begin(), videoFiles.end());

    } catch (const fs::filesystem_error& ex) {
        std::cerr << "❌ 文件系统错误: " << ex.what() << std::endl;
    }

    return videoFiles;
}

// 视频预处理函数
bool loadAndPreprocessVideo(const std::string& videoPath, int targetFrames,
                           int targetHeight, int targetWidth, std::vector<float>& output) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "❌ 无法打开视频文件: " << videoPath << std::endl;
        return false;
    }

    int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    int frameRate = cap.get(cv::CAP_PROP_FPS);
    int originalWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int originalHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::cout << "🎬 视频信息: " << totalFrames << "帧, " << frameRate << "FPS, "
              << originalWidth << "x" << originalHeight << std::endl;

    if (totalFrames == 0) {
        std::cerr << "❌ 视频文件为空或无法读取帧数" << std::endl;
        cap.release();
        return false;
    }

    // 均匀采样目标帧数
    std::vector<int> frameIndices;
    if (totalFrames <= targetFrames) {
        for (int i = 0; i < totalFrames; ++i) {
            frameIndices.push_back(i);
        }
    } else {
        for (int i = 0; i < targetFrames; ++i) {
            frameIndices.push_back(i * (totalFrames - 1) / (targetFrames - 1));
        }
    }

    // 清空并重新分配输出缓冲区
    output.clear();
    output.resize(targetFrames * 3 * targetHeight * targetWidth, 0.0f);

    // 与Python版本一致的归一化参数
    const float mean[3] = {0.5f, 0.5f, 0.5f};
    const float std[3] = {0.25f, 0.25f, 0.25f};

    int processedFrames = 0;
    cv::Mat frame;

    for (int i = 0; i < frameIndices.size(); ++i) {
        int frameIdx = frameIndices[i];
        cap.set(cv::CAP_PROP_POS_FRAMES, frameIdx);

        if (!cap.read(frame)) {
            std::cerr << "⚠️  无法读取帧 " << frameIdx << "，跳过" << std::endl;
            continue;
        }

        if (frame.empty()) {
            std::cerr << "⚠️  帧 " << frameIdx << " 为空，跳过" << std::endl;
            continue;
        }

        // 调整尺寸
        cv::Mat resizedFrame;
        cv::resize(frame, resizedFrame, cv::Size(targetWidth, targetHeight));

        // 转换 BGR 到 RGB
        cv::Mat rgbFrame;
        cv::cvtColor(resizedFrame, rgbFrame, cv::COLOR_BGR2RGB);

        // 转换为浮点并归一化到 [0,1]
        rgbFrame.convertTo(rgbFrame, CV_32FC3, 1.0/255.0);

        // 应用与Python一致的归一化: (x - mean) / std
        for (int h = 0; h < targetHeight; ++h) {
            for (int w = 0; w < targetWidth; ++w) {
                cv::Vec3f& pixel = rgbFrame.at<cv::Vec3f>(h, w);
                for (int c = 0; c < 3; ++c) {
                    pixel[c] = (pixel[c] - mean[c]) / std[c];
                }
            }
        }

        // 将数据复制到输出缓冲区 (C, T, H, W) 顺序
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < targetHeight; ++h) {
                for (int w = 0; w < targetWidth; ++w) {
                    int outputIdx = c * (targetFrames * targetHeight * targetWidth) +
                                   processedFrames * (targetHeight * targetWidth) +
                                   h * targetWidth + w;
                    output[outputIdx] = rgbFrame.at<cv::Vec3f>(h, w)[c];
                }
            }
        }

        processedFrames++;
        if (processedFrames >= targetFrames) break;
    }

    cap.release();

    if (processedFrames == 0) {
        std::cerr << "❌ 未能成功处理任何帧" << std::endl;
        return false;
    }

    // 如果实际处理的帧数少于目标帧数，用最后一帧填充
    if (processedFrames < targetFrames) {
        for (int f = processedFrames; f < targetFrames; ++f) {
            for (int c = 0; c < 3; ++c) {
                for (int h = 0; h < targetHeight; ++h) {
                    for (int w = 0; w < targetWidth; ++w) {
                        int srcIdx = c * (targetFrames * targetHeight * targetWidth) +
                                    (processedFrames - 1) * (targetHeight * targetWidth) +
                                    h * targetWidth + w;
                        int dstIdx = c * (targetFrames * targetHeight * targetWidth) +
                                    f * (targetHeight * targetWidth) +
                                    h * targetWidth + w;
                        output[dstIdx] = output[srcIdx];
                    }
                }
            }
        }
    }

    return true;
}

// 优化的TensorRT推理器
class OptimizedTrtInference {
private:
    std::unique_ptr<IRuntime> runtime;
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;
    MyLogger logger;

    // GPU资源
    void* dInput = nullptr;
    void* dOutput = nullptr;
    cudaStream_t stream;

    // 内存管理
    size_t currentInputSize = 0;
    size_t currentOutputSize = 0;

    // 性能统计
    std::vector<double> inferenceTimes;

public:
    OptimizedTrtInference(const std::string& enginePath) {
        // 创建CUDA流（只创建一次）
        CHECK_CUDA(cudaStreamCreate(&stream));

        // 加载引擎
        std::vector<char> engineData = loadEngineFile(enginePath);

        runtime = std::unique_ptr<IRuntime>(createInferRuntime(logger));
        if (!runtime) {
            throw std::runtime_error("Failed to create IRuntime");
        }

        engine = std::unique_ptr<ICudaEngine>(
            runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (!engine) {
            throw std::runtime_error("Failed to deserialize engine");
        }

        context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
        if (!context) {
            throw std::runtime_error("Failed to create execution context");
        }

        std::cout << "✅ Engine加载成功. NumIOTensors = " << engine->getNbIOTensors() << std::endl;
    }

    ~OptimizedTrtInference() {
        // 清理GPU资源
        if (dInput) {
            CHECK_CUDA(cudaFree(dInput));
            dInput = nullptr;
        }
        if (dOutput) {
            CHECK_CUDA(cudaFree(dOutput));
            dOutput = nullptr;
        }
        if (stream) {
            CHECK_CUDA(cudaStreamDestroy(stream));
        }
    }

    InferenceResult processVideo(const std::string& videoPath, int B, int T, int H, int W,
                                float threshold = 0.5f) {
        InferenceResult result;
        result.videoPath = videoPath;
        result.videoName = fs::path(videoPath).filename().string();

        auto startTime = std::chrono::high_resolution_clock::now();

        try {
            // 1. 预处理视频
            size_t inputVolume = static_cast<size_t>(B) * 3 * T * H * W;
            size_t outputVolume = static_cast<size_t>(B) * 2;  // 2个类别

            std::vector<float> hInput(inputVolume);
            if (!loadAndPreprocessVideo(videoPath, T, H, W, hInput)) {
                throw std::runtime_error("视频预处理失败");
            }

            // 2. 分配GPU内存（只在需要时重新分配）
            size_t inputSizeBytes = inputVolume * sizeof(float);
            size_t outputSizeBytes = outputVolume * sizeof(float);

            if (inputSizeBytes != currentInputSize) {
                if (dInput) CHECK_CUDA(cudaFree(dInput));
                CHECK_CUDA(cudaMalloc(&dInput, inputSizeBytes));
                currentInputSize = inputSizeBytes;
                std::cout << "🔄 重新分配输入内存: " << inputSizeBytes / 1024.0 / 1024.0 << " MB" << std::endl;
            }

            if (outputSizeBytes != currentOutputSize) {
                if (dOutput) CHECK_CUDA(cudaFree(dOutput));
                CHECK_CUDA(cudaMalloc(&dOutput, outputSizeBytes));
                currentOutputSize = outputSizeBytes;
                std::cout << "🔄 重新分配输出内存: " << outputSizeBytes / 1024.0 / 1024.0 << " MB" << std::endl;
            }

            // 3. 使用旧的API（兼容性更好）
            int inputIndex = 0;
            int outputIndex = 1;

            // 设置绑定维度
            Dims dims = context->getEngine().getBindingDimensions(inputIndex);
            dims.d[0] = B;
            dims.d[1] = 3;
            dims.d[2] = T;
            dims.d[3] = H;
            dims.d[4] = W;

            bool ok = context->setBindingDimensions(inputIndex, dims);
            if (!ok) {
                throw std::runtime_error("setBindingDimensions failed");
            }

            // 4. 绑定数组
            void* bindings[2];
            bindings[inputIndex] = dInput;
            bindings[outputIndex] = dOutput;

            // 5. 异步数据传输和推理
            auto inferenceStart = std::chrono::high_resolution_clock::now();

            // Host -> Device (异步)
            CHECK_CUDA(cudaMemcpyAsync(dInput, hInput.data(), inputSizeBytes,
                                     cudaMemcpyHostToDevice, stream));

            // 异步推理（使用旧的enqueueV2保持兼容性）
            ok = context->enqueueV2(bindings, stream, nullptr);
            if (!ok) {
                throw std::runtime_error("enqueueV2 failed");
            }

            // Device -> Host (异步)
            std::vector<float> hOutput(outputVolume);
            CHECK_CUDA(cudaMemcpyAsync(hOutput.data(), dOutput, outputSizeBytes,
                                     cudaMemcpyDeviceToHost, stream));

            // 等待所有操作完成
            CHECK_CUDA(cudaStreamSynchronize(stream));

            auto inferenceEnd = std::chrono::high_resolution_clock::now();
            double inferenceTime = std::chrono::duration<double>(inferenceEnd - inferenceStart).count();
            inferenceTimes.push_back(inferenceTime);

            // 6. 后处理
            float logit0 = hOutput[0];
            float logit1 = hOutput[1];
            float maxLogit = std::max(logit0, logit1);
            float e0 = std::exp(logit0 - maxLogit);
            float e1 = std::exp(logit1 - maxLogit);
            float sum = e0 + e1;

            result.probNoSteam = e0 / sum;
            result.probSteam = e1 / sum;
            result.isSteam = (result.probSteam >= threshold);
            result.processedFrames = T;
            result.inferenceTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count();

        } catch (const std::exception& e) {
            result.errorMsg = e.what();
        }

        return result;
    }

    void printPerformanceStats() {
        if (inferenceTimes.empty()) return;

        double totalTime = std::accumulate(inferenceTimes.begin(), inferenceTimes.end(), 0.0);
        double avgTime = totalTime / inferenceTimes.size();
        double minTime = *std::min_element(inferenceTimes.begin(), inferenceTimes.end());
        double maxTime = *std::max_element(inferenceTimes.begin(), inferenceTimes.end());

        std::cout << "\n📊 TensorRT性能统计:" << std::endl;
        std::cout << "  推理次数: " << inferenceTimes.size() << std::endl;
        std::cout << "  总推理时间: " << formatTime(totalTime) << std::endl;
        std::cout << "  平均推理时间: " << formatTime(avgTime) << std::endl;
        std::cout << "  最快推理: " << formatTime(minTime) << std::endl;
        std::cout << "  最慢推理: " << formatTime(maxTime) << std::endl;
        std::cout << "  平均FPS: " << (1.0 / avgTime) << std::endl;
    }
};

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cout << "Usage:\n"
                  << "  " << argv[0]
                  << " engine_path input_path B T H W [threshold]\n\n"
                  << "参数说明:\n"
                  << "  engine_path: TensorRT引擎文件路径\n"
                  << "  input_path:  输入视频文件路径或目录路径\n"
                  << "  B:           Batch大小 (通常为1)\n"
                  << "  T:           帧数 (如48)\n"
                  << "  H:           高度 (如256)\n"
                  << "  W:           宽度 (如256)\n"
                  << "  threshold:   steam判断阈值 (默认0.5)\n\n"
                  << "示例:\n"
                  << "  # 单个视频推理\n"
                  << "  " << argv[0]
                  << " steam.engine test_video.mp4 1 48 256 256 0.5\n\n"
                  << "  # 目录批量推理\n"
                  << "  " << argv[0]
                  << " steam.engine /path/to/videos 1 48 256 256 0.5\n";
        return 0;
    }

    std::string enginePath = argv[1];
    std::string inputPath = argv[2];
    int B = std::stoi(argv[3]);
    int T = std::stoi(argv[4]);
    int H = std::stoi(argv[5]);
    int W = std::stoi(argv[6]);
    float threshold = (argc > 7) ? std::stof(argv[7]) : 0.5f;

    // 检查GPU信息
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "🎯 GPU: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;

    if (prop.major >= 8) {
        std::cout << "🔧 启用TF32加速（Ampere+ GPU）" << std::endl;
    }

    // 获取所有要处理的视频文件
    std::vector<std::string> videoFiles = getVideoFiles(inputPath);
    if (videoFiles.empty()) {
        std::cerr << "❌ 未找到可处理的视频文件: " << inputPath << std::endl;
        return 1;
    }

    std::cout << "📁 找到 " << videoFiles.size() << " 个视频文件" << std::endl;
    std::cout << "⚙️  参数: B=" << B << ", T=" << T << ", H=" << H << ", W=" << W
              << ", threshold=" << threshold << std::endl;

    try {
        // 1. 初始化优化的TensorRT推理器
        std::cout << "🚀 初始化TensorRT推理器..." << std::endl;
        OptimizedTrtInference trtInference(enginePath);

        // 2. 批量处理视频
        std::vector<InferenceResult> results;
        int successCount = 0;
        int failCount = 0;

        double totalInferenceTime = 0.0;
        int totalProcessedFrames = 0;

        std::cout << "\n🚀 开始批量推理..." << std::endl;
        std::cout << "=" << std::string(80, '=') << "=" << std::endl;

        for (size_t i = 0; i < videoFiles.size(); ++i) {
            const auto& videoFile = videoFiles[i];
            std::cout << "[" << (i+1) << "/" << videoFiles.size() << "] 处理: "
                      << fs::path(videoFile).filename().string() << std::endl;

            InferenceResult result = trtInference.processVideo(videoFile, B, T, H, W, threshold);

            if (result.errorMsg.empty()) {
                successCount++;
                totalInferenceTime += result.inferenceTime;
                totalProcessedFrames += result.processedFrames;

                // 修复三元运算符语法错误
                std::string steamLabel = result.isSteam ? "STEAM" : "NO_STEAM";

                std::cout << "  结果: no_steam=" << result.probNoSteam
                          << ", steam=" << result.probSteam
                          << " | 判断: " << steamLabel
                          << " | 耗时: " << formatTime(result.inferenceTime) << std::endl;
            } else {
                failCount++;
                std::cout << "  ❌ 错误: " << result.errorMsg << std::endl;
            }

            results.push_back(result);
            std::cout << "-" << std::string(60, '-') << "-" << std::endl;

            // 每处理100个视频输出一次进度
            if ((i + 1) % 100 == 0) {
                std::cout << "📊 进度: " << (i + 1) << "/" << videoFiles.size()
                          << " (" << (100.0 * (i + 1) / videoFiles.size()) << "%)" << std::endl;
            }
        }

        // 3. 输出性能统计
        trtInference.printPerformanceStats();

        // 4. 输出总体统计信息
        std::cout << "\n📊 批量推理完成!" << std::endl;
        std::cout << "=" << std::string(80, '=') << "=" << std::endl;

        std::cout << "📈 总体统计:" << std::endl;
        std::cout << "  总视频数: " << videoFiles.size() << std::endl;
        std::cout << "  成功: " << successCount << std::endl;
        std::cout << "  失败: " << failCount << std::endl;

        if (successCount > 0) {
            double avgTime = totalInferenceTime / successCount;
            double fps = totalProcessedFrames / totalInferenceTime;

            std::cout << "  总耗时: " << formatTime(totalInferenceTime) << std::endl;
            std::cout << "  平均耗时: " << formatTime(avgTime) << " 每视频" << std::endl;
            std::cout << "  处理速度: " << fps << " FPS" << std::endl;
        }

        // 5. 分类统计
        int steamCount = 0;
        int noSteamCount = 0;
        std::vector<std::string> steamVideos;
        std::vector<std::string> noSteamVideos;

        for (const auto& result : results) {
            if (result.errorMsg.empty()) {
                if (result.isSteam) {
                    steamCount++;
                    steamVideos.push_back(result.videoPath);
                } else {
                    noSteamCount++;
                    noSteamVideos.push_back(result.videoPath);
                }
            }
        }

        std::cout << "\n🎯 分类统计 (阈值=" << threshold << "):" << std::endl;
        std::cout << "  STEAM: " << steamCount << " 个视频" << std::endl;
        std::cout << "  NO_STEAM: " << noSteamCount << " 个视频" << std::endl;

        // 6. 输出少数类
        std::string minorityClass;
        std::vector<std::string>* minorityVideos = nullptr;

        if (steamCount < noSteamCount) {
            minorityClass = "STEAM";
            minorityVideos = &steamVideos;
        } else if (noSteamCount < steamCount) {
            minorityClass = "NO_STEAM";
            minorityVideos = &noSteamVideos;
        }

        if (minorityVideos != nullptr && !minorityVideos->empty()) {
            std::cout << "\n🔍 少数类: " << minorityClass << " (" << minorityVideos->size() << " 个视频)" << std::endl;
            std::cout << "少数类视频列表:" << std::endl;
            for (size_t i = 0; i < minorityVideos->size(); ++i) {
                std::cout << "  [" << (i+1) << "] " << fs::path((*minorityVideos)[i]).filename().string() << std::endl;
                std::cout << "      " << (*minorityVideos)[i] << std::endl;
            }
        } else if (steamCount == noSteamCount && steamCount > 0) {
            std::cout << "\n⚖️  两类数量相等，各 " << steamCount << " 个视频" << std::endl;
        }

        // 7. 保存结果到文件（可选）
        std::string resultFile = "inference_results.csv";
        std::ofstream outFile(resultFile);
        if (outFile.is_open()) {
            outFile << "视频文件,Steam概率,NoSteam概率,判断结果,耗时(秒),状态\n";
            for (const auto& result : results) {
                outFile << result.videoPath << ","
                        << result.probSteam << ","
                        << result.probNoSteam << ","
                        << (result.isSteam ? "STEAM" : "NO_STEAM") << ","
                        << result.inferenceTime << ","
                        << (result.errorMsg.empty() ? "成功" : "失败:" + result.errorMsg)
                        << "\n";
            }
            outFile.close();
            std::cout << "\n💾 详细结果已保存到: " << resultFile << std::endl;
        }

        std::cout << "✅ 批量推理完成!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
