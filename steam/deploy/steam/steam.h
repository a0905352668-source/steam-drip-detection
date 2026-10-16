#ifndef CSTEAM_H
#define CSTEAM_H

#include "Steam_global.h"

#include <Yolov8Cls.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>
#include <experimental/filesystem>

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cassert>
#include <map>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <QJsonArray>
#include <QDir>
#include <QJsonObject>
#include <QDebug>
#include <QDateTime>
#include <QDate>
#include <QFile>
#include <QFileInfo>
#include <QString>

using namespace nvinfer1;
namespace fs = std::experimental::filesystem;

/* ====================== 时间跟踪器 ====================== */
struct TimeTracker {
    using Clock = std::chrono::steady_clock;

    Clock::time_point start;       // 整个过程开始时间
    Clock::time_point stageStart;  // 当前阶段开始时间
    std::string       currentStage;// 当前阶段名称
    std::map<std::string, double> stages; // 各阶段累计时间（秒）

    TimeTracker() {
        start      = Clock::now();
        stageStart = Clock::now();
    }

    // 开始一个阶段，会自动结束上一个阶段并累计时间
    void startStage(const std::string& name) {
        Clock::time_point now = Clock::now();
        if (!currentStage.empty()) {
            double elapsed = std::chrono::duration<double>(now - stageStart).count();
            stages[currentStage] += elapsed;
        }
        currentStage = name;
        stageStart   = now;
    }

    // 结束当前阶段
    void endStage() {
        Clock::time_point now = Clock::now();
        if (!currentStage.empty()) {
            double elapsed = std::chrono::duration<double>(now - stageStart).count();
            stages[currentStage] += elapsed;
            currentStage.clear();
        }
    }

    // 从构造到当前的总耗时
    double totalTime() const {
        Clock::time_point now = Clock::now();
        return std::chrono::duration<double>(now - start).count();
    }
};

/* ====================== 配置结构体 ====================== */
struct Config {
    // I/O 路径
    std::string INPUT_PATH;  // 输入：单视频或目录
    std::string OUTPUT_DIR;  // 输出根目录

    // 分组与 tile 覆盖
    bool  GROUP_TO_CLIPS;        // 是否按 clip 分组；false=整段一组
    int   CLIP_LEN;              // clip 长度（帧数）
    int   WINDOW_STRIDE_FRAMES;  // clip 滑动窗口步长（帧）
    int   TILE_SIZE;             // tile 边长（像素）
    float OVERLAP;               // tile 之间的重叠比例（0~1）
    std::string COVERAGE_MODE;   // "topk" 或 "all"
    int   TOPK_TILES;            // Top-K tile 数量
    bool  FALLBACK_ALL_IF_LOW;   // 目前未使用，保留占位

    // 分辨率/读帧
    int  RESIZE_SHORT_SIDE;      // 短边缩放尺寸；0 表示不缩放
    bool USE_NATIVE_FPS;         // 目前读取 FPS 时未使用，可扩展

    // 速度优化
    int   SCORE_FRAME_STRIDE;
    float SCORE_DOWNSCALE;
    int   HEAT_FRAME_STRIDE;
    float HEAT_DOWNSCALE;
    float TIGHTEN_EDGE_CHECK_DOWNSCALE;

    // 蒸汽综合打分
    std::string SCORE_MODE;   // "diff" 或 "steam"
    float STEAM_W_DIFF;
    float STEAM_W_WHITE;
    float STEAM_W_SMOOTH;

    // ECC 稳像
    bool STABILIZE_ECC;
    int  ECC_WINSIZE;
    int  ECC_ITERS;

    // 自动紧致裁剪 & 背景净化
    bool  AUTO_TIGHTEN;
    std::string TIGHTEN_THR_MODE; // "percentile" 或 "otsu"
    float TIGHTEN_THR_PCT;        // 百分位阈值
    int   TIGHTEN_MIN_AREA;       // 最小连通区域面积
    float TIGHTEN_MARGIN;         // 固定边缘扩展比例
    float EDGE_DENSITY_MAX;       // 允许的最大边缘密度（过滤高纹理石头）
    bool  ADAPTIVE_MARGIN;        // 是否启用自适应边缘扩展
    float ADAPTIVE_MARGIN_BASE;
    float ADAPTIVE_MARGIN_LO;
    float ADAPTIVE_MARGIN_HI;

    std::string BG_MODE;          // "blur"/"black"/"gray"/"mean"/"keep"
    int         BG_BLUR_K;        // 高斯模糊核大小

    // 导出配置
    std::string EXPORT_MODE;      // "video"/"frames"/"json"
    int   RESIZE_OUT;             // 最终导出尺寸
    int   JPEG_QUALITY;           // JPEG 质量
    bool  SAVE_VIS;               // 是否保存 vis.jpg（tile 覆盖可视化）
    bool  SKIP_EXISTING;          // 已存在的视频是否跳过重写

    std::vector<std::string> VIDEO_EXTS; // 识别的视频后缀

    Config() {
        GROUP_TO_CLIPS       = false;
        CLIP_LEN             = 16;
        WINDOW_STRIDE_FRAMES = 8;
        TILE_SIZE            = 512;
        OVERLAP              = 0.25f;
        COVERAGE_MODE        = "topk";
        TOPK_TILES           = 5;
        FALLBACK_ALL_IF_LOW  = false;

        RESIZE_SHORT_SIDE    = 0;
        USE_NATIVE_FPS       = true;

        SCORE_FRAME_STRIDE   = 2;
        SCORE_DOWNSCALE      = 0.5f;
        HEAT_FRAME_STRIDE    = 2;
        HEAT_DOWNSCALE       = 1.0f;
        TIGHTEN_EDGE_CHECK_DOWNSCALE = 0.75f;

        SCORE_MODE           = "steam";
        STEAM_W_DIFF         = 0.96f;
        STEAM_W_WHITE        = 0.02f;
        STEAM_W_SMOOTH       = 0.02f;

        STABILIZE_ECC        = false;
        ECC_WINSIZE          = 7;
        ECC_ITERS            = 30;

        AUTO_TIGHTEN         = true;
        TIGHTEN_THR_MODE     = "percentile";
        TIGHTEN_THR_PCT      = 85.0f;
        TIGHTEN_MIN_AREA     = 300;
        TIGHTEN_MARGIN       = 0.15f;
        EDGE_DENSITY_MAX     = 0.18f;
        ADAPTIVE_MARGIN      = false;
        ADAPTIVE_MARGIN_BASE = 0.12f;
        ADAPTIVE_MARGIN_LO   = 0.08f;
        ADAPTIVE_MARGIN_HI   = 0.25f;

        BG_MODE              = "blur";
        BG_BLUR_K            = 21;

        EXPORT_MODE          = "video";
        RESIZE_OUT           = 256;
        JPEG_QUALITY         = 95;
        SAVE_VIS             = true;
        SKIP_EXISTING        = true;

        VIDEO_EXTS.push_back(".mp4");
        VIDEO_EXTS.push_back(".avi");
        VIDEO_EXTS.push_back(".mov");
        VIDEO_EXTS.push_back(".mkv");
        VIDEO_EXTS.push_back(".flv");
        VIDEO_EXTS.push_back(".webm");
    }
};

struct ProcessResult {
    bool success;
    std::string videoPath;
    TimeTracker timer;
    int frameCount;
    int groupCount;
    int tileCount;
    int totalTilesProcessed;
    double avgTileTime;
    double minTileTime;
    double maxTileTime;
    std::string errorMsg;

    ProcessResult()
        : success(false), frameCount(0), groupCount(0), tileCount(0),
          totalTilesProcessed(0), avgTileTime(0.0),
          minTileTime(0.0), maxTileTime(0.0) {}
};

struct InferenceResult {
    std::string videoPath;
    std::string videoName;
    float probNoSteam;
    float probSteam;
    bool  isSteam;
    double inferenceTime;
    int    processedFrames;
    std::string errorMsg;

    InferenceResult()
        : probNoSteam(0.f), probSteam(0.f), isSteam(false),
          inferenceTime(0.0), processedFrames(0) {}
};

class STEAM_EXPORT CSteam
{
private:
    CSteam();

public:
    static CSteam& GetInstance();

    int Detect(const QString &VideoPath,
               CYolov8Cls* m_Yolov8Cls,
               const QJsonObject &input,
               QJsonObject &output);

    void WaterVaporAnalysis(int Return,
                            QJsonObject& FunctionResult,
                            QJsonObject& ResultStatus,
                            QJsonObject& Result,
                            QJsonArray& ResultList);

    // TensorRT 推理相关
    std::vector<std::string> getVideoFiles(const std::string& directoryPath);
    std::vector<char> loadEngineFile(const std::string& enginePath);

    InferenceResult processSingleVideo(
        IExecutionContext* context,
        int inputIndex,
        int outputIndex,
        const std::string& videoPath,
        int B, int T, int H, int W,
        float threshold = 0.5f);

    bool loadAndPreprocessVideo(const std::string& videoPath,
                                int targetFrames,
                                int targetHeight,
                                int targetWidth,
                                std::vector<float>& output);

    std::string formatTime(double seconds);

    // 预处理
    std::vector<cv::Mat> readAllFrames(const std::string& videoPath);
    void resizeShortSideFrames(std::vector<cv::Mat>& frames, int shortSide);
    void eccStabilize(std::vector<cv::Mat>& frames, int winsize, int iters);

    cv::Mat diffHeatmapFast(const std::vector<cv::Mat>& frames,
                            int frameStride,
                            float downscale,
                            const cv::Size& upsizeSize);

    cv::Mat steamScoreMapFast(const std::vector<cv::Mat>& frames,
                              const cv::Mat& reuseDiff);

    cv::Mat whitenessMapFast(const std::vector<cv::Mat>& frames,
                             int frameStride,
                             float downscale,
                             const cv::Size& upsizeSize);

    cv::Mat edgeDensityMapFast(const std::vector<cv::Mat>& frames,
                               int frameStride,
                               float downscale,
                               const cv::Size& upsizeSize);

    std::vector<cv::Mat> downscaleFrames(const std::vector<cv::Mat>& frames,
                                         float scale);

    cv::Mat normalize01Percentile(const cv::Mat& input);
    double getPercentile(const cv::Mat& mat, float pct);

    std::pair<std::vector<cv::Rect>, std::vector<float> >
    selectTopKByMap(const cv::Mat& scoreMap,
                    int tileSize,
                    float overlap,
                    int k);

    std::vector<cv::Rect> makeFullCoverTiles(int height, int width,
                                             int tileSize, float overlap);

    bool validateTileConsistency(const std::vector<cv::Rect>& selectedTiles,
                                 const std::vector<cv::Mat>& frames);

    cv::Mat drawTilesVis(const cv::Mat& rgb,
                         const std::vector<cv::Rect>& tiles,
                         const std::vector<float>* scores);

    void saveTilesMeta(const std::string& metaPath,
                       const std::string& videoPath,
                       int height, int width,
                       const Config& cfg,
                       const std::vector<cv::Rect>& tiles,
                       const std::vector<float>& scores);

    std::string toLower(const std::string& s);

    std::vector<std::vector<cv::Mat> >
    makeClips(const std::vector<cv::Mat>& frames,
              int clipLen, int stride);

    std::string padIndex(size_t idx, int width);

    float adaptiveMarginFunc(const cv::Rect& bbox, int tileH, int tileW,
                             float base, float lo, float hi);

    void applyBgMode(std::vector<cv::Mat>& frames,
                     const cv::Mat& mask,
                     const std::string& mode,
                     int blurK);

    void saveFramesDir(const std::string& outDir,
                       const std::vector<cv::Mat>& frames,
                       int jpegQuality);

    void saveVideo(const std::string& outPath,
                   const std::vector<cv::Mat>& frames,
                   double fps);

    // ========== 新增：画框相关 ==========
    double computeIoU(const cv::Rect& a, const cv::Rect& b);

    bool drawSteamBoxesOnVideo(
        const std::string& inputVideoPath,
        const std::string& outputVideoPath,
        const std::vector<cv::Rect>& tiles,
        const std::vector<cv::Rect>& tightBoxes,
        const std::vector<bool>& hasTightBox,
        const std::vector<float>& probSteamPerTile,
        float threshold,
        float mergeIoUThreshold   = 0.1f,
        float expandMergedBoxRatio= 0.05f);
};

// 简单 Logger
class MyLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

#endif // CSTEAM_H
