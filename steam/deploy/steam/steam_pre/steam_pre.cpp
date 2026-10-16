#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <experimental/filesystem>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <map>
#include <iomanip>
#include <cctype>

namespace fs = std::experimental::filesystem;

/* ====================== 时间格式化工具 ====================== */
// 将秒数格式化为 "x毫秒 / x.xx秒 / x.xx分钟 / x.xx小时"
std::string formatTime(double seconds) {
    if (seconds < 1.0) {
        int ms = static_cast<int>(seconds * 1000.0 + 0.5);
        return std::to_string(ms) + "毫秒";
    } else if (seconds < 60.0) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << seconds << "秒";
        return oss.str();
    } else if (seconds < 3600.0) {
        double minutes = seconds / 60.0;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << minutes << "分钟";
        return oss.str();
    } else {
        double hours = seconds / 3600.0;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << hours << "小时";
        return oss.str();
    }
}

// 将秒数格式化为 "x小时x分x.xx秒 / x分x.xx秒 / x.xx秒"
std::string formatTimeDetailed(double seconds) {
    int hours = static_cast<int>(seconds / 3600.0);
    int minutes = static_cast<int>((seconds - hours * 3600.0) / 60.0);
    double secs = seconds - hours * 3600.0 - minutes * 60.0;
    std::ostringstream oss;
    if (hours > 0) {
        oss << hours << "小时" << minutes << "分"
            << std::fixed << std::setprecision(2) << secs << "秒";
    } else if (minutes > 0) {
        oss << minutes << "分"
            << std::fixed << std::setprecision(2) << secs << "秒";
    } else {
        oss << std::fixed << std::setprecision(2) << secs << "秒";
    }
    return oss.str();
}

/* ====================== 时间跟踪器 ====================== */
// 用于记录各阶段处理时间，类似 Python TimeTracker
struct TimeTracker {
    typedef std::chrono::steady_clock Clock;
    Clock::time_point start;       // 整个过程开始时间
    Clock::time_point stageStart;  // 当前阶段开始时间
    std::string currentStage;      // 当前阶段名称
    std::map<std::string,double> stages; // 各阶段累计时间（秒）

    TimeTracker() {
        start = Clock::now();
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
        stageStart = now;
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
// 绝大部分字段都和 Python CONFIG 字段对齐
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

    // 速度优化参数（采样步长和下采样比例）
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

    std::string BG_MODE;          // 背景模式："blur"/"black"/"gray"/"mean"/"keep"
    int         BG_BLUR_K;        // 高斯模糊核大小

    // 导出配置
    std::string EXPORT_MODE;      // "video"/"frames"/"json"
    int   RESIZE_OUT;             // 最终导出尺寸（宽高）
    int   JPEG_QUALITY;           // JPEG 质量
    bool  SAVE_VIS;               // 是否保存 vis.jpg（Tile 覆盖可视化）
    bool  SKIP_EXISTING;          // 已存在的视频是否跳过重写

    std::vector<std::string> VIDEO_EXTS; // 识别的视频后缀

    // 构造函数：设置默认值（与 Python 对齐）
    Config() {
        // 默认值设为空，由命令行参数决定
        INPUT_PATH  = "";
        OUTPUT_DIR  = "";

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

/* ====================== 命令行参数解析 ====================== */

// 显示用法信息
void showUsage(const std::string& programName) {
    std::cout << "用法: " << programName << " -i <输入路径> -o <输出目录> [选项]" << std::endl;
    std::cout << std::endl;
    std::cout << "必需参数:" << std::endl;
    std::cout << "  -i, --input <路径>      输入视频文件或目录" << std::endl;
    std::cout << "  -o, --output <目录>     输出目录" << std::endl;
    std::cout << std::endl;
    std::cout << "可选参数:" << std::endl;
    std::cout << "  -h, --help              显示此帮助信息" << std::endl;
    std::cout << "  -t, --tile-size <尺寸>  tile尺寸 (默认: 512)" << std::endl;
    std::cout << "  -k, --topk <数量>       选择top-k tiles (默认: 5)" << std::endl;
    std::cout << "  --overlap <比例>        tile重叠比例 (默认: 0.25)" << std::endl;
    std::cout << "  --export-mode <模式>    导出模式: video/frames/json (默认: video)" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  " << programName << " -i /path/to/video.mp4 -o /path/to/output" << std::endl;
    std::cout << "  " << programName << " -i /path/to/videos/ -o /path/to/output -t 256 -k 3" << std::endl;
}

// 解析命令行参数
bool parseArguments(int argc, char* argv[], Config& config) {
    if (argc == 1) {
        showUsage(argv[0]);
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            showUsage(argv[0]);
            return false;
        }
        else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                config.INPUT_PATH = argv[++i];
            } else {
                std::cerr << "错误: " << arg << " 选项需要参数" << std::endl;
                return false;
            }
        }
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                config.OUTPUT_DIR = argv[++i];
            } else {
                std::cerr << "错误: " << arg << " 选项需要参数" << std::endl;
                return false;
            }
        }
        else if (arg == "-t" || arg == "--tile-size") {
            if (i + 1 < argc) {
                config.TILE_SIZE = std::stoi(argv[++i]);
            } else {
                std::cerr << "错误: " << arg << " 选项需要参数" << std::endl;
                return false;
            }
        }
        else if (arg == "-k" || arg == "--topk") {
            if (i + 1 < argc) {
                config.TOPK_TILES = std::stoi(argv[++i]);
            } else {
                std::cerr << "错误: " << arg << " 选项需要参数" << std::endl;
                return false;
            }
        }
        else if (arg == "--overlap") {
            if (i + 1 < argc) {
                config.OVERLAP = std::stof(argv[++i]);
            } else {
                std::cerr << "错误: " << arg << " 选项需要参数" << std::endl;
                return false;
            }
        }
        else if (arg == "--export-mode") {
            if (i + 1 < argc) {
                config.EXPORT_MODE = argv[++i];
            } else {
                std::cerr << "错误: " << arg << " 选项需要参数" << std::endl;
                return false;
            }
        }
        else {
            std::cerr << "错误: 未知选项 " << arg << std::endl;
            showUsage(argv[0]);
            return false;
        }
    }

    // 检查必需参数
    if (config.INPUT_PATH.empty()) {
        std::cerr << "错误: 必须指定输入路径 (-i 或 --input)" << std::endl;
        return false;
    }

    if (config.OUTPUT_DIR.empty()) {
        std::cerr << "错误: 必须指定输出目录 (-o 或 --output)" << std::endl;
        return false;
    }

    // 检查输入路径是否存在
    if (!fs::exists(config.INPUT_PATH)) {
        std::cerr << "错误: 输入路径不存在: " << config.INPUT_PATH << std::endl;
        return false;
    }

    return true;
}

/* ====================== 通用小工具 ====================== */

// 将字符串转为小写（用于比较扩展名等）
std::string toLower(const std::string& s) {
    std::string r = s;
    for (size_t i = 0; i < r.size(); ++i) {
        r[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(r[i])));
    }
    return r;
}

// 将索引左侧补零，生成固定宽度的字符串（例如 3 -> "003"）
std::string padIndex(size_t idx, int width) {
    std::ostringstream oss;
    oss << std::setw(width) << std::setfill('0') << idx;
    return oss.str();
}

/* percentile 计算：mat 为 CV_32F，返回 pct 百分位值 */
// 用于动态阈值计算，比如 85% 分位数
double getPercentile(const cv::Mat& mat, float pct) {
      // 1. 拉平到一维向量，用 double 存，减少精度误差
    std::vector<double> vals;
    vals.reserve(mat.total());

    CV_Assert(mat.type() == CV_32F || mat.type() == CV_64F);

    if (mat.isContinuous()) {
        if (mat.type() == CV_32F) {
            const float* p = mat.ptr<float>(0);
            for (size_t i = 0; i < mat.total(); ++i)
                vals.push_back(static_cast<double>(p[i]));
        } else { // CV_64F
            const double* p = mat.ptr<double>(0);
            vals.assign(p, p + mat.total());
        }
    } else {
        if (mat.type() == CV_32F) {
            for (int r = 0; r < mat.rows; ++r) {
                const float* row = mat.ptr<float>(r);
                for (int c = 0; c < mat.cols; ++c)
                    vals.push_back(static_cast<double>(row[c]));
            }
        } else { // CV_64F
            for (int r = 0; r < mat.rows; ++r) {
                const double* row = mat.ptr<double>(r);
                vals.insert(vals.end(), row, row + mat.cols);
            }
        }
    }

    if (vals.empty()) return 0.0;

    // 2. clamp 百分位范围
    if (pct < 0.0)  pct = 0.0;
    if (pct > 100.) pct = 100.;

    // 3. 排序（O(N log N)，一般对一张图问题不大）
    std::sort(vals.begin(), vals.end());

    if (vals.size() == 1) return vals[0];

    // 4. 线性插值位置：
    //    numpy 的逻辑大致是 q% * (N-1)，在相邻两点之间插值
    double pos  = pct / 100.0 * (static_cast<double>(vals.size() - 1));
    size_t idx0 = static_cast<size_t>(std::floor(pos));
    size_t idx1 = static_cast<size_t>(std::ceil(pos));
    double frac = pos - static_cast<double>(idx0);

    double v0 = vals[idx0];
    double v1 = vals[idx1];
    double v  = v0 + (v1 - v0) * frac;
    return v;
}

/* 使用 2~98 百分位做归一化，与 Python _norm01 类似 */
// 把输入矩阵线性映射到 [0,1]，剪裁极端值
cv::Mat normalize01Percentile(const cv::Mat& input) {
    CV_Assert(input.type() == CV_32F);
    double lo = getPercentile(input, 2.0f);
    double hi = getPercentile(input, 98.0f);
    cv::Mat out(input.size(), CV_32F);
    if (hi <= lo) {
        out.setTo(0);
        return out;
    }
    out = (input - lo) / (hi - lo);
    cv::threshold(out, out, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(out, out, 0.0, 0.0, cv::THRESH_TOZERO);
    return out;
}

/* 自适应 margin 函数：tile 内目标区域越小，边缘扩展比例越大 */
float adaptiveMarginFunc(const cv::Rect& bbox, int tileH, int tileW,
                         float base, float lo, float hi) {
    double areaRatio = (double)bbox.area() / std::max(1.0, (double)tileH * tileW);
    double v = base + (1.0 - std::min(areaRatio / 0.25, 1.0)) * 0.10;
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return static_cast<float>(v);
}

/* ====================== 视频列表枚举 ====================== */
// 将 INPUT_PATH 下满足扩展名 VIDEO_EXTS 的所有视频文件列出
std::vector<fs::path> listVideos(const fs::path& inputPath,
                                 const std::vector<std::string>& exts) {
    std::vector<fs::path> videos;
    std::vector<std::string> lowerExts;
    lowerExts.reserve(exts.size());
    for (size_t i = 0; i < exts.size(); ++i)
        lowerExts.push_back(toLower(exts[i]));

    // 如果是单文件，直接判断后缀是否属于视频
    if (fs::is_regular_file(inputPath)) {
        std::string ext = toLower(inputPath.extension().string());
        if (std::find(lowerExts.begin(), lowerExts.end(), ext) != lowerExts.end())
            videos.push_back(inputPath);
        return videos;
    }

    // 如果是目录，则递归遍历所有文件
    if (fs::is_directory(inputPath)) {
        for (fs::recursive_directory_iterator it(inputPath), end; it != end; ++it) {
            if (!fs::is_regular_file(*it)) continue;
            std::string ext = toLower(it->path().extension().string());
            if (std::find(lowerExts.begin(), lowerExts.end(), ext) != lowerExts.end())
                videos.push_back(it->path());
        }
    }
    std::sort(videos.begin(), videos.end());
    return videos;
}

/* ====================== VideoProcessor 类 ====================== */
// 核心处理类：负责单个视频的所有预处理步骤
class VideoProcessor {
public:
    // 用于保存单个视频处理结果和统计信息
    struct ProcessResult {
        bool success;                // 是否成功
        std::string videoPath;       // 视频路径
        TimeTracker timer;           // 时间跟踪器
        int frameCount;              // 总帧数
        int groupCount;              // 分组数（group_*）
        int tileCount;               // tile 个数
        int totalTilesProcessed;     // 实际处理的 tile 总数
        double avgTileTime;          // 平均每个 tile 耗时
        double minTileTime;          // 最快 tile 耗时
        double maxTileTime;          // 最慢 tile 耗时
        std::string errorMsg;        // 错误信息（若失败）

        ProcessResult()
            : success(false), frameCount(0), groupCount(0), tileCount(0),
              totalTilesProcessed(0), avgTileTime(0.0),
              minTileTime(0.0), maxTileTime(0.0) {}
    };

    explicit VideoProcessor(const Config& cfg) : config(cfg) {}

    // 处理单个视频：返回包含时间和统计信息的结果
    ProcessResult processVideo(const std::string& inputPathStr,
                               const std::string& outputRootStr) {
        ProcessResult result;
        result.videoPath = inputPathStr;
        TimeTracker timer;

        try {
            /* ---------- 1. 读取所有帧 ---------- */
            timer.startStage("read_frames");
            std::vector<cv::Mat> frames = readAllFrames(inputPathStr);
            timer.endStage();
            result.frameCount = static_cast<int>(frames.size());
            if (frames.empty())
                throw std::runtime_error("读取到0帧");

            /* ---------- 2. 按短边缩放（可选） ---------- */
            timer.startStage("resize");
            if (config.RESIZE_SHORT_SIDE > 0) {
                resizeShortSideFrames(frames, config.RESIZE_SHORT_SIDE);
            }
            timer.endStage();

            /* ---------- 3. ECC 稳像（可选） ---------- */
            if (config.STABILIZE_ECC) {
                timer.startStage("stabilization");
                eccStabilize(frames, config.ECC_WINSIZE, config.ECC_ITERS);
                timer.endStage();
            }

            const int height = frames[0].rows;
            const int width  = frames[0].cols;

            /* ---------- 4. 全局帧差热度图 ---------- */
            timer.startStage("global_heatmap");
            cv::Mat globalHeat = diffHeatmapFast(
                frames,
                config.HEAT_FRAME_STRIDE,
                config.HEAT_DOWNSCALE,
                cv::Size(width, height)
            );
            timer.endStage();

            /* ---------- 5. 根据打分图选 tile ---------- */
            timer.startStage("tile_selection");
            std::vector<cv::Rect> selectedTiles;
            std::vector<float>    tileScores;
            if (config.COVERAGE_MODE == "topk") {
                cv::Mat scoreMap;
                if (config.SCORE_MODE == "steam") {
                    // 蒸汽综合打分，复用 globalHeat 作为 diff
                    scoreMap = steamScoreMapFast(frames, globalHeat);
                } else {
                    // 仅使用帧差热度图
                    scoreMap = globalHeat;
                }
                std::pair<std::vector<cv::Rect>, std::vector<float> > sel =
                    selectTopKByMap(scoreMap,
                                    config.TILE_SIZE,
                                    config.OVERLAP,
                                    config.TOPK_TILES);
                selectedTiles = sel.first;
                tileScores    = sel.second;
            } else {
                // 全覆盖模式：不筛选 Top-K，划满整个画面
                selectedTiles = makeFullCoverTiles(height, width,
                                                   config.TILE_SIZE, config.OVERLAP);
                tileScores.assign(selectedTiles.size(), 0.0f);
            }
            timer.endStage();
            result.tileCount = static_cast<int>(selectedTiles.size());

            // 验证tile选择的正确性
            if (!validateTileConsistency(selectedTiles, frames)) {
                throw std::runtime_error("Tile选择验证失败");
            }

            // 调试输出：验证选择的tile坐标
            std::cout << "选择的tile数量: " << selectedTiles.size() << std::endl;
            for (size_t i = 0; i < selectedTiles.size(); ++i) {
                const cv::Rect& r = selectedTiles[i];
                std::cout << "Tile " << i << ": (" << r.x << ", " << r.y
                          << ", " << r.width << "x" << r.height
                          << "), 分数: " << tileScores[i] << std::endl;
            }

            // 每个视频对应一个子目录：OUTPUT_DIR / 视频文件名无扩展
            fs::path inPath(inputPathStr);
            std::string videoStem = inPath.stem().string();
            fs::path videoDir = fs::path(outputRootStr) / videoStem;
            fs::create_directories(videoDir);

            /* ---------- 6. 保存可视化首帧 vis.jpg ---------- */
            timer.startStage("visualization");
            if (config.SAVE_VIS && !frames.empty()) {
                cv::Mat vis = drawTilesVis(
                    frames[0],
                    selectedTiles,
                    (config.COVERAGE_MODE == "topk" ? &tileScores : NULL)
                );
                cv::imwrite((videoDir / "vis.jpg").string(), vis);
            }
            timer.endStage();

            /* ---------- 7. 保存 tiles_meta.json ---------- */
            timer.startStage("metadata_save");
            saveTilesMeta(
                (videoDir / "tiles_meta.json").string(),
                inputPathStr, height, width,
                config, selectedTiles, tileScores
            );
            timer.endStage();

            // 若只需要 JSON/vis，不做后续切块导出
            if (toLower(config.EXPORT_MODE) == "json") {
                timer.endStage();
                result.success = true;
                result.timer = timer;
                return result;
            }

            /* ---------- 8. 按 clip 分组（group_XXXX） ---------- */
            timer.startStage("group_organization");
            std::vector<std::vector<cv::Mat> > groups;
            if (config.GROUP_TO_CLIPS) {
                groups = makeClips(frames, config.CLIP_LEN,
                                   config.WINDOW_STRIDE_FRAMES);
            } else {
                groups.push_back(frames);
            }
            timer.endStage();
            result.groupCount = static_cast<int>(groups.size());

            /* ---------- 9. 获取 FPS ---------- */
            timer.startStage("fps_extraction");
            double fps = 25.0;
            {
                cv::VideoCapture cap(inputPathStr);
                if (cap.isOpened()) {
                    double f = cap.get(cv::CAP_PROP_FPS);
                    if (f > 1e-3) fps = f;
                }
            }
            timer.endStage();

            /* ---------- 10. 遍历 group & tile 做裁剪 + 背景净化 + 导出 ---------- */
            timer.startStage("tile_processing");
            int processedTiles = 0;
            std::vector<double> tileTimes;

            // 用于调试：记录每个tile的处理信息
            std::vector<std::string> tileDebugInfo;

            for (size_t gi = 0; gi < groups.size(); ++gi) {
                const std::vector<cv::Mat>& g = groups[gi];
                if (g.empty()) continue;

                // group_XXXX 目录
                fs::path groupDir = videoDir / ("group_" + padIndex(gi, 4));
                fs::create_directories(groupDir);

                for (size_t ti = 0; ti < selectedTiles.size(); ++ti) {
                    TimeTracker::Clock::time_point startTileT = TimeTracker::Clock::now();

                    const cv::Rect& tile = selectedTiles[ti];

                    // 调试信息：记录当前处理的tile信息
                    std::ostringstream debugMsg;
                    debugMsg << "Group " << gi << ", Tile " << ti
                             << ": 全局坐标(" << tile.x << ", " << tile.y
                             << ", " << tile.width << "x" << tile.height << ")";
                    tileDebugInfo.push_back(debugMsg.str());
                    std::cout << debugMsg.str() << std::endl;

                    // 1) 验证tile坐标是否有效
                    if (tile.x < 0 || tile.y < 0 ||
                        tile.x + tile.width > g[0].cols ||
                        tile.y + tile.height > g[0].rows) {
                        std::cerr << "警告: Tile " << ti << " 坐标越界，跳过处理" << std::endl;
                        continue;
                    }

                    // 2) 先从该 group 的每帧中提取 tile 区域
                    std::vector<cv::Mat> crops;
                    crops.reserve(g.size());
                    for (size_t k = 0; k < g.size(); ++k) {
                        // 确保tile在图像范围内
                        cv::Rect safeTile = tile;
                        safeTile.x = std::max(0, std::min(safeTile.x, g[k].cols - 1));
                        safeTile.y = std::max(0, std::min(safeTile.y, g[k].rows - 1));
                        safeTile.width = std::min(safeTile.width, g[k].cols - safeTile.x);
                        safeTile.height = std::min(safeTile.height, g[k].rows - safeTile.y);

                        if (safeTile.width <= 0 || safeTile.height <= 0) {
                            std::cerr << "错误: Tile " << ti << " 尺寸无效，跳过" << std::endl;
                            continue;
                        }

                        cv::Mat roi = g[k](safeTile);
                        crops.push_back(roi.clone());
                    }

                    if (crops.empty()) {
                        std::cerr << "警告: Tile " << ti << " 没有有效帧，跳过" << std::endl;
                        continue;
                    }

                    // maskSel 表示紧致后的运动区域掩膜（0/255）
                    cv::Mat maskSel;
                    cv::Rect actualCropRegion = tile; // 记录实际裁剪的全局区域

                    if (config.AUTO_TIGHTEN) {
                        // 3) 用 globalHeat 的该 tile ROI 做二值阈值 + 连通域
                        cv::Mat heatROI = globalHeat(tile);
                        cv::Mat mask;
                        if (toLower(config.TIGHTEN_THR_MODE) == "otsu") {
                            // OTSU 自动阈值
                            cv::Mat heat8;
                            heatROI.convertTo(heat8, CV_8U, 255.0);
                            cv::threshold(heat8, mask, 0, 255,
                                        cv::THRESH_BINARY | cv::THRESH_OTSU);
                        } else {
                            // 百分位阈值（比如 85%）
                            double thr = getPercentile(heatROI, config.TIGHTEN_THR_PCT);
                            cv::threshold(heatROI, mask, thr, 255, cv::THRESH_BINARY);
                        }
                        mask.convertTo(mask, CV_8U);

                        // 开运算 + 闭运算，去噪和填洞
                        cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                                        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
                        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                                        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5)));

                        // 4) 查找外部轮廓，取面积最大的一块
                        std::vector<std::vector<cv::Point> > contours;
                        cv::findContours(mask, contours, cv::RETR_EXTERNAL,
                                        cv::CHAIN_APPROX_SIMPLE);

                        // expandedBox 是相对于 tile 的局部坐标
                        cv::Rect expandedBox(0, 0, tile.width, tile.height);
                        if (!contours.empty()) {
                            size_t best = 0;
                            double maxA = cv::contourArea(contours[0]);
                            for (size_t c = 1; c < contours.size(); ++c) {
                                double a = cv::contourArea(contours[c]);
                                if (a > maxA) { maxA = a; best = c; }
                            }
                            cv::Rect bbox = cv::boundingRect(contours[best]);
                            if (bbox.area() >= config.TIGHTEN_MIN_AREA) {
                                // 5) 在中位帧上做边缘密度过滤（防石头）
                                cv::Mat midFrame = g[g.size() / 2];
                                cv::Mat roi = midFrame(tile)(bbox);
                                if (config.TIGHTEN_EDGE_CHECK_DOWNSCALE < 0.999f) {
                                    int nw = std::max(1, (int)std::round(roi.cols * config.TIGHTEN_EDGE_CHECK_DOWNSCALE));
                                    int nh = std::max(1, (int)std::round(roi.rows * config.TIGHTEN_EDGE_CHECK_DOWNSCALE));
                                    cv::resize(roi, roi, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
                                }
                                cv::Mat gray, edges;
                                cv::cvtColor(roi, gray, cv::COLOR_RGB2GRAY);
                                cv::Canny(gray, edges, 50, 150);
                                double edgeDensity =
                                    (double)cv::countNonZero(edges) / (double)edges.total();
                                if (edgeDensity <= config.EDGE_DENSITY_MAX) {
                                    // 6) 通过高纹理过滤后，按 margin 扩展 bbox
                                    float margin = config.TIGHTEN_MARGIN;
                                    if (config.ADAPTIVE_MARGIN) {
                                        margin = adaptiveMarginFunc(
                                            bbox, tile.height, tile.width,
                                            config.ADAPTIVE_MARGIN_BASE,
                                            config.ADAPTIVE_MARGIN_LO,
                                            config.ADAPTIVE_MARGIN_HI);
                                    }
                                    int marginY = (int)std::round(bbox.height * margin);
                                    int marginX = (int)std::round(bbox.width  * margin);

                                    expandedBox.x = std::max(0, bbox.x - marginX);
                                    expandedBox.y = std::max(0, bbox.y - marginY);
                                    expandedBox.width  = std::min(tile.width  - expandedBox.x,
                                                                bbox.width  + 2 * marginX);
                                    expandedBox.height = std::min(tile.height - expandedBox.y,
                                                                bbox.height + 2 * marginY);

                                    // 关键修复：计算实际裁剪的全局区域
                                    actualCropRegion.x = tile.x + expandedBox.x;
                                    actualCropRegion.y = tile.y + expandedBox.y;
                                    actualCropRegion.width = expandedBox.width;
                                    actualCropRegion.height = expandedBox.height;
                                }
                            }
                        }

                        // 7) 用 expandedBox 裁剪 crops，同时裁剪 mask
                        std::vector<cv::Mat> tmp;
                        tmp.reserve(crops.size());
                        for (size_t i = 0; i < crops.size(); ++i) {
                            tmp.push_back(crops[i](expandedBox).clone());
                        }
                        crops.swap(tmp);

                        maskSel = mask(expandedBox).clone();
                    } else {
                        // 不做紧致裁剪时，maskSel 全 255
                        maskSel = cv::Mat(crops[0].rows, crops[0].cols,
                                        CV_8U, cv::Scalar(255));
                    }

                    // 8) resize 到模型输入尺寸（RESIZE_OUT）
                    for (size_t i = 0; i < crops.size(); ++i) {
                        cv::resize(crops[i], crops[i],
                                cv::Size(config.RESIZE_OUT, config.RESIZE_OUT),
                                0, 0, cv::INTER_AREA);
                    }
                    cv::Mat maskResized;
                    cv::resize(maskSel, maskResized,
                            cv::Size(config.RESIZE_OUT, config.RESIZE_OUT),
                            0, 0, cv::INTER_NEAREST);

                    // 9) 背景净化
                    applyBgMode(crops, maskResized,
                                config.BG_MODE, config.BG_BLUR_K);

                    // 10) 导出：frames 或 video
                    std::string expMode = toLower(config.EXPORT_MODE);
                    if (expMode == "frames") {
                        // 每个 tile 一个目录，存多帧 jpg
                        fs::path tileDir = groupDir /
                            ("tile_" + padIndex(ti, 3));
                        saveFramesDir(tileDir.string(), crops, config.JPEG_QUALITY);

                        // 保存tile信息到文本文件用于调试
                        std::ofstream infoFile((tileDir / "tile_info.txt").string());
                        if (infoFile.is_open()) {
                            infoFile << "原始tile: " << tile.x << ", " << tile.y
                                    << ", " << tile.width << "x" << tile.height << "\n";
                            infoFile << "实际裁剪: " << actualCropRegion.x << ", " << actualCropRegion.y
                                    << ", " << actualCropRegion.width << "x" << actualCropRegion.height << "\n";
                            infoFile << "组索引: " << gi << ", Tile索引: " << ti << "\n";
                            infoFile.close();
                        }
                    } else if (expMode == "video") {
                        // 每个 tile 一个 mp4
                        fs::path tilePath = groupDir /
                            ("tile_" + padIndex(ti, 3) + ".mp4");
                        if (config.SKIP_EXISTING && fs::exists(tilePath)) {
                            // 若已存在且允许跳过，就不再重写（仍计时）
                        } else {
                            saveVideo(tilePath.string(), crops, fps);
                        }
                    }

                    // 11) 记录当前 tile 耗时
                    TimeTracker::Clock::time_point endTileT = TimeTracker::Clock::now();
                    double tileSec =
                        std::chrono::duration<double>(endTileT - startTileT).count();
                    tileTimes.push_back(tileSec);
                    processedTiles++;
                }
            }
            timer.endStage();  // tile_processing 结束

            // 统计 tile 时间信息
            result.totalTilesProcessed = processedTiles;
            if (!tileTimes.empty()) {
                double sum = 0.0;
                double mn = tileTimes[0];
                double mx = tileTimes[0];
                for (size_t i = 0; i < tileTimes.size(); ++i) {
                    double v = tileTimes[i];
                    sum += v;
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
                result.avgTileTime = sum / tileTimes.size();
                result.minTileTime = mn;
                result.maxTileTime = mx;
            }

            timer.endStage();  // 整体结束
            result.success = true;
            result.timer = timer;
        } catch (const std::exception& e) {
            // 捕获异常，标记失败
            timer.endStage();
            result.success = false;
            result.errorMsg = e.what();
            result.timer = timer;
        }
        return result;
    }

private:
    Config config;

    // 检查tile选择和处理的一致性
    bool validateTileConsistency(const std::vector<cv::Rect>& selectedTiles,
                                 const std::vector<cv::Mat>& frames) {
        if (selectedTiles.empty()) {
            std::cerr << "错误: 没有选择任何tile" << std::endl;
            return false;
        }

        if (frames.empty()) {
            std::cerr << "错误: 没有帧数据" << std::endl;
            return false;
        }

        int frameHeight = frames[0].rows;
        int frameWidth = frames[0].cols;

        std::cout << "帧尺寸: " << frameWidth << "x" << frameHeight << std::endl;
        std::cout << "选择的tile数量: " << selectedTiles.size() << std::endl;

        for (size_t i = 0; i < selectedTiles.size(); ++i) {
            const cv::Rect& tile = selectedTiles[i];
            std::cout << "Tile " << i << ": "
                      << "x=" << tile.x << ", y=" << tile.y
                      << ", w=" << tile.width << ", h=" << tile.height;

            // 检查tile是否在帧范围内
            if (tile.x < 0 || tile.y < 0 ||
                tile.x + tile.width > frameWidth ||
                tile.y + tile.height > frameHeight) {
                std::cout << " [越界!]" << std::endl;
                return false;
            } else {
                std::cout << " [有效]" << std::endl;
            }
        }

        return true;
    }

    /* ---------- 读取整个视频为 RGB 帧数组 ---------- */
    std::vector<cv::Mat> readAllFrames(const std::string& videoPath) {
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened())
            throw std::runtime_error("无法打开视频: " + videoPath);
        std::vector<cv::Mat> frames;
        cv::Mat frame;
        while (cap.read(frame)) {
            cv::Mat rgb;
            cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
            frames.push_back(rgb.clone());
        }
        if (frames.empty())
            throw std::runtime_error("读取到0帧: " + videoPath);
        return frames;
    }

    /* ---------- 按短边缩放所有帧 ---------- */
    void resizeShortSideFrames(std::vector<cv::Mat>& frames, int shortSide) {
        if (frames.empty() || shortSide <= 0) return;
        int h = frames[0].rows;
        int w = frames[0].cols;
        int newH, newW;
        if (h < w) {
            newH = shortSide;
            newW = static_cast<int>(std::round(w * (shortSide / (double)h)));
        } else {
            newW = shortSide;
            newH = static_cast<int>(std::round(h * (shortSide / (double)w)));
        }
        for (size_t i = 0; i < frames.size(); ++i) {
            cv::resize(frames[i], frames[i], cv::Size(newW, newH),
                       0, 0, cv::INTER_AREA);
        }
    }

    /* ---------- 按比例下采样一组帧 ---------- */
    std::vector<cv::Mat> downscaleFrames(const std::vector<cv::Mat>& frames,
                                         float scale) {
        if (scale >= 0.999f) return frames;
        std::vector<cv::Mat> result;
        result.reserve(frames.size());
        for (size_t i = 0; i < frames.size(); ++i) {
            const cv::Mat& f = frames[i];
            int nw = std::max(1, (int)std::round(f.cols * scale));
            int nh = std::max(1, (int)std::round(f.rows * scale));
            cv::Mat resized;
            cv::resize(f, resized, cv::Size(nw, nh),
                       0, 0, cv::INTER_AREA);
            result.push_back(resized);
        }
        return result;
    }

    /* ---------- 计算全局帧差热度图（归一化到 0~1） ---------- */
    cv::Mat diffHeatmapFast(const std::vector<cv::Mat>& frames,
                            int frameStride,
                            float downscale,
                            const cv::Size& upsizeSize) {
    // 1. 帧采样 - 确保与Python一致
    std::vector<cv::Mat> sampled;
    int step = std::max(1, frameStride);
    for (size_t i = 0; i < frames.size(); i += step) {
        sampled.push_back(frames[i]);
    }
    if (sampled.empty()) sampled.push_back(frames[0]);

    // 2. 下采样 - 使用精确计算
    if (downscale < 0.999f) {
        std::vector<cv::Mat> downsampled;
        for (const auto& frame : sampled) {
            int newWidth = static_cast<int>(std::round(frame.cols * downscale));
            int newHeight = static_cast<int>(std::round(frame.rows * downscale));
            newWidth = std::max(1, newWidth);
            newHeight = std::max(1, newHeight);
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(newWidth, newHeight),
                      0, 0, cv::INTER_AREA);
            downsampled.push_back(resized);
        }
        sampled = downsampled;
    }

    if (sampled.size() <= 1) {
        return cv::Mat::zeros(sampled[0].size(), CV_32F);
    }

    // 3. 转灰度并计算帧差 - 精确匹配Python
    std::vector<cv::Mat> grays;
    for (const auto& frame : sampled) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        gray.convertTo(gray, CV_32F);
        grays.push_back(gray);
    }

    // 4. 累积帧差 - 与Python完全一致
    cv::Mat totalDiff = cv::Mat::zeros(grays[0].size(), CV_32F);
    for (size_t i = 1; i < grays.size(); ++i) {
        cv::Mat diff;
        cv::absdiff(grays[i], grays[i-1], diff);
        totalDiff += diff;
    }

    // 5. 平均 - 使用精确除法
    cv::Mat heat = totalDiff / static_cast<float>(grays.size() - 1);

    // 6. 归一化 - 使用与Python相同的2~98百分位
    double minVal, maxVal;
    cv::minMaxLoc(heat, &minVal, &maxVal);
    if (maxVal <= minVal) {
        heat.setTo(0);
    } else {
        heat = (heat - minVal) / (maxVal - minVal);
    }

    // 7. 高斯模糊 - 使用相同参数
    cv::GaussianBlur(heat, heat, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);

    // 8. 上采样 - 使用双线性插值
    if (upsizeSize.width > 0 && upsizeSize.height > 0 &&
        (heat.rows != upsizeSize.height || heat.cols != upsizeSize.width)) {
        cv::resize(heat, heat, upsizeSize, 0, 0, cv::INTER_LINEAR);
    }

    return heat;
    }

    /* ---------- 计算"白度"热度图 ---------- */
    cv::Mat whitenessMapFast(const std::vector<cv::Mat>& frames,
                             int frameStride,
                             float downscale,
                             const cv::Size& upsizeSize) {
        std::vector<cv::Mat> sampled;
        int step = std::max(1, frameStride);
        for (size_t i = 0; i < frames.size(); i += (size_t)step)
            sampled.push_back(frames[i]);
        if (sampled.empty()) sampled.push_back(frames[0]);
        if (downscale < 0.999f)
            sampled = downscaleFrames(sampled, downscale);

        // 累积 HSV 中的 V 和 S 分量
        cv::Mat totalV = cv::Mat::zeros(sampled[0].size(), CV_32F);
        cv::Mat totalS = cv::Mat::zeros(sampled[0].size(), CV_32F);
        for (size_t i = 0; i < sampled.size(); ++i) {
            cv::Mat hsv;
            cv::cvtColor(sampled[i], hsv, cv::COLOR_RGB2HSV);
            std::vector<cv::Mat> ch;
            cv::split(hsv, ch);
            cv::Mat v, s;
            ch[2].convertTo(v, CV_32F);
            ch[1].convertTo(s, CV_32F);
            totalV += v / 255.0f;
            totalS += s / 255.0f;
        }
        cv::Mat avgV = totalV / (float)sampled.size();
        cv::Mat avgS = totalS / (float)sampled.size();

        // "白度" ~ 高亮度、低饱和
        cv::Mat white = avgV.mul(1.0f - avgS);
        white = normalize01Percentile(white);
        if (upsizeSize.width > 0 && upsizeSize.height > 0 &&
            (white.rows != upsizeSize.height || white.cols != upsizeSize.width)) {
            cv::resize(white, white, upsizeSize, 0, 0, cv::INTER_LINEAR);
        }
        return white;
    }

    /* ---------- 计算边缘密度图 ---------- */
    cv::Mat edgeDensityMapFast(const std::vector<cv::Mat>& frames,
                               int frameStride,
                               float downscale,
                               const cv::Size& upsizeSize) {
        std::vector<cv::Mat> sampled;
        int step = std::max(1, frameStride);
        for (size_t i = 0; i < frames.size(); i += (size_t)step)
            sampled.push_back(frames[i]);
        if (sampled.empty()) sampled.push_back(frames[0]);
        if (downscale < 0.999f)
            sampled = downscaleFrames(sampled, downscale);

        cv::Mat totalEdges = cv::Mat::zeros(sampled[0].size(), CV_32F);
        for (size_t i = 0; i < sampled.size(); ++i) {
            cv::Mat gray;
            cv::cvtColor(sampled[i], gray, cv::COLOR_RGB2GRAY);
            cv::Mat edges;
            cv::Canny(gray, edges, 50, 150);
            cv::Mat e32;
            edges.convertTo(e32, CV_32F);
            e32 /= 255.0f;
            totalEdges += e32;
        }
        cv::Mat density = totalEdges / (float)sampled.size();
        density = normalize01Percentile(density);
        if (upsizeSize.width > 0 && upsizeSize.height > 0 &&
            (density.rows != upsizeSize.height || density.cols != upsizeSize.width)) {
            cv::resize(density, density, upsizeSize, 0, 0, cv::INTER_LINEAR);
        }
        return density;
    }

    /* ---------- 蒸汽综合打分图：w_diff * diff + w_white * white + w_smooth * (1-edge) ---------- */
    cv::Mat steamScoreMapFast(const std::vector<cv::Mat>& frames,
                              const cv::Mat& reuseDiff) {
        cv::Size originalSize(frames[0].cols, frames[0].rows);
        cv::Mat diff;
        if (reuseDiff.empty()) {
            diff = diffHeatmapFast(frames, config.SCORE_FRAME_STRIDE,
                                   config.SCORE_DOWNSCALE, originalSize);
        } else {
            diff = reuseDiff;
        }
        cv::Mat white = whitenessMapFast(frames, config.SCORE_FRAME_STRIDE,
                                         config.SCORE_DOWNSCALE, originalSize);
        cv::Mat smooth = 1.0f - edgeDensityMapFast(frames, config.SCORE_FRAME_STRIDE,
                                                   config.SCORE_DOWNSCALE, originalSize);
        cv::Mat score = config.STEAM_W_DIFF * diff +
                        config.STEAM_W_WHITE * white +
                        config.STEAM_W_SMOOTH * smooth;
        cv::GaussianBlur(score, score, cv::Size(5,5), 0);
        return score;
    }

    /* ---------- 生成完整覆盖的 tile 网格 ---------- */
    std::vector<cv::Rect> makeFullCoverTiles(int height, int width,
                                             int tileSize, float overlap) {
        if (tileSize <= 0)
            throw std::invalid_argument("TILE_SIZE必须>0");
        int stride = std::max(1, (int)std::round(tileSize * (1.0f - overlap)));

        std::vector<int> yStarts;
        std::vector<int> xStarts;

        // Y 方向起点
        if (height <= tileSize) {
            yStarts.push_back(0);
        } else {
            for (int y = 0; y <= height - tileSize; y += stride)
                yStarts.push_back(y);
            int last = height - tileSize;
            if (yStarts.back() != last) yStarts.push_back(last);
        }

        // X 方向起点
        if (width <= tileSize) {
            xStarts.push_back(0);
        } else {
            for (int x = 0; x <= width - tileSize; x += stride)
                xStarts.push_back(x);
            int last = width - tileSize;
            if (xStarts.back() != last) xStarts.push_back(last);
        }

        // 组合为 Rect
        std::vector<cv::Rect> tiles;
        for (size_t iy = 0; iy < yStarts.size(); ++iy) {
            for (size_t ix = 0; ix < xStarts.size(); ++ix) {
                int y = yStarts[iy];
                int x = xStarts[ix];
                int yEnd = std::min(height, y + tileSize);
                int xEnd = std::min(width,  x + tileSize);
                tiles.push_back(cv::Rect(x, y, xEnd - x, yEnd - y));
            }
        }
        return tiles;
    }

    /* ---------- 根据 scoreMap 选 Top-K tile ---------- */
    std::pair<std::vector<cv::Rect>, std::vector<float> >
    selectTopKByMap(const cv::Mat& scoreMap,
                    int tileSize,
                    float overlap,
                    int k) {
        int height = scoreMap.rows;
        int width  = scoreMap.cols;
        std::vector<cv::Rect> tiles = makeFullCoverTiles(height, width,
                                                         tileSize, overlap);
        std::vector<float> scores(tiles.size());
        for (size_t i = 0; i < tiles.size(); ++i) {
            cv::Mat roi = scoreMap(tiles[i]);
            scores[i] = (float)cv::mean(roi)[0];
        }

        // indices 排序，按分数从大到小
        std::vector<size_t> idx(scores.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&](size_t a, size_t b){ return scores[a] > scores[b]; });
        int actualK = std::min(k, (int)tiles.size());
        std::vector<cv::Rect> selTiles;
        std::vector<float>    selScores;
        for (int i = 0; i < actualK; ++i) {
            selTiles.push_back(tiles[idx[i]]);
            selScores.push_back(scores[idx[i]]);
        }
        return std::make_pair(selTiles, selScores);
    }

    /* ---------- 滑窗分组：生成 group clips ---------- */
    std::vector<std::vector<cv::Mat> >
    makeClips(const std::vector<cv::Mat>& frames,
              int clipLen, int stride) {
        std::vector<std::vector<cv::Mat> > groups;
        int T = (int)frames.size();
        if (clipLen <= 0) {
            groups.push_back(frames);
            return groups;
        }
        int step = std::max(1, stride);
        int limit = std::max(1, T - clipLen + 1);
        for (int s = 0; s < limit; s += step) {
            std::vector<cv::Mat> seg;
            int end = std::min(s + clipLen, T);
            for (int i = s; i < end; ++i) seg.push_back(frames[i]);
            // 不足 clipLen 的部分重复最后一帧补齐
            while ((int)seg.size() < clipLen && !seg.empty())
                seg.push_back(seg.back());
            groups.push_back(seg);
        }
        if (groups.empty()) groups.push_back(frames);
        return groups;
    }

    /* ---------- 背景净化：根据掩膜将背景替换为模糊/黑/灰/均值 ---------- */
    void applyBgMode(std::vector<cv::Mat>& frames,
                     const cv::Mat& mask, // 0/255，1=运动区域，0=背景
                     const std::string& mode,
                     int blurK) {
        if (toLower(mode) == "keep") return;   // 不做任何处理
        int k = blurK | 1; // 确保为奇数
        cv::Mat maskFloat;
        mask.convertTo(maskFloat, CV_32F, 1.0 / 255.0);
        cv::Mat invFloat = 1.0f - maskFloat;

        for (size_t i = 0; i < frames.size(); ++i) {
            cv::Mat& im = frames[i];
            cv::Mat bg;
            std::string m = toLower(mode);
            if (m == "black") {
                bg = cv::Mat::zeros(im.size(), im.type());
            } else if (m == "gray") {
                bg = cv::Mat(im.size(), im.type(), cv::Scalar(128,128,128));
            } else if (m == "mean") {
                cv::Scalar meanVal = cv::mean(im);
                bg = cv::Mat(im.size(), im.type(), meanVal);
            } else {
                // blur 模式
                cv::GaussianBlur(im, bg, cv::Size(k,k), 0);
            }

            cv::Mat im32, bg32;
            im.convertTo(im32, CV_32F);
            bg.convertTo(bg32, CV_32F);

            std::vector<cv::Mat> chIm, chBg;
            cv::split(im32, chIm);
            cv::split(bg32, chBg);
            for (size_t c = 0; c < chIm.size(); ++c) {
                // 前景 * mask + 背景 * (1-mask)
                chIm[c] = chIm[c].mul(maskFloat) + chBg[c].mul(invFloat);
            }
            cv::merge(chIm, im32);
            im32.convertTo(im, CV_8U);
        }
    }

    /* ---------- 将帧序列保存为 JPG ---------- */
    void saveFramesDir(const std::string& outDir,
                       const std::vector<cv::Mat>& frames,
                       int jpegQuality) {
        fs::create_directories(outDir);
        std::vector<int> params;
        params.push_back(cv::IMWRITE_JPEG_QUALITY);
        params.push_back(jpegQuality);
        for (size_t i = 0; i < frames.size(); ++i) {
            fs::path p = fs::path(outDir) /
                         ("frame_" + padIndex(i, 6) + ".jpg");
            cv::Mat bgr;
            cv::cvtColor(frames[i], bgr, cv::COLOR_RGB2BGR);
            cv::imwrite(p.string(), bgr, params);
        }
    }

    /* ---------- 将帧序列保存为 mp4 视频 ---------- */
    void saveVideo(const std::string& outPath,
                   const std::vector<cv::Mat>& frames,
                   double fps) {
        if (frames.empty()) return;
        fs::create_directories(fs::path(outPath).parent_path());
        cv::Size sz(frames[0].cols, frames[0].rows);
        int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
        cv::VideoWriter writer;
        bool isColor = (frames[0].type() == CV_8UC3);
        writer.open(outPath, fourcc, fps, sz, isColor);
        if (!writer.isOpened()) {
            std::cerr << "无法写入: " << outPath << std::endl;
            return;
        }
        for (size_t i = 0; i < frames.size(); ++i) {
            cv::Mat bgr;
            cv::cvtColor(frames[i], bgr, cv::COLOR_RGB2BGR);
            writer.write(bgr);
        }
        writer.release();
    }

    /* ---------- 绘制首帧+Tile 可视化图 ---------- */
    cv::Mat drawTilesVis(const cv::Mat& rgb,
                         const std::vector<cv::Rect>& tiles,
                         const std::vector<float>* scores) {
        cv::Mat vis;
        cv::cvtColor(rgb, vis, cv::COLOR_RGB2BGR);

        for (size_t i = 0; i < tiles.size(); ++i) {
            float s = 0.f;
            if (scores && i < scores->size()) s = (*scores)[i];
            if (s < 0.f) s = 0.f;
            if (s > 1.f) s = 1.f;

            // 根据分数选择颜色
            cv::Scalar color;
            if (scores) {
                // 分数越高越偏红，越低越偏蓝
                color = cv::Scalar(255 * s, 120 * (1 - s), 255 * (1 - s));
            } else {
                color = cv::Scalar(0, 255, 0); // 全覆盖模式用绿色
            }

            cv::rectangle(vis, tiles[i], color, 2);

            // 显示tile索引和分数
            std::string label = std::to_string(i);
            if (scores) {
                label += "(" + std::to_string(s).substr(0, 4) + ")";
            }

            cv::putText(vis, label,
                        cv::Point(tiles[i].x + 5, tiles[i].y + 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

            // 在tile中心画一个小点表示处理顺序
            cv::Point center(tiles[i].x + tiles[i].width / 2,
                            tiles[i].y + tiles[i].height / 2);
            cv::circle(vis, center, 3, color, -1);
        }
        return vis;
    }

    /* ---------- 输出 tiles_meta.json （简易 JSON） ---------- */
    void saveTilesMeta(const std::string& metaPath,
                       const std::string& videoPath,
                       int height, int width,
                       const Config& cfg,
                       const std::vector<cv::Rect>& tiles,
                       const std::vector<float>& scores) {
        std::ofstream ofs(metaPath.c_str());
        if (!ofs.is_open()) return;
        ofs << "{\n";
        ofs << "  \"video\": \"" << videoPath << "\",\n";
        ofs << "  \"size_hw\": [" << height << ", " << width << "],\n";
        ofs << "  \"tile_size\": " << cfg.TILE_SIZE << ",\n";
        ofs << "  \"overlap\": " << cfg.OVERLAP << ",\n";
        ofs << "  \"coverage_mode\": \"" << cfg.COVERAGE_MODE << "\",\n";
        ofs << "  \"score_mode\": \"" << cfg.SCORE_MODE << "\",\n";
        ofs << "  \"tiles\": [\n";
        for (size_t i = 0; i < tiles.size(); ++i) {
            const cv::Rect& r = tiles[i];
            ofs << "    [" << r.y << ", " << r.x
                << ", " << (r.y + r.height)
                << ", " << (r.x + r.width) << "]";
            if (i + 1 < tiles.size()) ofs << ",";
            ofs << "\n";
        }
        ofs << "  ],\n";
        ofs << "  \"scores\": [\n";
        for (size_t i = 0; i < scores.size(); ++i) {
            ofs << "    " << scores[i];
            if (i + 1 < scores.size()) ofs << ",";
            ofs << "\n";
        }
        ofs << "  ]\n";
        ofs << "}\n";
    }

    /* ---------- ECC 视频稳像（可选） ---------- */
    void eccStabilize(std::vector<cv::Mat>& frames,
                      int winsize, int iters) {
        if (frames.size() <= 1) return;
        cv::Mat refGray;
        cv::cvtColor(frames[0], refGray, cv::COLOR_RGB2GRAY);
        cv::Size sz(refGray.cols, refGray.rows);
        cv::TermCriteria criteria(
            cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
            iters, 1e-6);
        int warp_mode = cv::MOTION_AFFINE;

        for (size_t i = 1; i < frames.size(); ++i) {
            cv::Mat curGray;
            cv::cvtColor(frames[i], curGray, cv::COLOR_RGB2GRAY);
            cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
            try {
                cv::findTransformECC(refGray, curGray, warp,
                                     warp_mode, criteria,
                                     cv::noArray(), winsize);
                cv::Mat aligned;
                cv::warpAffine(frames[i], aligned, warp, sz,
                               cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                               cv::BORDER_REFLECT);
                frames[i] = aligned.clone();
            } catch (...) {
                // ECC 失败则保持原帧
            }
        }
    }
};

/* ====================== main：多视频入口 ====================== */
int main(int argc, char* argv[]) {
    Config config;

    // 解析命令行参数
    if (!parseArguments(argc, argv, config)) {
        return 1;
    }

    // 显示配置信息
    std::cout << "输入路径: " << config.INPUT_PATH << std::endl;
    std::cout << "输出目录: " << config.OUTPUT_DIR << std::endl;
    std::cout << "Tile尺寸: " << config.TILE_SIZE << std::endl;
    std::cout << "Top-K数量: " << config.TOPK_TILES << std::endl;
    std::cout << "重叠比例: " << config.OVERLAP << std::endl;
    std::cout << "导出模式: " << config.EXPORT_MODE << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    VideoProcessor processor(config);

    fs::path inPath(config.INPUT_PATH);
    fs::path outRoot(config.OUTPUT_DIR);
    fs::create_directories(outRoot);

    // 递归列出所有需要处理的视频
    std::vector<fs::path> videos = listVideos(inPath, config.VIDEO_EXTS);
    if (videos.empty()) {
        std::cerr << "未找到视频: " << inPath << std::endl;
        return 1;
    }

    std::cout << "发现 " << videos.size()
              << " 个视频，输出到: " << outRoot << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    TimeTracker::Clock::time_point totalStart = TimeTracker::Clock::now();
    int processedCount = 0;
    int failedCount    = 0;
    std::vector<VideoProcessor::ProcessResult> results;

    // 逐个视频处理
    for (size_t i = 0; i < videos.size(); ++i) {
        const fs::path& v = videos[i];
        std::cout << "\n处理视频 " << (i+1) << "/" << videos.size()
                  << ": " << v.filename().string() << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        VideoProcessor::ProcessResult res =
            processor.processVideo(v.string(), outRoot.string());
        results.push_back(res);

        double totalT = res.timer.totalTime();

        if (res.success) {
            processedCount++;
            std::cout << "✓ 处理成功 - 总耗时: "
                      << formatTimeDetailed(totalT) << std::endl;
            std::cout << "  帧数: " << res.frameCount
                      << ", 分组数: " << res.groupCount
                      << ", Tile数: " << res.tileCount << std::endl;
            std::cout << "  各阶段耗时:" << std::endl;

            // 阶段名称映射（英文 -> 中文）
            std::map<std::string,std::string> stageNames;
            stageNames["read_frames"]        = "读取帧";
            stageNames["resize"]             = "缩放处理";
            stageNames["stabilization"]      = "稳像处理";
            stageNames["global_heatmap"]     = "全局热度图";
            stageNames["tile_selection"]     = "Tile选择";
            stageNames["visualization"]      = "可视化";
            stageNames["metadata_save"]      = "元数据保存";
            stageNames["group_organization"] = "分组组织";
            stageNames["fps_extraction"]     = "FPS提取";
            stageNames["tile_processing"]    = "Tile处理";

            double totalStage = 0.0;
            std::map<std::string,double>::const_iterator it;
            for (it = res.timer.stages.begin(); it != res.timer.stages.end(); ++it)
                totalStage += it->second;

            // 按阶段输出耗时和占比
            for (it = res.timer.stages.begin(); it != res.timer.stages.end(); ++it) {
                std::string name = it->first;
                double duration = it->second;
                double pct = (totalStage > 0) ? (duration / totalStage * 100.0) : 0.0;
                std::map<std::string,std::string>::const_iterator itName =
                    stageNames.find(name);
                std::string cn = (itName != stageNames.end() ? itName->second : name);
                std::cout << "    - " << cn << ": "
                          << formatTime(duration)
                          << " (" << std::fixed << std::setprecision(1)
                          << pct << "%)" << std::endl;
            }

            if (res.totalTilesProcessed > 0) {
                std::cout << "  处理了 " << res.totalTilesProcessed << " 个tile:" << std::endl;
                std::cout << "    - 平均每个tile: "
                          << formatTime(res.avgTileTime) << std::endl;
                std::cout << "    - 最快tile: "
                          << formatTime(res.minTileTime) << std::endl;
                std::cout << "    - 最慢tile: "
                          << formatTime(res.maxTileTime) << std::endl;
            }
        } else {
            failedCount++;
            std::cout << "✗ 处理失败: " << res.errorMsg << std::endl;
            std::cout << "  耗时: " << formatTimeDetailed(totalT) << std::endl;
        }
    }

    // 整体统计
    TimeTracker::Clock::time_point totalEnd = TimeTracker::Clock::now();
    double totalTime = std::chrono::duration<double>(totalEnd - totalStart).count();

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "处理完成!" << std::endl;
    std::cout << "成功: " << processedCount
              << ", 失败: " << failedCount << std::endl;
    std::cout << "总用时: " << formatTimeDetailed(totalTime) << std::endl;
    if (processedCount > 0) {
        double avg = totalTime / processedCount;
        std::cout << "平均每个视频: " << formatTimeDetailed(avg) << std::endl;
    }

    // 统计最耗时的 3 个成功视频
    if (!results.empty()) {
        std::vector<VideoProcessor::ProcessResult> succ;
        for (size_t i = 0; i < results.size(); ++i) {
            if (results[i].success) succ.push_back(results[i]);
        }
        if (!succ.empty()) {
            std::sort(succ.begin(), succ.end(),
                      [](const VideoProcessor::ProcessResult& a,
                         const VideoProcessor::ProcessResult& b) {
                          return a.timer.totalTime() > b.timer.totalTime();
                      });
            std::cout << "\n最耗时的3个视频:" << std::endl;
            for (size_t i = 0; i < succ.size() && i < 3; ++i) {
                fs::path p(succ[i].videoPath);
                std::cout << "  " << (i+1) << ". "
                          << p.filename().string() << ": "
                          << formatTimeDetailed(succ[i].timer.totalTime())
                          << std::endl;
            }
        }
    }

    std::cout << "输出目录: " << outRoot << std::endl;
    return 0;
}
