#include "Steam.h"

CSteam::CSteam()
{
}

CSteam &CSteam::GetInstance(){
    static CSteam Instance;
    return Instance;
}

#define CHECK_CUDA(call)                                                               \
    do {                                                                               \
    cudaError_t err = call;                                                        \
    if (err != cudaSuccess) {                                                      \
    std::cerr << "CUDA error: " << cudaGetErrorString(err)                     \
    << " at " << __FILE__ << ":" << __LINE__ << std::endl;           \
    std::exit(EXIT_FAILURE);                                                   \
    }                                                                              \
    } while (0)

// 将字符串转为小写（用于比较扩展名等）
std::string CSteam::toLower(const std::string& s) {
    std::string r = s;
    for (size_t i = 0; i < r.size(); ++i) {
        r[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(r[i])));
    }
    return r;
}

// 将索引左侧补零，生成固定宽度的字符串（例如 3 -> "003"）
std::string CSteam::padIndex(size_t idx, int width) {
    std::ostringstream oss;
    oss << std::setw(width) << std::setfill('0') << idx;
    return oss.str();
}


int CSteam::Detect(const QString &VideoPath, CYolov8Cls *m_Yolov8Cls, const QJsonObject &input, QJsonObject &output){

//    cv::VideoCapture tempVideoCapture;
//    if(!tempVideoCapture.open(VideoPath.toStdString()/*, cv::CAP_FFMPEG*/)){
//        qDebug() << "视频文件打开失败";
//        return -1;
//    }

    ProcessResult result;
    TimeTracker timer;
    Config config;
    timer.startStage("read_frames");
    std::vector<cv::Mat> frames = readAllFrames(VideoPath.toStdString());
    timer.endStage();
    result.frameCount = static_cast<int>(frames.size());
    if (frames.empty()){
        qDebug() << "视频文件打开失败";
        return -1;
    }
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

    /* ---------- 6. 保存可视化首帧 vis.jpg ---------- */
    QDateTime currentDateTime = QDateTime::currentDateTime();
    QString dateString = currentDateTime.toString("yyyy-MM-dd");
    QString hourString = currentDateTime.toString("HH-mm-ss");  // 注意这里是 mm 和 ss，不是 MM
    // 创建日期文件夹路径
    QString dateFolderPath = "./VisualVideo_Steam/pretreatment/"  + dateString;
    QString fullFolderPath = dateFolderPath + "/" + hourString;
    // 创建文件夹（如果不存在）
    QDir dir;
    if (!dir.exists(fullFolderPath)) {
        if (!dir.mkpath(fullFolderPath)) {
            qDebug() << "Failed to create directory:" << fullFolderPath;
            return false;
        }
    }
    QString tempMatPath = fullFolderPath + "/" + QDateTime::currentDateTime().toString("yyyy_MM_dd_hh_mm_ss_zzz") + ".jpg";

    timer.startStage("visualization");
    if (config.SAVE_VIS && !frames.empty()) {
        cv::Mat vis = drawTilesVis(
                    frames[0],
                selectedTiles,
                (config.COVERAGE_MODE == "topk" ? &tileScores : NULL)
                );

        cv::imwrite(tempMatPath.toStdString(), vis);
    }
    timer.endStage();

    /* ---------- 7. 保存 tiles_meta.json ---------- */
    timer.startStage("metadata_save");
    saveTilesMeta(
                (fullFolderPath+"/" + "tiles_meta.json").toStdString(),
                VideoPath.toStdString(), height, width,
                config, selectedTiles, tileScores
                );
    timer.endStage();

    // 若只需要 JSON/vis，不做后续切块导出
    if (toLower(config.EXPORT_MODE) == "json") {
        timer.endStage();
        result.success = true;
        result.timer = timer;
//        return result;
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
        cv::VideoCapture cap(VideoPath.toStdString());
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
        fs::path groupDir = fullFolderPath.toStdString();
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
            bool hasTightBoxForThisTile = false; // 标记是否找到有效紧致框

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
                            hasTightBoxForThisTile = true; // 标记找到有效紧致框
                        }
                    }
                }

                // 保存紧致裁剪框信息（用于后续绘制）
                if (hasTightBoxForThisTile && ti < allTightBoxes.size()) {
                    allTightBoxes[ti] = actualCropRegion;
                    allHasTightBox[ti] = true;
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

    qDebug() << "预处理完成";

    std::string enginePath = "./model/steamX3D.trt";
    int B = 1;
    int T = 48;
    int H = 256;
    int W = 256;
    float threshold =  0.5f;

    const std::string inputName = "video";
    const int numClasses = 2;

    MyLogger logger;
//    std::vector<std::string> videoFiles = getVideoFiles(fullFolderPath.toStdString());
    std::vector<std::string> videoFiles = getVideoFiles(fullFolderPath.toStdString());
    if (videoFiles.empty()) {
        std::cerr << "❌ 未找到可处理的视频文件: " << fullFolderPath.toStdString() << std::endl;
        return 1;
    }

    std::cout << "📁 找到 " << videoFiles.size() << " 个视频文件" << std::endl;
    std::cout << "⚙️  参数: B=" << B << ", T=" << T << ", H=" << H << ", W=" << W
              << ", threshold=" << threshold << std::endl;


    int steamCount = 0;
    int noSteamCount = 0;

    // 用于存储每个tile的推理结果（用于后续绘制检测框）
    std::vector<cv::Rect> allTightBoxes;  // 紧致裁剪框
    std::vector<bool> allHasTightBox;     // 是否有紧致框的标记
    std::vector<float> allProbSteam;      // 每个tile的steam概率
    allTightBoxes.resize(selectedTiles.size());
    allHasTightBox.resize(selectedTiles.size(), false);
    allProbSteam.resize(selectedTiles.size(), 0.0f);

    try {
        // 1. 加载引擎（只需一次）
        std::vector<char> engineData = loadEngineFile(enginePath);

        auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(logger));
        if (!runtime) {
            throw std::runtime_error("Failed to create IRuntime");
        }

        auto engine = std::unique_ptr<ICudaEngine>(
            runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (!engine) {
            throw std::runtime_error("Failed to deserialize engine");
        }

        auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
        if (!context) {
            throw std::runtime_error("Failed to create execution context");
        }

        std::cout << "✅ Engine加载成功. NbBindings = " << engine->getNbBindings() << std::endl;

        // 2. 获取 input/output binding index
        int inputIndex = engine->getBindingIndex(inputName.c_str());
        if (inputIndex < 0) {
            throw std::runtime_error("Cannot find input tensor with name: " + inputName);
        }

        int outputIndex = -1;
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            if (!engine->bindingIsInput(i)) {
                outputIndex = i;
                break;
            }
        }
        if (outputIndex < 0) {
            throw std::runtime_error("Cannot find output binding");
        }

        // 3. 批量处理视频
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

            InferenceResult result = processSingleVideo(
                context.get(), inputIndex, outputIndex, videoFile, B, T, H, W, threshold);

            if (result.errorMsg.empty()) {
                successCount++;
                totalInferenceTime += result.inferenceTime;
                totalProcessedFrames += result.processedFrames;

                // 存储当前tile的推理结果
                if (i < allProbSteam.size()) {
                    allProbSteam[i] = result.probSteam;
                }

                std::cout << "  结果: no_steam=" << result.probNoSteam
                          << ", steam=" << result.probSteam
                          << " | 判断: " << (result.isSteam ? "STEAM" : "NO_STEAM")
                          << " | 耗时: " << formatTime(result.inferenceTime) << std::endl;
                if (result.probSteam >= 0.7){
                    qDebug() << "有水蒸气！！！";
                    QDate currentDate = QDate::currentDate();
                    QString dateString = currentDate.toString("yyyy-MM-dd");
                    // 创建日期文件夹路径
                    QString dateFolderPath = "./VisualVideo_Steam/TRUE/"  + dateString;
                    // 创建文件夹（如果不存在）
                    QDir dir;
                    if (!dir.exists(dateFolderPath)) {
                        if (!dir.mkpath(dateFolderPath)) {
                            qDebug() << "Failed to create directory:" << dateFolderPath;
                            return false;
                        }
                    }

                    // 绘制检测框
                    QString videoWithBoxesPath = dateFolderPath + "/" + QDateTime::currentDateTime().toString("yyyy_MM_dd_hh_mm_ss_zzz") + "_detected.mp4";
                    qDebug() << "开始绘制检测框...";
                    bool drawSuccess = drawSteamBoxesOnVideo(
                        VideoPath.toStdString(),
                        videoWithBoxesPath.toStdString(),
                        selectedTiles,
                        allTightBoxes,
                        allHasTightBox,
                        allProbSteam,
                        threshold,
                        0.1f,   // mergeIoUThreshold
                        0.05f   // expandMergedBoxRatio
                    );
                    if (drawSuccess) {
                        qDebug() << "检测框绘制成功: " << videoWithBoxesPath;
                    } else {
                        qDebug() << "检测框绘制失败";
                    }

                    // 复制原始视频
                    QFileInfo videoInfo(VideoPath);
                    QString destPath = dateFolderPath + "/" + QDateTime::currentDateTime().toString("yyyy_MM_dd_hh_mm_ss_zzz") + ".mp4";
                    if (!QFile::copy(VideoPath, destPath)) {
                        qDebug() << "视频复制失败: " ;
                        return -5;
                    }
                    output.insert("result", true);
                    return 0;
                }
            } else {
                failCount++;
                std::cout << "  ❌ 错误: " << result.errorMsg << std::endl;
            }

            results.push_back(result);
            std::cout << "-" << std::string(60, '-') << "-" << std::endl;
        }

        // 4. 输出统计信息
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

        std::cout << "✅ 批量推理完成!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
        return 1;
    }


    if (steamCount>=2){
        qDebug() << "有水蒸气！！！";
        QDate currentDate = QDate::currentDate();
        QString dateString = currentDate.toString("yyyy-MM-dd");
        // 创建日期文件夹路径
        QString dateFolderPath = "./VisualVideo_Steam/TRUE/"  + dateString;
        // 创建文件夹（如果不存在）
        QDir dir;
        if (!dir.exists(dateFolderPath)) {
            if (!dir.mkpath(dateFolderPath)) {
                qDebug() << "Failed to create directory:" << dateFolderPath;
                return false;
            }
        }

        // 绘制检测框
        QString videoWithBoxesPath = dateFolderPath + "/" + QDateTime::currentDateTime().toString("yyyy_MM_dd_hh_mm_ss_zzz") + "_detected.mp4";
        qDebug() << "开始绘制检测框...";
        bool drawSuccess = drawSteamBoxesOnVideo(
            VideoPath.toStdString(),
            videoWithBoxesPath.toStdString(),
            selectedTiles,
            allTightBoxes,
            allHasTightBox,
            allProbSteam,
            threshold,
            0.1f,   // mergeIoUThreshold
            0.05f   // expandMergedBoxRatio
        );
        if (drawSuccess) {
            qDebug() << "检测框绘制成功: " << videoWithBoxesPath;
        } else {
            qDebug() << "检测框绘制失败";
        }

        // 复制原始视频
        QFileInfo videoInfo(VideoPath);
        QString destPath = dateFolderPath + "/" + QDateTime::currentDateTime().toString("yyyy_MM_dd_hh_mm_ss_zzz") + ".mp4";
        if (!QFile::copy(VideoPath, destPath)) {
            qDebug() << "视频复制失败: " ;
            return -5;
        }
        output.insert("result", true);
        return 0;
    }else{
        qDebug() << "无水蒸气！！！";
        QDate currentDate = QDate::currentDate();
        QString dateString = currentDate.toString("yyyy-MM-dd");
        // 创建日期文件夹路径
        QString dateFolderPath = "./VisualVideo_Steam/FALSE/"  + dateString;
        // 创建文件夹（如果不存在）
        QDir dir;
        if (!dir.exists(dateFolderPath)) {
            if (!dir.mkpath(dateFolderPath)) {
                qDebug() << "Failed to create directory:" << dateFolderPath;
                return false;
            }
        }
        QFileInfo videoInfo(VideoPath);
        QString destPath = dateFolderPath + "/" + QDateTime::currentDateTime().toString("yyyy_MM_dd_hh_mm_ss_zzz") + ".mp4";
        if (!QFile::copy(VideoPath, destPath)) {
            qDebug() << "视频复制失败: " ;
            return -5;
        }
        output.insert("result", false);
        return 0;
    }
    return 0;
}

void CSteam::WaterVaporAnalysis(int Return, QJsonObject &FunctionResult, QJsonObject &ResultStatus, QJsonObject &Result, QJsonArray &ResultList){
    switch (Return) {
    case 0:
        qDebug() << "水蒸气检测结果: " << FunctionResult.value("result").toBool();
        ResultStatus.insert("code", 200);
        ResultStatus.insert("desc", "ok");
        Result.insert("status", ResultStatus);
        Result.insert("data_type", "STR");
        if(FunctionResult.value("result").toBool()){
            Result.insert("data", "true");
        }else{
            Result.insert("data", "false");
        }
        Result.insert("confidence", 0);
        ResultList.append(Result);
        break;
    case -1:
        qDebug() << "水蒸气检测，视频文件打开失败";
        ResultStatus.insert("code", 502);
        ResultStatus.insert("desc", "水蒸气检测，视频文件打开失败");
        Result.insert("status", ResultStatus);
        Result.insert("data_type", "STR");
        Result.insert("data", "error");
        Result.insert("confidence", 0);
        ResultList.append(Result);
        break;
    case -2:
        qDebug() << "水蒸气检测，检测失败";
        ResultStatus.insert("code", 502);
        ResultStatus.insert("desc", "水蒸气检测，视频文件打开失败");
        Result.insert("status", ResultStatus);
        Result.insert("data_type", "STR");
        Result.insert("data", "error");
        Result.insert("confidence", 0);
        ResultList.append(Result);
        break;
    default:
        break;
    }
}

/* ---------- 读取整个视频为 RGB 帧数组 ---------- */
std::vector<cv::Mat> CSteam::readAllFrames(const std::string& videoPath) {
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
void CSteam::resizeShortSideFrames(std::vector<cv::Mat>& frames, int shortSide) {
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

/* ---------- ECC 视频稳像（可选） ---------- */
void CSteam::eccStabilize(std::vector<cv::Mat>& frames,
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

/* ---------- 计算全局帧差热度图（归一化到 0~1） ---------- */
cv::Mat CSteam::diffHeatmapFast(const std::vector<cv::Mat>& frames,
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

/* ---------- 蒸汽综合打分图：w_diff * diff + w_white * white + w_smooth * (1-edge) ---------- */
cv::Mat CSteam::steamScoreMapFast(const std::vector<cv::Mat>& frames,
                          const cv::Mat& reuseDiff) {
    Config config;
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

/* ---------- 计算"白度"热度图 ---------- */
cv::Mat CSteam::whitenessMapFast(const std::vector<cv::Mat>& frames,
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
cv::Mat CSteam::edgeDensityMapFast(const std::vector<cv::Mat>& frames,
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

/* ---------- 按比例下采样一组帧 ---------- */
std::vector<cv::Mat> CSteam::downscaleFrames(const std::vector<cv::Mat>& frames,
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

/* 使用 2~98 百分位做归一化，与 Python _norm01 类似 */
// 把输入矩阵线性映射到 [0,1]，剪裁极端值
cv::Mat CSteam::normalize01Percentile(const cv::Mat& input) {
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

/* percentile 计算：mat 为 CV_32F，返回 pct 百分位值 */
// 用于动态阈值计算，比如 85% 分位数
double CSteam::getPercentile(const cv::Mat& mat, float pct) {
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

/* ---------- 根据 scoreMap 选 Top-K tile ---------- */
std::pair<std::vector<cv::Rect>, std::vector<float> > CSteam::selectTopKByMap(const cv::Mat& scoreMap,
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

/* ---------- 生成完整覆盖的 tile 网格 ---------- */
std::vector<cv::Rect> CSteam::makeFullCoverTiles(int height, int width,
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

bool CSteam::validateTileConsistency(const std::vector<cv::Rect>& selectedTiles,
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

/* ---------- 绘制首帧+Tile 可视化图 ---------- */
cv::Mat CSteam::drawTilesVis(const cv::Mat& rgb,
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
void CSteam::saveTilesMeta(const std::string& metaPath,
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

/* ---------- 滑窗分组：生成 group clips ---------- */
std::vector<std::vector<cv::Mat> > CSteam::makeClips(const std::vector<cv::Mat>& frames,
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

/* 自适应 margin 函数：tile 内目标区域越小，边缘扩展比例越大 */
float CSteam::adaptiveMarginFunc(const cv::Rect& bbox, int tileH, int tileW,
                         float base, float lo, float hi) {
    double areaRatio = (double)bbox.area() / std::max(1.0, (double)tileH * tileW);
    double v = base + (1.0 - std::min(areaRatio / 0.25, 1.0)) * 0.10;
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return static_cast<float>(v);
}

/* ---------- 背景净化：根据掩膜将背景替换为模糊/黑/灰/均值 ---------- */
void CSteam::applyBgMode(std::vector<cv::Mat>& frames,
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
void CSteam::saveFramesDir(const std::string& outDir,
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
void CSteam::saveVideo(const std::string& outPath,
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

// 获取目录中的所有视频文件
std::vector<std::string> CSteam::getVideoFiles(const std::string& directoryPath) {
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
            if (fs::is_regular_file(entry.status())) {
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

// 读取 .engine 文件
std::vector<char> CSteam::loadEngineFile(const std::string& enginePath) {
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

// 单个视频推理函数
InferenceResult CSteam::processSingleVideo(
    IExecutionContext* context,
    int inputIndex,
    int outputIndex,
    const std::string& videoPath,
    int B, int T, int H, int W,
    float threshold) {

    InferenceResult result;
    result.videoPath = videoPath;
    result.videoName = fs::path(videoPath).filename().string();

    auto startTime = std::chrono::high_resolution_clock::now();

    try {
        // 设置动态维度
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

        // 计算 buffer 大小
        size_t inputVolume = static_cast<size_t>(B) * 3 * T * H * W;
        size_t outputVolume = static_cast<size_t>(B) * 2;  // 2个类别

        size_t inputSizeBytes = inputVolume * sizeof(float);
        size_t outputSizeBytes = outputVolume * sizeof(float);

        // 从视频文件加载数据
        std::vector<float> hInput(inputVolume);
        if (!loadAndPreprocessVideo(videoPath, T, H, W, hInput)) {
            throw std::runtime_error("视频预处理失败");
        }

        // 分配GPU内存
        void* dInput = nullptr;
        void* dOutput = nullptr;
        CHECK_CUDA(cudaMalloc(&dInput, inputSizeBytes));
        CHECK_CUDA(cudaMalloc(&dOutput, outputSizeBytes));

        // 绑定数组
        void* bindings[2];
        bindings[inputIndex] = dInput;
        bindings[outputIndex] = dOutput;

        // 拷贝到GPU
        CHECK_CUDA(cudaMemcpy(dInput, hInput.data(), inputSizeBytes, cudaMemcpyHostToDevice));

        // 创建流并推理
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));

        bool okRun = context->enqueueV2(bindings, stream, nullptr);
        if (!okRun) {
            throw std::runtime_error("enqueueV2 failed");
        }

        // 同步并获取结果
        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::vector<float> hOutput(outputVolume);
        CHECK_CUDA(cudaMemcpy(hOutput.data(), dOutput, outputSizeBytes, cudaMemcpyDeviceToHost));

        // 解析输出
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

        // 清理资源
        CHECK_CUDA(cudaStreamDestroy(stream));
        CHECK_CUDA(cudaFree(dInput));
        CHECK_CUDA(cudaFree(dOutput));

    } catch (const std::exception& e) {
        result.errorMsg = e.what();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    result.inferenceTime = std::chrono::duration<double>(endTime - startTime).count();

    return result;
}

// 视频预处理函数
bool CSteam::loadAndPreprocessVideo(const std::string& videoPath, int targetFrames,
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

// 时间格式化函数
std::string CSteam::formatTime(double seconds) {
    if (seconds < 1e-3) {
        return std::to_string(seconds * 1e6) + " μs";
    } else if (seconds < 1) {
        return std::to_string(seconds * 1000) + " ms";
    } else {
        return std::to_string(seconds) + " s";
    }
}

/* ====================== 绘制检测框相关函数 ====================== */

// 计算两个矩形的IoU (Intersection over Union)
double CSteam::computeIoU(const cv::Rect& a, const cv::Rect& b) {
    // 计算交集区域
    int interX0 = std::max(a.x, b.x);
    int interY0 = std::max(a.y, b.y);
    int interX1 = std::min(a.x + a.width, b.x + b.width);
    int interY1 = std::min(a.y + a.height, b.y + b.height);

    if (interX1 <= interX0 || interY1 <= interY0) {
        return 0.0;
    }

    int interArea = (interX1 - interX0) * (interY1 - interY0);
    int aArea = a.width * a.height;
    int bArea = b.width * b.height;
    int unionArea = aArea + bArea - interArea;

    return (unionArea > 0) ? (double)interArea / (double)unionArea : 0.0;
}

/**
 * 在原始视频上绘制steam检测框
 * @param inputVideoPath 输入视频路径
 * @param outputVideoPath 输出视频路径
 * @param tiles 原始tile区域列表
 * @param tightBoxes 紧致裁剪框列表（与tiles对应）
 * @param hasTightBox 标记每个tile是否有紧致框
 * @param probSteamPerTile 每个tile的steam概率
 * @param threshold steam判定阈值
 * @param mergeIoUThreshold 合并框的IoU阈值
 * @param expandMergedBoxRatio 合并后框的扩展比例
 */
bool CSteam::drawSteamBoxesOnVideo(
    const std::string& inputVideoPath,
    const std::string& outputVideoPath,
    const std::vector<cv::Rect>& tiles,
    const std::vector<cv::Rect>& tightBoxes,
    const std::vector<bool>& hasTightBox,
    const std::vector<float>& probSteamPerTile,
    float threshold,
    float mergeIoUThreshold,
    float expandMergedBoxRatio)
{
    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) {
        std::cerr << "❌ 无法打开视频: " << inputVideoPath << std::endl;
        return false;
    }

    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps < 1.0) fps = 25.0;

    // 创建视频写入器
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(outputVideoPath, fourcc, fps,
                          cv::Size(frameWidth, frameHeight), true);
    if (!writer.isOpened()) {
        std::cerr << "❌ 无法创建视频写入器: " << outputVideoPath << std::endl;
        cap.release();
        return false;
    }

    // 只筛选steam框（概率 >= threshold）
    std::vector<cv::Rect> steamBoxes;
    std::vector<float> steamProbs;
    std::vector<cv::Rect> steamOriginalTiles;

    for (size_t i = 0; i < tiles.size(); ++i) {
        if (i >= probSteamPerTile.size()) continue;

        float probSteam = probSteamPerTile[i];

        // 只处理steam框（概率 >= threshold）
        if (probSteam >= threshold) {
            cv::Rect box;
            // 优先使用紧致框，否则使用原始tile
            if (i < hasTightBox.size() && hasTightBox[i] && i < tightBoxes.size()) {
                box = tightBoxes[i];
            } else {
                box = tiles[i];
            }

            steamBoxes.push_back(box);
            steamProbs.push_back(probSteam);
            steamOriginalTiles.push_back(tiles[i]);
        }
    }

    // 合并相邻的steam框（使用并查集算法）
    std::vector<cv::Rect> boxesToDraw;
    std::vector<float> probsToDraw;

    if (steamBoxes.size() > 1) {
        size_t n = steamBoxes.size();
        std::vector<int> parent(n);
        for (size_t i = 0; i < n; ++i) parent[i] = static_cast<int>(i);

        // 并查集的find函数
        std::function<int(int)> find = [&](int x) -> int {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        };

        // 并查集的union函数
        auto unionSets = [&](int x, int y) {
            int px = find(x);
            int py = find(y);
            if (px != py) {
                parent[px] = py;
            }
        };

        // 遍历所有框对，判断是否需要合并
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                bool shouldMerge = false;

                // 策略1: 检查紧致框的IoU和距离
                double iou = computeIoU(steamBoxes[i], steamBoxes[j]);

                double y1Center = steamBoxes[i].y + steamBoxes[i].height / 2.0;
                double x1Center = steamBoxes[i].x + steamBoxes[i].width / 2.0;
                double y2Center = steamBoxes[j].y + steamBoxes[j].height / 2.0;
                double x2Center = steamBoxes[j].x + steamBoxes[j].width / 2.0;

                double avgH = (steamBoxes[i].height + steamBoxes[j].height) / 2.0;
                double avgW = (steamBoxes[i].width + steamBoxes[j].width) / 2.0;
                double avgSize = (avgH + avgW) / 2.0;

                double centerDist = std::sqrt(
                    (y1Center - y2Center) * (y1Center - y2Center) +
                    (x1Center - x2Center) * (x1Center - x2Center)
                );

                // 如果IoU大于阈值或中心距离很近，合并
                if (iou > mergeIoUThreshold || centerDist < avgSize * 1.5) {
                    shouldMerge = true;
                }

                // 策略2: 检查原始tile是否相邻或重叠
                const cv::Rect& origI = steamOriginalTiles[i];
                const cv::Rect& origJ = steamOriginalTiles[j];

                int tolerance = 50; // 容忍距离（像素）

                // 检查水平或垂直相邻
                bool hAdjacent = (std::abs((origI.x + origI.width) - origJ.x) < tolerance ||
                                 std::abs((origJ.x + origJ.width) - origI.x) < tolerance);
                bool vAdjacent = (std::abs((origI.y + origI.height) - origJ.y) < tolerance ||
                                 std::abs((origJ.y + origJ.height) - origI.y) < tolerance);

                // 检查是否有重叠
                bool hasOverlap = !((origI.x + origI.width) <= origJ.x ||
                                   (origJ.x + origJ.width) <= origI.x ||
                                   (origI.y + origI.height) <= origJ.y ||
                                   (origJ.y + origJ.height) <= origI.y);

                if (hAdjacent || vAdjacent || hasOverlap) {
                    shouldMerge = true;
                }

                if (shouldMerge) {
                    unionSets(static_cast<int>(i), static_cast<int>(j));
                }
            }
        }

        // 将同一组的框合并为一个大框
        std::map<int, std::vector<size_t>> groups;
        for (size_t i = 0; i < n; ++i) {
            groups[find(static_cast<int>(i))].push_back(i);
        }

        for (const auto& pair : groups) {
            const auto& indices = pair.second;

            // 计算最小外接矩形
            int xMin = steamBoxes[indices[0]].x;
            int yMin = steamBoxes[indices[0]].y;
            int xMax = steamBoxes[indices[0]].x + steamBoxes[indices[0]].width;
            int yMax = steamBoxes[indices[0]].y + steamBoxes[indices[0]].height;

            float maxProb = steamProbs[indices[0]];

            for (size_t i = 1; i < indices.size(); ++i) {
                size_t idx = indices[i];
                xMin = std::min(xMin, steamBoxes[idx].x);
                yMin = std::min(yMin, steamBoxes[idx].y);
                xMax = std::max(xMax, steamBoxes[idx].x + steamBoxes[idx].width);
                yMax = std::max(yMax, steamBoxes[idx].y + steamBoxes[idx].height);
                maxProb = std::max(maxProb, steamProbs[idx]);
            }

            // 扩展合并后的框
            int w = xMax - xMin;
            int h = yMax - yMin;
            int dx = static_cast<int>(w * expandMergedBoxRatio);
            int dy = static_cast<int>(h * expandMergedBoxRatio);

            xMin = std::max(0, xMin - dx);
            yMin = std::max(0, yMin - dy);
            xMax = std::min(frameWidth, xMax + dx);
            yMax = std::min(frameHeight, yMax + dy);

            boxesToDraw.push_back(cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
            probsToDraw.push_back(maxProb);
        }
    } else if (steamBoxes.size() == 1) {
        boxesToDraw = steamBoxes;
        probsToDraw = steamProbs;
    }

    std::cout << "  绘制 " << boxesToDraw.size()
              << " 个steam区域（使用紧致裁剪区域，考虑原始tile相邻性）" << std::endl;

    // 逐帧处理并绘制
    cv::Mat frame;
    int frameIdx = 0;
    while (cap.read(frame)) {
        // 绘制所有steam框
        for (size_t i = 0; i < boxesToDraw.size(); ++i) {
            const cv::Rect& box = boxesToDraw[i];
            float probSteam = probsToDraw[i];

            // 只绘制红色steam框
            cv::Scalar color = cv::Scalar(0, 0, 255); // BGR: 红色表示steam
            std::string labelText = "Steam " + std::to_string(probSteam).substr(0, 4);

            // 绘制矩形框
            cv::rectangle(frame, box, color, 4);

            // 绘制标签（带背景）
            double fontScale = 1.0;
            int textThickness = 2;
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX,
                                               fontScale, textThickness, &baseline);

            // 绘制文本背景
            cv::rectangle(frame,
                         cv::Point(box.x, box.y - textSize.height - baseline - 10),
                         cv::Point(box.x + textSize.width + 10, box.y),
                         color, -1);

            // 绘制文本
            cv::putText(frame, labelText,
                       cv::Point(box.x + 5, box.y - baseline - 5),
                       cv::FONT_HERSHEY_SIMPLEX, fontScale,
                       cv::Scalar(255, 255, 255), textThickness);
        }

        writer.write(frame);
        frameIdx++;
    }

    cap.release();
    writer.release();

    std::cout << "✅ 可视化视频已保存: " << outputVideoPath << std::endl;
    return true;
}
