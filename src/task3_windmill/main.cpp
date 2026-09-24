// 任务3：真实能量机关视频的识别与稳定跟踪
//
// 处理 task_3.mp4（小能量机关，同时最多 1 个目标）与 task_4.mp4（大能量机关，最多 2 个）。
// 同一程序通过输入路径参数处理两个视频，输出完整叠加视频与二值化过程视频。
//
// 检测思路（阈值与几何门控均由素材实测标定）：
//   1. 掩膜：能量机关发光为深橙红，H∈[0,17]、S≥55、V≥80；基地灯偏黄（H 16~31）、
//      横幅偏绿（H≈95~115），基本落在掩膜外，残余干扰靠形状特征排除。
//   2. 目标簇：双环圆靶在掩膜中常断成 2~4 段弧，将半径≥8 的核心块按 55px 链式聚类；
//      形状筛选：外接半径∈[15,95]、填充率≤0.55（灯带字母≈0.9）、长宽比≤4（条带簇）。
//   3. 点亮判别：点亮靶标为"双环+中心标"结构，其 0.5R 内侧圆有显著掩膜能量
//      （实测点亮目标 inner p50=0.24）；未点亮靶标为单环，内侧近似为空（p50=0.07）。
//      inner_density = 0.5R 圆内掩膜占比，≥0.10 判为点亮。
//   4. 跟踪与稳定锁定：维持"扇叶位置池"，簇按近邻关联到池条目（含速度 EMA 以适应
//      相机平移）。锁定 = 点亮且稳定的条目，ID 只在锁定/重选时分配，绝不使用每帧轮廓
//      顺序；出现第二个点亮目标时不切换；当前目标熄灭（inner 连续低于阈值）或持续丢失
//      超过容忍帧数后释放，并按"点亮且稳定、面积最大"重选，分配新 ID，全程事件留痕。
//   5. R 标中心：R 字母为脉冲 LED（实测可见率仅 28%~37%），采用字形检测（实心、小、
//      圆形度低）+ EMA 跟踪；字形不可见时按"场景漂移速度"外推（R 相对风车静止，随相
//      机一起运动），状态标记 detected / estimated。R 估计位置附近 45px 内的块不参与
//      聚类，防止 R 字母被流向圆点条带簇吸收。
//
// 用法（在 build/ 目录下运行）：
//   ./task3_windmill [输入视频路径] [输出目录] [--show] [--debug]
// 默认输入 ../resources/task_3.mp4，输出 ../result/task3_windmill/<视频文件名>/

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// ---------------- 可调参数（均由 task_3/task_4 实测标定） ----------------
struct Config {
    // HSV 掩膜
    int hue_max = 17;
    int s_min = 55, v_min = 80;
    // 块筛选
    float blob_min_area = 40.f, blob_max_radius = 80.f;
    float border_margin = 35.f;
    // 目标簇
    float core_min_radius = 8.f;
    float cluster_merge_dist = 48.f;
    float cluster_min_enc = 15.f, cluster_max_enc = 95.f;
    float cluster_fill_max = 0.55f;
    float cluster_aspect_max = 4.0f;
    float cluster_area_max = 3200.f;
    // 点亮判别（滞回：进入/退出阈值不同，避免闪烁导致误报丢失）
    float inner_radius_ratio = 0.5f;
    float lit_inner_min = 0.13f;      // 进入"点亮"的阈值
    float lit_inner_exit = 0.09f;     // 退出"点亮"的阈值（低于该值才判未点亮）
    float tid_min_area = 200.f;       // 展示用 ID 的面积门槛（低于锁定门槛）
    int   tid_min_age = 5;
    // 锁定与丢失
    float pool_match_gate = 55.f;     // 簇-条目关联门限（px）
    float pool_gate_per_absent = 4.f;
    float pool_gate_max = 110.f;
    float pool_new_min_area = 150.f;  // 新条目所需最小簇面积
    int   pool_absent_drop = 90;      // 连续缺席多久删除条目
    int   stable_min_hits = 6;        // 最近 10 帧在场≥6 次视为稳定结构
    int   stable_window = 10;
    int   pool_min_age = 10;          // 锁定要求条目至少存在 10 帧
    float lock_area_min = 250.f;
    int   lit_min_hits = 4;           // 最近 10 帧点亮≥4 次才算持续点亮（作 ID/锁定门槛）
    int   lost_tolerance = 45;        // 持续丢失容忍帧数（1.5s）
    int   dark_lost_frames = 5;       // 熄灭多少帧后才把状态从 detected 降为 lost（抗抖动）
    int   dark_switch_frames = 15;    // 熄灭连续帧数达到即允许切换（0.5s，抑制闪烁误切）
    // R 标：字形签名（实测：r≈12、area≈260、circ 0.33~0.62、fill 0.42~0.75、孤立）
    float r_sig_r_min = 11.f, r_sig_r_max = 14.f;
    float r_sig_area_min = 215.f, r_sig_area_max = 325.f;
    float r_sig_circ_min = 0.33f, r_sig_circ_max = 0.62f;
    float r_sig_fill_min = 0.42f, r_sig_fill_max = 0.75f;
    // R 链跟踪
    float r_chain_gate = 40.f, r_chain_gate_dt = 8.f, r_chain_gate_max = 200.f;
    int   r_chain_drop = 240;         // 链多久无命中即淘汰
    int   r_hold_frames = 90;         // 无字形时保持外推多久（超过标 lost）
    float r_max_target_dist = 700.f;  // R 到点亮目标的距离上限（超出视为假链）
    float scene_vel_max = 6.f;
};

// ---------------- 数据结构 ----------------
struct Blob {
    Point2f center;
    float radius = 0, area = 0, circ = 0;
    int holes = 0;
};

struct Cluster {
    Point2f center;
    float enc_radius = 0, total_area = 0, fill = 0, aspect = 1;
    float small_frac = 0;      // 小圆点成员占比（条带簇 ≈1，扇叶环簇较低）
    float inner_density = 0;   // 0.5R 内掩膜占比：点亮双环结构显著高于未点亮单环
    bool usable = false;
    vector<int> blob_ids;
};

struct PoolEntry {
    int id;
    Point2f pos, vel;
    float area = 0;
    float inner_ema = 0;
    deque<int> recent;          // 最近 stable_window 帧是否在场
    deque<int> lit_recent;      // 最近 stable_window 帧是否点亮（区分持续点亮与瞬态高亮）
    int absent = 0;
    int age = 0;                // 入池以来的帧数
    int since_lit = 1000;       // 距上次判为点亮的帧数（初始视为久未点亮）
    int tid = 0;                // 作为"有效目标"对外展示的 ID（首次点亮且稳定时分配）
    bool lit = false;
};

struct SceneState { Point2f vel{0, 0}; };   // 场景整体漂移（相机运动估计）

struct GlyphCand {          // R 标初始化用的字形候选
    Point2f pos;
    deque<int> recent;
};

struct RChain {             // R 字形链（脉冲 LED 的断续观测累积）
    Point2f pos, vel;
    int last = 0;
    deque<int> hits;
};

// ---------------- 基础处理 ----------------
static Mat makeMask(const Mat& frame, const Config& cfg) {
    Mat hsv, mask;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    inRange(hsv, Scalar(0, cfg.s_min, cfg.v_min),
            Scalar(cfg.hue_max, 255, 255), mask);
    morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
    return mask;
}

static vector<Blob> extractBlobs(const Mat& mask, const Config& cfg) {
    vector<Blob> blobs;
    vector<vector<Point>> contours;
    vector<Vec4i> hier;
    findContours(mask, contours, hier, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    const int W = mask.cols, H = mask.rows;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (hier[i][3] >= 0) continue;
        double a = contourArea(contours[i]);
        if (a < cfg.blob_min_area) continue;
        Point2f c; float r;
        minEnclosingCircle(contours[i], c, r);
        if (r > cfg.blob_max_radius) continue;
        if (c.x < cfg.border_margin || c.y < cfg.border_margin ||
            c.x > W - cfg.border_margin || c.y > H - cfg.border_margin) continue;
        double peri = arcLength(contours[i], true);
        Blob b;
        b.center = c; b.radius = r; b.area = (float)a;
        b.circ = (float)(4 * CV_PI * a / (peri * peri + 1e-6));
        for (int ch = hier[i][2]; ch >= 0; ch = hier[ch][0])
            if (contourArea(contours[ch]) > 8) ++b.holes;
        blobs.push_back(b);
    }
    return blobs;
}

// 聚类 + 形状筛选 + 点亮特征。R 字母由 blob 签名独立检测（见主循环），不依赖聚类。
static vector<Cluster> clusterTargets(const vector<Blob>& blobs, const Mat& mask,
                                      const Config& cfg) {
    vector<int> core_ids;
    for (int i = 0; i < (int)blobs.size(); ++i) {
        if (blobs[i].radius < cfg.core_min_radius) continue;
        core_ids.push_back(i);
    }
    vector<bool> used(blobs.size(), false);
    vector<Cluster> clusters;
    for (int i : core_ids) {
        if (used[i]) continue;
        Cluster cl;
        cl.blob_ids.push_back(i);
        used[i] = true;
        for (size_t k = 0; k < cl.blob_ids.size(); ++k) {
            int ref = cl.blob_ids[k];
            for (int j : core_ids) {
                if (used[j]) continue;
                if (norm(blobs[j].center - blobs[ref].center) < cfg.cluster_merge_dist) {
                    used[j] = true;
                    cl.blob_ids.push_back(j);
                }
            }
        }
        double sw = 0, sx = 0, sy = 0;
        float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
        for (int id : cl.blob_ids) {
            double w = blobs[id].area;
            sw += w; sx += w * blobs[id].center.x; sy += w * blobs[id].center.y;
            x0 = min(x0, blobs[id].center.x - blobs[id].radius);
            y0 = min(y0, blobs[id].center.y - blobs[id].radius);
            x1 = max(x1, blobs[id].center.x + blobs[id].radius);
            y1 = max(y1, blobs[id].center.y + blobs[id].radius);
        }
        if (sw <= 0) continue;
        cl.center = Point2f((float)(sx / sw), (float)(sy / sw));
        for (int id : cl.blob_ids) {
            cl.enc_radius = max(cl.enc_radius,
                                (float)norm(blobs[id].center - cl.center) + blobs[id].radius);
            cl.total_area += blobs[id].area;
        }
        cl.fill = cl.total_area / (float)(CV_PI * cl.enc_radius * cl.enc_radius + 1e-6f);
        cl.aspect = max(x1 - x0, y1 - y0) / (min(x1 - x0, y1 - y0) + 1e-6f);
        // 成员构成：小圆点占比高 = 流向条带；含大块 = 扇叶/环结构
        int n_small = 0;
        for (int id : cl.blob_ids)
            if (blobs[id].radius < 10.f) ++n_small;
        cl.small_frac = cl.blob_ids.empty() ? 0.f : (float)n_small / cl.blob_ids.size();
        // usable：可作为"有效目标"的簇（尺寸/形状/面积合适）
        cl.usable = cl.enc_radius >= cfg.cluster_min_enc &&
                    cl.enc_radius <= cfg.cluster_max_enc &&
                    cl.fill <= cfg.cluster_fill_max &&
                    cl.aspect <= cfg.cluster_aspect_max &&
                    cl.total_area <= cfg.cluster_area_max;
        // 点亮特征：0.5R 内侧圆的掩膜占比（仅对合理尺寸的簇计算）
        if (cl.enc_radius <= 200.f) {
            float ir = cfg.inner_radius_ratio * cl.enc_radius;
            int x0i = max(0, (int)(cl.center.x - ir)), x1i = min(mask.cols, (int)(cl.center.x + ir));
            int y0i = max(0, (int)(cl.center.y - ir)), y1i = min(mask.rows, (int)(cl.center.y + ir));
            if (x1i > x0i && y1i > y0i) {
                Mat roi = mask(Rect(x0i, y0i, x1i - x0i, y1i - y0i));
                float inside = 0, total = 0;
                for (int yy = 0; yy < roi.rows; ++yy)
                    for (int xx = 0; xx < roi.cols; ++xx) {
                        float dx = x0i + xx - cl.center.x, dy = y0i + yy - cl.center.y;
                        if (dx * dx + dy * dy > ir * ir) continue;
                        ++total;
                        if (roi.at<uchar>(yy, xx)) ++inside;
                    }
                cl.inner_density = total > 0 ? inside / total : 0;
            }
        }
        // 全量保留（条带簇外接半径可能 >95，仍需用于 R 共线性估计）
        clusters.push_back(cl);
    }
    return clusters;
}

static string fmtf(float v, int prec = 1) {
    ostringstream os; os << fixed << setprecision(prec) << v; return os.str();
}

static void drawLabel(Mat& img, const string& text, Point org, Scalar color,
                      double scale = 0.55, int thick = 1) {
    int base = 0;
    Size sz = getTextSize(text, FONT_HERSHEY_SIMPLEX, scale, thick, &base);
    rectangle(img, Point(org.x - 2, org.y - sz.height - 4),
              Point(org.x + sz.width + 2, org.y + 4), Scalar(0, 0, 0), FILLED);
    putText(img, text, org, FONT_HERSHEY_SIMPLEX, scale, color, thick, LINE_AA);
}

// ---------------- 主流程 ----------------
int main(int argc, char** argv) {
    Config cfg;
    string video_path = "../resources/task_3.mp4";
    string out_dir;
    bool show = false, debug = false;
    vector<string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--show") show = true;
        else if (args[i] == "--debug") debug = true;
        else if (video_path == "../resources/task_3.mp4" && out_dir.empty() && args[i].rfind("--", 0) != 0) video_path = args[i];
        else if (args[i].rfind("--", 0) != 0) out_dir = args[i];
    }
    string stem = fs::path(video_path).stem().string();
    if (out_dir.empty()) out_dir = "../result/task3_windmill/" + stem;
    fs::create_directories(out_dir);

    VideoCapture cap(video_path);
    if (!cap.isOpened()) { cerr << "Cannot open video: " << video_path << endl; return 1; }
    const int W = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    const int H = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 1) fps = 30;

    VideoWriter w_overlay((fs::path(out_dir) / "recognition_overlay.mp4").string(),
                          VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(W, H));
    VideoWriter w_binary((fs::path(out_dir) / "binary_process.mp4").string(),
                         VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(W, H));
    if (!w_overlay.isOpened()) { cerr << "Cannot write overlay video!" << endl; return 1; }
    ofstream csv((fs::path(out_dir) / "track_log.csv").string());
    csv << "frame,r_status,r_x,r_y,t_status,t_id,t_x,t_y,theta_deg,n_clusters,n_lit\n";
    vector<string> events;

    // ---- 位置池 ----
    vector<PoolEntry> pool;
    int pool_next_id = 1;
    SceneState scene;

    // ---- 锁定状态 ----
    int next_id = 1;
    int locked_entry = -1;      // pool id
    int t_id = 0;
    int t_lost = 0;
    string t_status = "searching";
    Point2f t_show(-1, -1);

    // ---- R 标状态 ----
    Point2f r_est(-1, -1), r_vel(0, 0);
    bool r_has = false;
    int r_lost = 0;
    string r_status = "searching";
    vector<RChain> r_chains;
    struct DroppedTid { Point2f pos; int tid; int frame; };
    vector<DroppedTid> dropped_tids;

    int frame_idx = 0;
    int n_detected = 0, n_lost = 0, n_lit_frames = 0, n_dual_frames = 0;
    int n_lock_selections = 0;
    int r_detected_cnt = 0, r_estimated_cnt = 0;
    Mat frame, mask;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        ++frame_idx;
        mask = makeMask(frame, cfg);
        vector<Blob> blobs = extractBlobs(mask, cfg);
        vector<Cluster> clusters = clusterTargets(blobs, mask, cfg);

        // ================= 1. 位置池更新 =================
        // 贪心近邻关联：簇 ↔ 条目
        struct Cand { float d; int ci, ei; };
        vector<Cand> cands;
        for (int ci = 0; ci < (int)clusters.size(); ++ci) {
            if (!clusters[ci].usable) continue;   // 只用通过形状筛选的簇更新已有条目
            for (int ei = 0; ei < (int)pool.size(); ++ei) {
                Point2f pred = pool[ei].pos + pool[ei].vel;
                float gate = min(cfg.pool_gate_max,
                                 cfg.pool_match_gate + cfg.pool_gate_per_absent * pool[ei].absent);
                float d = (float)norm(clusters[ci].center - pred);
                if (d < gate) cands.push_back({d, ci, ei});
            }
        }
        sort(cands.begin(), cands.end(),
             [](const Cand& a, const Cand& b) { return a.d < b.d; });
        vector<bool> cl_used(clusters.size(), false), e_used(pool.size(), false);
        for (auto& c : cands) {
            if (cl_used[c.ci] || e_used[c.ei]) continue;
            cl_used[c.ci] = e_used[c.ei] = true;
            PoolEntry& e = pool[c.ei];
            Point2f prev = e.pos;
            e.pos = 0.7f * e.pos + 0.3f * clusters[c.ci].center;
            e.vel = 0.85f * e.vel + 0.15f * (e.pos - prev);
            e.vel.x = max(-cfg.scene_vel_max, min(cfg.scene_vel_max, e.vel.x));
            e.vel.y = max(-cfg.scene_vel_max, min(cfg.scene_vel_max, e.vel.y));
            e.area = clusters[c.ci].total_area;
            e.inner_ema = 0.8f * e.inner_ema + 0.2f * clusters[c.ci].inner_density;
            e.recent.push_back(1);
            e.absent = 0;
            // 滞回：已点亮时需跌破退出阈值才熄灭，避免靶标动画造成状态闪烁
            e.lit = e.lit ? (e.inner_ema >= cfg.lit_inner_exit)
                          : (e.inner_ema >= cfg.lit_inner_min);
            e.lit_recent.push_back(e.lit ? 1 : 0);
            if (e.lit) e.since_lit = 0; else ++e.since_lit;
        }
        for (int ei = 0; ei < (int)pool.size(); ++ei) {
            PoolEntry& e = pool[ei];
            if (e_used[ei]) { ++e.age; continue; }
            e.recent.push_back(0);
            e.lit_recent.push_back(0);
            ++e.absent;
            ++e.age;
            e.pos += e.vel;               // 按自身速度外推（相机漂移）
            e.lit = false;
            ++e.since_lit;
        }
        for (auto& e : pool) {
            while ((int)e.recent.size() > cfg.stable_window) e.recent.pop_front();
            while ((int)e.lit_recent.size() > cfg.stable_window) e.lit_recent.pop_front();
        }
        // 场景漂移：在场条目速度的逐分量中位数（比均值抗旋转分量污染）
        {
            vector<float> vx, vy;
            for (int ei = 0; ei < (int)min(pool.size(), e_used.size()); ++ei)
                if (e_used[ei]) { vx.push_back(pool[ei].vel.x); vy.push_back(pool[ei].vel.y); }
            if ((int)vx.size() >= 2) {
                sort(vx.begin(), vx.end()); sort(vy.begin(), vy.end());
                Point2f sv(vx[vx.size() / 2], vy[vy.size() / 2]);
                sv.x = max(-cfg.scene_vel_max, min(cfg.scene_vel_max, sv.x));
                sv.y = max(-cfg.scene_vel_max, min(cfg.scene_vel_max, sv.y));
                scene.vel = 0.9f * scene.vel + 0.1f * sv;
            }
        }
        // 新条目（同样要求通过形状筛选，避免灯带/实心块入池）
        for (int ci = 0; ci < (int)clusters.size(); ++ci) {
            if (cl_used[ci]) continue;
            if (!clusters[ci].usable) continue;
            if (clusters[ci].total_area < cfg.pool_new_min_area) continue;
            PoolEntry e;
            e.id = pool_next_id++;
            e.pos = clusters[ci].center;
            e.vel = Point2f(0, 0);
            e.area = clusters[ci].total_area;
            e.inner_ema = clusters[ci].inner_density;
            e.recent.push_back(1);
            e.age = 1;
            e.since_lit = e.inner_ema >= cfg.lit_inner_min ? 0 : 1000;
            e.lit = e.inner_ema >= cfg.lit_inner_min;
            e.lit_recent.push_back(e.lit ? 1 : 0);
            pool.push_back(e);
        }
        for (auto& e : pool)
            if (e.absent > cfg.pool_absent_drop && e.tid > 0)
                dropped_tids.push_back({e.pos, e.tid, frame_idx});
        pool.erase(remove_if(pool.begin(), pool.end(),
                             [&](const PoolEntry& e) { return e.absent > cfg.pool_absent_drop; }),
                   pool.end());

        auto stable = [&](const PoolEntry& e) {
            int h = 0;
            for (int v : e.recent) h += v;
            return h >= cfg.stable_min_hits;
        };
        // 持续点亮：最近窗口内点亮次数（区分真点亮与未点亮结构的瞬态高亮）
        auto litPersistent = [&](const PoolEntry& e) {
            int h = 0;
            for (int v : e.lit_recent) h += v;
            return h >= cfg.lit_min_hits;
        };

        // ================= 2. 锁定状态机 =================
        auto entryById = [&](int id) -> PoolEntry* {
            for (auto& e : pool) if (e.id == id) return &e;
            return nullptr;
        };
        auto bestLitEntry = [&](int exclude_id) -> PoolEntry* {
            PoolEntry* best = nullptr;
            for (auto& e : pool) {
                if (e.id == exclude_id) continue;
                if (!e.lit || !litPersistent(e) || !stable(e) || e.age < cfg.pool_min_age)
                    continue;
                if (e.area < cfg.lock_area_min) continue;
                if (!best || e.area > best->area) best = &e;
            }
            return best;
        };

        // 多目标 ID：点亮且稳定的条目都分配对外展示 ID（task_4 双目标场景）。
        // 展示门槛略低于锁定门槛，保证"另一个有效目标"也能被标注；
        // 条目重建时若位置接近刚删除的条目，继承其 ID（避免 ID 号膨胀/跳变）。
        for (auto& e : pool)
            if (e.tid == 0 && e.lit && stable(e) &&
                e.age >= cfg.tid_min_age && e.area >= cfg.tid_min_area) {
                int inherited = 0;
                for (size_t k = 0; k < dropped_tids.size(); ++k) {
                    if (frame_idx - dropped_tids[k].frame <= 150 &&
                        norm(e.pos - dropped_tids[k].pos) < 80.f) {
                        inherited = dropped_tids[k].tid;
                        dropped_tids.erase(dropped_tids.begin() + k);
                        break;
                    }
                }
                e.tid = inherited > 0 ? inherited : next_id++;
            }

        if (locked_entry >= 0) {
            PoolEntry* e = entryById(locked_entry);
            if (!e) {   // 条目被删除（长期缺席）
                events.push_back("frame " + to_string(frame_idx) + ": target ID " +
                                 to_string(t_id) + " entry dropped, release");
                locked_entry = -1; t_status = "searching";
            } else if (e->absent == 0 && e->since_lit <= cfg.dark_lost_frames) {
                // 簇在场且最近若干帧内点亮过 → 有效观测（抗点亮状态短暂抖动）
                t_lost = 0;
                t_status = "detected";
                t_show = e->pos;
            } else {
                ++t_lost;
                t_show = e->pos + e->vel * (float)min(t_lost, 10);
                t_status = "lost";
                // 熄灭切换：当前目标连续非点亮，且存在其他点亮稳定目标
                if (e->since_lit >= cfg.dark_switch_frames) {
                    PoolEntry* other = bestLitEntry(locked_entry);
                    if (other) {
                        int new_id = other->tid > 0 ? other->tid : next_id++;
                        other->tid = new_id;
                        events.push_back("frame " + to_string(frame_idx) + ": target ID " +
                                         to_string(t_id) + " extinguished, reselect -> ID " +
                                         to_string(new_id) +
                                         (new_id == t_id ? " (same blade re-acquired)" : ""));
                        locked_entry = other->id;
                        t_id = new_id;
                        ++n_lock_selections;
                        t_lost = 0;
                        t_status = "detected";
                        t_show = other->pos;
                    }
                }
            }
            // 持续丢失超时：释放，等待重选
            if (locked_entry >= 0 && t_lost > cfg.lost_tolerance) {
                events.push_back("frame " + to_string(frame_idx) + ": target ID " +
                                 to_string(t_id) + " released after " +
                                 to_string(t_lost) + " lost frames");
                locked_entry = -1; t_status = "searching";
            }
        }
        if (locked_entry < 0) {
            PoolEntry* best = bestLitEntry(-1);
            if (best) {
                locked_entry = best->id;
                t_id = best->tid > 0 ? best->tid : next_id++;
                best->tid = t_id;
                ++n_lock_selections;
                t_lost = 0;
                t_status = "detected";
                t_show = best->pos;
                events.push_back("frame " + to_string(frame_idx) + ": locked target ID " +
                                 to_string(t_id) + " at (" + fmtf(best->pos.x, 0) + "," +
                                 fmtf(best->pos.y, 0) + ")");
            } else {
                t_status = "searching";
            }
        }
        if (t_status == "detected") ++n_detected;
        else if (locked_entry >= 0) ++n_lost;
        int lit_cnt = 0;
        vector<Point2f> lit_positions;
        for (auto& e : pool)
            if (e.lit && stable(e) && e.area >= cfg.lock_area_min) {
                ++lit_cnt;
                lit_positions.push_back(e.pos);
            }
        if (lit_cnt > 0) ++n_lit_frames;
        if (lit_cnt >= 2) ++n_dual_frames;

        // ================= 3. R 标：签名检测 + 场景补偿链式跟踪 =================
        // R 字母 LED 的实测签名（两种视频验证一致）：半径≈12、面积≈260、圆形度 0.33~0.62、
        // 填充率 0.42~0.75，且在同尺寸块中孤立（流向圆点成串、环碎片圆形度过低）。
        // R 为脉冲 LED，可见率约 30%，因此按"链"累积命中：链用 位置+速度+场景漂移 预测，
        // 允许长时间间隙；命中最多的链即 R。不可见时保持上一位置并随场景漂移外推。
        auto rSignature = [&](const Blob& b) {
            if (b.radius < cfg.r_sig_r_min || b.radius > cfg.r_sig_r_max) return false;
            if (b.area < cfg.r_sig_area_min || b.area > cfg.r_sig_area_max) return false;
            if (b.circ < cfg.r_sig_circ_min || b.circ > cfg.r_sig_circ_max) return false;
            float fill = b.area / (float)(CV_PI * b.radius * b.radius + 1e-6f);
            if (fill < cfg.r_sig_fill_min || fill > cfg.r_sig_fill_max) return false;
            // 孤立性：同尺寸（±50%）块中，40px 内无邻居
            for (auto& o : blobs) {
                if (&o == &b) continue;
                float rr = max(o.radius, b.radius) / max(1e-6f, min(o.radius, b.radius));
                if (rr <= 1.5f && norm(o.center - b.center) < 40.f) return false;
            }
            return true;
        };
        vector<Point2f> r_dets;
        for (auto& b : blobs)
            if (rSignature(b)) r_dets.push_back(b.center);
        // 链匹配（贪心最近邻，带场景漂移补偿）
        {
            vector<bool> ch_used(r_chains.size(), false);
            for (auto& dpos : r_dets) {
                int best = -1; float bd = 1e9f;
                for (int ci = 0; ci < (int)r_chains.size(); ++ci) {
                    if (ch_used[ci]) continue;
                    RChain& ch = r_chains[ci];
                    int dt = frame_idx - ch.last;
                    Point2f pred = ch.pos + ch.vel * (float)dt + scene.vel * (float)dt;
                    float dd = (float)norm(dpos - pred);
                    float gate = min(cfg.r_chain_gate_max, cfg.r_chain_gate + cfg.r_chain_gate_dt * dt);
                    if (dd < gate && dd < bd) { bd = dd; best = ci; }
                }
                if (best >= 0) {
                    RChain& ch = r_chains[best];
                    Point2f prev = ch.pos;
                    ch.pos = 0.6f * ch.pos + 0.4f * dpos;
                    ch.vel = 0.8f * ch.vel + 0.2f * (ch.pos - prev);
                    ch.last = frame_idx;
                    ch.hits.push_back(1);
                    ch_used[best] = true;
                } else {
                    RChain ch;
                    ch.pos = dpos;
                    ch.vel = Point2f(0, 0);
                    ch.last = frame_idx;
                    ch.hits.push_back(1);
                    r_chains.push_back(ch);
                    ch_used.push_back(true);
                }
            }
            for (auto& ch : r_chains) {
                if (ch.last != frame_idx) ch.hits.push_back(0);
                while ((int)ch.hits.size() > 30) ch.hits.pop_front();
            }
            r_chains.erase(remove_if(r_chains.begin(), r_chains.end(),
                                     [&](const RChain& ch) {
                                         int h = 0;
                                         for (int v : ch.hits) h += v;
                                         return h == 0 || frame_idx - ch.last > cfg.r_chain_drop;
                                     }),
                           r_chains.end());
        }
        // 选命中最多的链作为 R
        RChain* best_chain = nullptr;
        int best_hits = 0;
        for (auto& ch : r_chains) {
            int h = 0;
            for (int v : ch.hits) h += v;
            // 位置合理性：R 是转轴，距当前点亮目标不应过远
            float dmin = 1e9f;
            for (auto& tp : lit_positions) dmin = min(dmin, (float)norm(tp - ch.pos));
            if (!lit_positions.empty() && dmin > cfg.r_max_target_dist) continue;
            if (h > best_hits) { best_hits = h; best_chain = &ch; }
        }
        if (best_chain) {
            if (frame_idx - best_chain->last == 0) {
                // 本帧命中：字形可见
                Point2f g = best_chain->pos;
                if (!r_has) { r_est = g; r_vel = Point2f(0, 0); }
                else {
                    r_vel = 0.8f * r_vel + 0.2f * (g - r_est);
                    r_est = 0.5f * r_est + 0.5f * g;
                }
                r_has = true; r_lost = 0; r_status = "detected";
                ++r_detected_cnt;
            } else {
                // 未命中：保持位置 + 场景漂移外推
                if (!r_has) { r_est = best_chain->pos; r_vel = Point2f(0, 0); r_has = true; }
                r_est += scene.vel;
                ++r_lost;
                r_status = (frame_idx - best_chain->last <= cfg.r_hold_frames) ? "estimated" : "lost";
                ++r_estimated_cnt;
            }
        } else if (r_has) {
            // 无任何链（长时间不可见）：停止外推，标记丢失，避免给出错误 R 位置
            ++r_lost;
            r_status = "lost";
        }

        // ================= 4. 角度 =================
        float theta_deg = 0;
        bool theta_ok = false;
        if (r_has && t_status == "detected") {
            theta_deg = atan2(r_est.y - t_show.y, t_show.x - r_est.x) * 180.f / (float)CV_PI;
            if (theta_deg < 0) theta_deg += 360.f;
            theta_ok = true;
        }

        // ================= 5. 可视化 =================
        Mat annotated = frame.clone();
        Mat binary;
        cvtColor(mask, binary, COLOR_GRAY2BGR);

        // 其他有效目标（点亮且稳定、已分配 ID，但不是当前锁定目标）：青色圈 + ID
        int n_lit_now = 0;
        for (auto& e : pool) {
            if (!stable(e) || e.absent > 2) continue;
            bool is_locked = e.id == locked_entry;
            if (is_locked) continue;
            if (e.lit) ++n_lit_now;
            if (e.lit && e.tid > 0) {
                Scalar c = Scalar(255, 200, 0);
                float tr = max(8.f, sqrt(e.area / (float)CV_PI));
                circle(annotated, e.pos, (int)tr, c, 2, LINE_AA);
                circle(binary, e.pos, (int)tr, c, 2, LINE_AA);
                drawMarker(annotated, e.pos, c, MARKER_TILTED_CROSS, 10, 1);
                drawLabel(annotated, "ID " + to_string(e.tid),
                          Point((int)e.pos.x + (int)tr + 6, (int)e.pos.y - (int)tr), c, 0.55, 1);
            } else {
                Scalar c = Scalar(130, 130, 130);
                circle(annotated, e.pos, (int)max(8.f, sqrt(e.area / (float)CV_PI)), c, 1, LINE_AA);
            }
        }
        for (auto& cl : clusters) {
            Scalar c = cl.usable ? Scalar(160, 160, 160) : Scalar(90, 90, 90);
            circle(binary, cl.center, (int)cl.enc_radius, c, 1, LINE_AA);
        }

        if (r_has) {
            Scalar rc = (r_status == "detected") ? Scalar(0, 255, 0) : Scalar(0, 180, 255);
            circle(annotated, r_est, 6, rc, FILLED, LINE_AA);
            circle(annotated, r_est, 12, rc, 2, LINE_AA);
            drawLabel(annotated, "R (" + r_status + ")",
                      Point((int)r_est.x + 14, (int)r_est.y - 10), rc);
            circle(binary, r_est, 12, rc, 2, LINE_AA);
        } else {
            drawLabel(annotated, "R searching", Point(20, 70), Scalar(0, 165, 255), 0.6, 1);
        }

        if (locked_entry >= 0) {
            Scalar tc = (t_status == "detected") ? Scalar(0, 255, 255) : Scalar(0, 0, 255);
            PoolEntry* e = entryById(locked_entry);
            float tr = e ? max(8.f, sqrt(e->area / (float)CV_PI)) : 20.f;
            if (t_status == "detected") {
                circle(annotated, t_show, (int)tr, tc, 2, LINE_AA);
                drawMarker(annotated, t_show, tc, MARKER_CROSS, 14, 2);
                circle(binary, t_show, (int)tr, tc, 2, LINE_AA);
                if (r_has) line(annotated, r_est, t_show, Scalar(255, 0, 255), 2, LINE_AA);
                string info = "ID " + to_string(t_id) + " detected";
                if (theta_ok) info += " th=" + fmtf(theta_deg);
                drawLabel(annotated, info,
                          Point((int)t_show.x + (int)tr + 8, (int)t_show.y - (int)tr), tc, 0.6, 1);
            } else {
                circle(annotated, t_show, 26, tc, 2, LINE_AA);
                drawLabel(annotated, "ID " + to_string(t_id) + " LOST " + to_string(t_lost),
                          Point((int)t_show.x - 60, (int)t_show.y - 32), tc, 0.6, 1);
            }
        } else {
            drawLabel(annotated, "target searching", Point(20, 95), Scalar(0, 255, 255), 0.6, 1);
        }

        drawLabel(annotated,
                  "frame " + to_string(frame_idx) + "  clusters=" + to_string(clusters.size()) +
                  "  lit=" + to_string(n_lit_now),
                  Point(20, 28), Scalar(255, 255, 255), 0.6, 1);
        w_overlay.write(annotated);
        w_binary.write(binary);

        csv << frame_idx << "," << r_status << ","
            << (r_has ? fmtf(r_est.x) : "") << "," << (r_has ? fmtf(r_est.y) : "") << ","
            << t_status << "," << (locked_entry >= 0 ? to_string(t_id) : "") << ","
            << fmtf(t_show.x) << "," << fmtf(t_show.y) << ","
            << (theta_ok ? fmtf(theta_deg) : "") << ","
            << clusters.size() << "," << n_lit_now << "\n";

        if (debug) {
            cout << "f" << frame_idx << " " << t_status << " id=" << t_id
                 << " t=(" << fmtf(t_show.x, 0) << "," << fmtf(t_show.y, 0) << ")"
                 << " R[" << r_status << "](" << fmtf(r_est.x, 0) << "," << fmtf(r_est.y, 0) << ")"
                 << " pool:";
            for (auto& e : pool)
                cout << " #" << e.id << "(" << fmtf(e.pos.x, 0) << "," << fmtf(e.pos.y, 0)
                     << ";in" << fmtf(e.inner_ema, 2) << (e.lit ? "L" : "")
                     << (stable(e) ? "S" : "") << "a" << e.absent << ")";
            cout << endl;
        }

        if (show) {
            imshow("Overlay", annotated);
            imshow("Binary", binary);
            int key = waitKey(1);
            if (key == 'q' || key == 27) break;
        }
    }

    int total = frame_idx;
    cout << "==== summary: " << stem << " ====" << endl;
    cout << "frames=" << total << " fps=" << fps << " size=" << W << "x" << H << endl;
    cout << "target detected " << n_detected << " (" << 100.0 * n_detected / max(total, 1) << "%)"
         << ", lost " << n_lost << " (" << 100.0 * n_lost / max(total, 1) << "%)"
         << ", searching " << (total - n_detected - n_lost) << " ("
         << 100.0 * (total - n_detected - n_lost) / max(total, 1) << "%)" << endl;
    cout << "frames with any lit stable entry: " << n_lit_frames << " ("
         << 100.0 * n_lit_frames / max(total, 1) << "%)" << endl;
    cout << "frames with >=2 lit stable entries (dual target): " << n_dual_frames << " ("
         << 100.0 * n_dual_frames / max(total, 1) << "%)" << endl;
    cout << "R detected " << r_detected_cnt << " (" << 100.0 * r_detected_cnt / max(total, 1) << "%)"
         << ", estimated " << r_estimated_cnt << " (" << 100.0 * r_estimated_cnt / max(total, 1) << "%)" << endl;
    cout << "display ids total = " << next_id - 1
         << ", lock selections (ID assignments) = " << n_lock_selections << endl;
    cout << "events:" << endl;
    for (auto& e : events) cout << "  " << e << endl;

    ofstream summary((fs::path(out_dir) / "summary.txt").string());
    summary << "frames=" << total << "\nfps=" << fps << "\nsize=" << W << "x" << H << "\n";
    summary << "target_detected=" << n_detected << "\ntarget_lost=" << n_lost << "\n";
    summary << "lit_frames=" << n_lit_frames << "\n";
    summary << "r_detected=" << r_detected_cnt << "\nr_estimated=" << r_estimated_cnt << "\n";
    summary << "ids_total=" << next_id - 1 << "\nlock_selections=" << n_lock_selections << "\n";
    for (auto& e : events) summary << "event: " << e << "\n";

    csv.close(); summary.close();
    w_overlay.release(); w_binary.release();
    cap.release();
    if (show) destroyAllWindows();
    return 0;
}
