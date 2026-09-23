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
    // 点亮判别
    float inner_radius_ratio = 0.5f;
    float lit_inner_min = 0.13f;
    // 位置池与锁定
    float pool_match_gate = 55.f;     // 簇-条目关联门限（px）
    float pool_gate_per_absent = 4.f;
    float pool_gate_max = 110.f;
    float pool_new_min_area = 150.f;  // 新条目所需最小簇面积
    int   pool_absent_drop = 90;      // 连续缺席多久删除条目
    int   stable_min_hits = 6;        // 最近 10 帧在场≥6 次视为稳定结构
    int   stable_window = 10;
    int   pool_min_age = 10;          // 锁定要求条目至少存在 10 帧
    float lock_area_min = 250.f;
    int   lost_tolerance = 45;        // 持续丢失容忍帧数（1.5s）
    int   dark_switch_frames = 15;    // 熄灭连续帧数达到即允许切换（且有其他点亮目标）
    // R 标
    float glyph_r_min = 5.f, glyph_r_max = 16.f;
    float glyph_area_min = 60.f, glyph_area_max = 450.f;
    float glyph_circ_max = 0.82f;
    float glyph_gate = 65.f;
    float glyph_protect = 45.f;       // R 估计位置附近的块不参与聚类
    float scene_vel_max = 6.f;
    int   r_coast_giveup = 150;       // 字形连续不可见多久后放弃 R 估计
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
    int absent = 0;
    int age = 0;                // 入池以来的帧数
    int dark = 0;               // 连续非点亮帧数
    bool lit = false;
};

struct SceneState { Point2f vel{0, 0}; };   // 场景整体漂移（相机运动估计）

struct GlyphCand {          // R 标初始化用的字形候选
    Point2f pos;
    deque<int> recent;
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

// 聚类 + 形状筛选 + 点亮特征。r_protect 有效时，其附近块不作为核心（防 R 字母被条带簇吸收）。
static vector<Cluster> clusterTargets(const vector<Blob>& blobs, const Mat& mask,
                                      const Config& cfg,
                                      bool r_protect, const Point2f& r_protect_at) {
    vector<int> core_ids;
    for (int i = 0; i < (int)blobs.size(); ++i) {
        if (blobs[i].radius < cfg.core_min_radius) continue;
        if (r_protect && norm(blobs[i].center - r_protect_at) < cfg.glyph_protect) continue;
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
        cl.usable = cl.enc_radius >= cfg.cluster_min_enc &&
                    cl.enc_radius <= cfg.cluster_max_enc &&
                    cl.fill <= cfg.cluster_fill_max &&
                    cl.aspect <= cfg.cluster_aspect_max &&
                    cl.total_area <= cfg.cluster_area_max;
        if (cl.enc_radius < cfg.cluster_min_enc || cl.enc_radius > cfg.cluster_max_enc) continue;
        // 点亮特征：0.5R 内侧圆的掩膜占比
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
    int t_lost = 0, t_dark = 0;
    string t_status = "searching";
    Point2f t_show(-1, -1);

    // ---- R 标状态 ----
    Point2f r_est(-1, -1), r_vel(0, 0);
    bool r_has = false;
    int r_lost = 0;
    string r_status = "searching";
    vector<GlyphCand> glyph_cands;

    int frame_idx = 0;
    int n_detected = 0, n_lost = 0, n_lit_frames = 0;
    int r_detected_cnt = 0, r_estimated_cnt = 0;
    Mat frame, mask;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        ++frame_idx;
        mask = makeMask(frame, cfg);
        vector<Blob> blobs = extractBlobs(mask, cfg);
        vector<Cluster> clusters = clusterTargets(
            blobs, mask, cfg, r_has, r_est + r_vel);

        // ================= 1. 位置池更新 =================
        // 贪心近邻关联：簇 ↔ 条目
        struct Cand { float d; int ci, ei; };
        vector<Cand> cands;
        for (int ci = 0; ci < (int)clusters.size(); ++ci) {
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
            if (clusters[c.ci].usable) {
                e.inner_ema = 0.75f * e.inner_ema + 0.25f * clusters[c.ci].inner_density;
            }
            e.recent.push_back(1);
            e.absent = 0;
            e.lit = e.inner_ema >= cfg.lit_inner_min;
            e.dark = e.lit ? 0 : e.dark + 1;
        }
        for (int ei = 0; ei < (int)pool.size(); ++ei) {
            PoolEntry& e = pool[ei];
            if (e_used[ei]) { ++e.age; continue; }
            e.recent.push_back(0);
            ++e.absent;
            ++e.age;
            e.pos += e.vel;               // 按自身速度外推（相机漂移）
            e.lit = false;
            e.dark += 1;
        }
        for (auto& e : pool)
            while ((int)e.recent.size() > cfg.stable_window) e.recent.pop_front();
        // 场景漂移：所有在场条目速度的均值（相机运动）
        Point2f sv(0, 0);
        int nsv = 0;
        for (int ei = 0; ei < (int)min(pool.size(), e_used.size()); ++ei)
            if (e_used[ei]) { sv += pool[ei].vel; ++nsv; }
        if (nsv > 0) {
            sv = sv * (1.f / nsv);
            sv.x = max(-cfg.scene_vel_max, min(cfg.scene_vel_max, sv.x));
            sv.y = max(-cfg.scene_vel_max, min(cfg.scene_vel_max, sv.y));
            scene.vel = 0.9f * scene.vel + 0.1f * sv;
        }
        // 新条目
        for (int ci = 0; ci < (int)clusters.size(); ++ci) {
            if (cl_used[ci]) continue;
            if (clusters[ci].total_area < cfg.pool_new_min_area) continue;
            PoolEntry e;
            e.id = pool_next_id++;
            e.pos = clusters[ci].center;
            e.vel = Point2f(0, 0);
            e.area = clusters[ci].total_area;
            e.inner_ema = clusters[ci].usable ? clusters[ci].inner_density : 0;
            e.recent.push_back(1);
            e.age = 1;
            e.lit = e.inner_ema >= cfg.lit_inner_min;
            pool.push_back(e);
        }
        pool.erase(remove_if(pool.begin(), pool.end(),
                             [&](const PoolEntry& e) { return e.absent > cfg.pool_absent_drop; }),
                   pool.end());

        auto stable = [&](const PoolEntry& e) {
            int h = 0;
            for (int v : e.recent) h += v;
            return h >= cfg.stable_min_hits;
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
                if (!e.lit || !stable(e) || e.age < cfg.pool_min_age) continue;
                if (e.area < cfg.lock_area_min) continue;
                if (!best || e.area > best->area) best = &e;
            }
            return best;
        };

        if (locked_entry >= 0) {
            PoolEntry* e = entryById(locked_entry);
            if (!e) {   // 条目被删除（长期缺席）
                events.push_back("frame " + to_string(frame_idx) + ": target ID " +
                                 to_string(t_id) + " entry dropped, release");
                locked_entry = -1; t_status = "searching";
            } else if (e->lit && e->absent == 0) {
                t_lost = 0; t_dark = 0;
                t_status = "detected";
                t_show = e->pos;
            } else {
                ++t_lost;
                t_show = e->pos + e->vel * (float)min(t_lost, 10);
                t_status = "lost";
                // 熄灭切换：当前目标连续非点亮，且存在其他点亮稳定目标
                if (e->dark >= cfg.dark_switch_frames) {
                    PoolEntry* other = bestLitEntry(locked_entry);
                    if (other) {
                        events.push_back("frame " + to_string(frame_idx) +
                                         ": target ID " + to_string(t_id) +
                                         " extinguished, reselect -> new ID");
                        locked_entry = other->id;
                        t_id = next_id++;
                        t_lost = 0; t_dark = 0;
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
                t_id = next_id++;
                t_lost = 0; t_dark = 0;
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
        bool any_lit = false;
        for (auto& e : pool) if (e.lit && stable(e)) any_lit = true;
        if (any_lit) ++n_lit_frames;

        // ================= 3. R 标：字形检测 + 场景漂移外推 =================
        // 候选字形：实心、小、圆形度低，且未被聚进任何簇
        vector<bool> in_cluster(blobs.size(), false);
        for (auto& cl : clusters) for (int id : cl.blob_ids) in_cluster[id] = true;
        const Blob* glyph = nullptr;
        float gd = 1e9f;
        Point2f r_pred = r_has ? r_est + r_vel + scene.vel : Point2f(-1, -1);
        for (size_t i = 0; i < blobs.size(); ++i) {
            if (in_cluster[i]) continue;
            const Blob& b = blobs[i];
            if (b.radius < cfg.glyph_r_min || b.radius > cfg.glyph_r_max) continue;
            if (b.area < cfg.glyph_area_min || b.area > cfg.glyph_area_max) continue;
            if (b.circ > cfg.glyph_circ_max || b.holes > 1) continue;
            if (r_has) {
                float d = (float)norm(b.center - r_pred);
                if (d < cfg.glyph_gate && d < gd) { gd = d; glyph = &b; }
            } else {
                // 初始化：累积字形候选，跨帧复现且被扇叶簇包围（≥3 个可用簇、角度跨度≥90°）
                // 才采用，避免锁到基地灯支架等孤立形状
                GlyphCand* best = nullptr;
                for (auto& g : glyph_cands) {
                    if (norm(g.pos - b.center) < 40.f) { best = &g; break; }
                }
                if (!best) {
                    GlyphCand g;
                    g.pos = b.center;
                    g.recent.push_back(1);
                    glyph_cands.push_back(g);
                } else {
                    best->pos = 0.8f * best->pos + 0.2f * b.center;
                    best->recent.push_back(1);
                }
            }
        }
        if (!r_has) {
            for (auto& g : glyph_cands) g.recent.push_back(0);
            for (auto& g : glyph_cands)
                while ((int)g.recent.size() > 12) g.recent.pop_front();
            for (size_t gi = 0; gi < glyph_cands.size(); ++gi) {
                GlyphCand& g = glyph_cands[gi];
                int hits = 0;
                for (int v : g.recent) hits += v;
                if (hits < 8) continue;
                // 周围扇叶簇包围检验
                vector<float> angs;
                for (auto& cl : clusters)
                    if (cl.usable) {
                        float d = (float)norm(cl.center - g.pos);
                        if (d > 80.f && d < 500.f)
                            angs.push_back(atan2(cl.center.y - g.pos.y, cl.center.x - g.pos.x));
                    }
                if (angs.size() < 3) continue;
                float amin = *min_element(angs.begin(), angs.end());
                float amax = *max_element(angs.begin(), angs.end());
                if (amax - amin < 90.f * (float)CV_PI / 180.f) continue;
                r_est = g.pos; r_vel = Point2f(0, 0);
                r_has = true; r_lost = 0; r_status = "detected";
                ++r_detected_cnt;
                events.push_back("frame " + to_string(frame_idx) + ": R adopted at (" +
                                 fmtf(r_est.x, 0) + "," + fmtf(r_est.y, 0) + ")");
                break;
            }
            if (!r_has) {
                glyph_cands.erase(remove_if(glyph_cands.begin(), glyph_cands.end(),
                                            [](const GlyphCand& g) { return g.recent.empty(); }),
                                  glyph_cands.end());
            }
        }
        if (glyph) {
            Point2f g = glyph->center;
            if (!r_has) { r_est = g; r_vel = Point2f(0, 0); }
            else {
                r_vel = 0.8f * r_vel + 0.2f * (g - r_est);
                r_est = 0.5f * r_est + 0.5f * g;
            }
            r_has = true; r_lost = 0; r_status = "detected";
            ++r_detected_cnt;
        } else if (r_has) {
            ++r_lost;
            r_est += scene.vel;   // 随场景漂移外推（R 相对风车静止）
            r_status = "estimated";
            ++r_estimated_cnt;
            if (r_lost > cfg.r_coast_giveup) {
                r_has = false; r_status = "searching";
                events.push_back("frame " + to_string(frame_idx) +
                                 ": R glyph lost too long, give up estimate");
            }
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

        int n_lit_now = 0;
        for (auto& e : pool) {
            if (!stable(e) || e.absent > 2) continue;
            bool is_locked = e.id == locked_entry;
            if (is_locked) continue;
            Scalar c = e.lit ? Scalar(255, 200, 0) : Scalar(130, 130, 130);
            if (e.lit) ++n_lit_now;
            circle(annotated, e.pos, (int)max(8.f, sqrt(e.area / (float)CV_PI)), c, 1, LINE_AA);
            if (e.lit)
                drawLabel(annotated, "cand", Point((int)e.pos.x + 10, (int)e.pos.y - 10), c, 0.45, 1);
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
    cout << "R detected " << r_detected_cnt << " (" << 100.0 * r_detected_cnt / max(total, 1) << "%)"
         << ", estimated " << r_estimated_cnt << " (" << 100.0 * r_estimated_cnt / max(total, 1) << "%)" << endl;
    cout << "ids used = " << next_id - 1 << endl;
    cout << "events:" << endl;
    for (auto& e : events) cout << "  " << e << endl;

    ofstream summary((fs::path(out_dir) / "summary.txt").string());
    summary << "frames=" << total << "\nfps=" << fps << "\nsize=" << W << "x" << H << "\n";
    summary << "target_detected=" << n_detected << "\ntarget_lost=" << n_lost << "\n";
    summary << "lit_frames=" << n_lit_frames << "\n";
    summary << "r_detected=" << r_detected_cnt << "\nr_estimated=" << r_estimated_cnt << "\n";
    summary << "ids_used=" << next_id - 1 << "\n";
    for (auto& e : events) summary << "event: " << e << "\n";

    csv.close(); summary.close();
    w_overlay.release(); w_binary.release();
    cap.release();
    if (show) destroyAllWindows();
    return 0;
}
