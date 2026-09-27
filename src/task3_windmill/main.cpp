// 任务3：真实能量机关视频的识别与稳定跟踪（v6：移植 RM_Buff_Tracker_GUT 架构）
//
// 参考实现：https://github.com/DH13768095744/RM_Buff_Tracker_GUT（桂林理工大学-群星战队，
// 纯传统视觉，无分类器）。本文件为其思路的 C++ 实现，并按本任务素材做了三处适配。
//
// 仓库三大核心（readme + utils/buffTracker.py）：
//   1) 双圆遮挡：以 R 为圆心，内圆（0.73*radius）实心涂黑抹掉 R 与流水灯，外圆
//      （1.4*radius）空心涂黑隔断外部光源——环带外的东西根本不进轮廓，
//      从结构上消灭"箭头串干扰""基地灯/横幅误锁"，无需任何排除判据。
//   2) 扇叶 = 绕 R 旋转 72°*i 的固定槽位：首帧拿到击打框与 R 后，其余 4 片位置直接
//      旋转得到；帧间关联 = 上一帧槽位框平移 ΔR 后与当前候选做 IoU（"对 IoU 引入
//      了以 R 为圆心的极坐标系下的角度差"）——相机平移被 R 差值天然抵消。
//   3) 扇叶分类：亮起 1 片 → 必为待击打目标；亮起数增加 → 新片为待击打、旧片为已击打。
//
// 按本任务素材做的适配（与仓库的差异，均有实测依据）：
//   a) 首帧前提自动化：仓库用人工 selectROI 给出首帧击打框与 R；本实现用 R 字形签名
//      （半径 11~14px、面积 215~325px²、圆形度 0.33~0.62、填充率 0.42~0.75）自动找 R，
//      再用 inner_density（点亮双环 0.24 / 未点亮单环 0.07，HSV 掩膜标定）自动找首个
//      击打框，两者都要求跨帧稳定后才初始化扇叶表。
//   b) 点亮判定：仓库素材的 HSV 下限是 V≥254（未点亮结构根本不进掩膜），因此它用
//      "计数"即可分类；本素材未点亮扇叶在部分时段也发光，且实测点亮靶标 V≥250 占比
//      仅 0.001~0.017（高亮判据不可分），故点亮 = inner_density 滞回（0.13 进/0.09 出），
//      仓库的计数状态机作为目标角色分配的辅助。
//   c) 膨胀核：仓库 7（其分辨率/目标更大），本素材结构较小，默认 5（Config 可调）；
//      膨胀掩膜仅用于扇叶片段发现，R 签名与 inner_density 用原始掩膜
//      （实测全掩膜膨胀会使 R 检出率 43%→1%）。
//   d) R 跟踪保留本工程已验证的"字形签名 + 单候选累积"（R 为脉冲 LED，可见率约
//      30%~40%，仓库的 CIoU 跟踪要求 R 每帧都可见，不适用）；不可见时按速度外推。
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

// ---------------- 参数 ----------------
struct Config {
    // HSV 掩膜（点亮结构为深橙红；基地灯偏黄、横幅偏绿在掩膜外）
    int hue_max = 17, s_min = 55, v_min = 80;
    int open_size = 3;
    int dilate_size = 5;               // 仓库 kernel=7，本素材结构较小取 5
    // 发光块通用筛选
    float blob_min_area = 18.f, blob_max_radius = 90.f, border_margin = 25.f;
    // 双圆遮挡比例（仓库：insideRate=0.73, outsideRate=1.4）
    float inside_rate = 0.73f, outside_rate = 1.4f;
    // 扇叶候选判据（仓库：0.4r < disBtm < disTop < 1.5r 且 rotated-rect 面积 > 2*R_bbox_area）
    float blade_dis_in = 0.4f, blade_dis_out = 1.5f;
    float blade_area_ratio = 1.2f;     // 仓库为 2.0；本素材弧片段较小，放宽到 1.2
    // R 字形签名（两视频实测一致）
    float r_r_min = 11.f, r_r_max = 14.f, r_area_min = 215.f, r_area_max = 325.f;
    float r_circ_min = 0.33f, r_circ_max = 0.62f, r_fill_min = 0.42f, r_fill_max = 0.75f;
    float r_gate = 70.f, r_vel_max = 60.f;
    int   r_init_hits = 3, r_window = 10, r_hold = 90;
    // 点亮判据（HSV 掩膜标定）
    float lit_inner_min = 0.13f, lit_inner_exit = 0.09f;
    float lit_ring_ema_min = 0.40f;    // 环峰值覆盖的 EMA 下限（时间积分，抗单帧闪烁）
    float ring_ema_a = 0.25f;          // EMA 系数（新样本权重）
    float ring_ema_decay = 0.92f;      // 未匹配帧的衰减
    float lit_ring_peak_min = 0.35f;   // 环峰值覆盖下限（双环 0.75~1.0，长条/箭头 ≤0.31；0.45 对 t4 暗环过严）
    float lit_max_dist_ratio = 1.5f;   // 点亮槽位到 R 的最大距离（×radius）
    // 首个击打框自动发现
    float fan_dist_min = 130.f, fan_dist_max = 320.f;   // 真实风车半径 ~170-215px；下限排除辐条/流水灯簇(~102px)，上限排除把远处杂块配成扇叶（task_4 曾配出 640px）
    float fan_box_area_min = 800.f;
    int   fan_init_hits = 3;
    // 箭头串（点亮目标的可靠结构信号：基地灯没有箭头指向它）
    float dot_r_min = 3.5f, dot_r_max = 12.f, dot_area_min = 20.f, dot_area_max = 250.f;
    float dot_link = 32.f;
    int   strip_min_dots = 3;
    float strip_max_perp = 8.f, strip_min_len = 40.f;
    float strip_perp_tol = 45.f, strip_along_max = 160.f;
    // 身份与丢失
    float iou_min = 0.02f;             // 槽位-片段关联的 IoU 下限（保持存活）
    float iou_strong = 0.15f;          // 强匹配阈值：只有强匹配才移动槽位框
    float assoc_center_ratio = 1.2f;   // 片段中心距离门限（×对角线；1.2 ≈ 不启用）
    int   slot_miss_reset = 60;        // 槽位连续未匹配多久后转为 shot
    int   lost_confirm = 3, lost_tolerance = 45;  // 3 帧即显示 LOST（身份仍保留 45 帧）
    int   dark_switch = 15;            // 已选槽位连续不点亮多少帧才允许切换（0.5s 防乒乓）
};

// ---------------- 基础数据结构 ----------------
struct BBox {
    float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
    float width()  const { return xmax - xmin; }
    float height() const { return ymax - ymin; }
    float area()   const { return max(0.f, width()) * max(0.f, height()); }
    Point2f center() const { return Point2f((xmin + xmax) / 2, (ymin + ymax) / 2); }
    float iou(const BBox& o) const {
        float ix0 = max(xmin, o.xmin), iy0 = max(ymin, o.ymin);
        float ix1 = min(xmax, o.xmax), iy1 = min(ymax, o.ymax);
        float iw = max(0.f, ix1 - ix0), ih = max(0.f, iy1 - iy0);
        float inter = iw * ih;
        float uni = area() + o.area() - inter;
        return uni > 0 ? inter / uni : 0.f;
    }
    static BBox fromCenter(const Point2f& c, float w, float h) {
        return BBox{c.x - w / 2, c.y - h / 2, c.x + w / 2, c.y + h / 2};
    }
    BBox translated(const Point2f& d) const {
        return BBox{xmin + d.x, ymin + d.y, xmax + d.x, ymax + d.y};
    }
};

struct Blob {
    Point2f c;
    float r = 0, area = 0, circ = 0;
    int holes = 0;
};

struct Fragment {              // 遮挡后环带内的连通域（扇叶候选片段）
    BBox box;
    float area = 0;
    float dis_in = 0, dis_out = 0;    // 旋转矩形内/外沿中点到 R 的距离（仓库 disBtm/disTop）
};

struct Strip {                 // 箭头串（流向灯）：从靠近 R 端指向远端（靶标一侧）
    Point2f near_end, far_end, dir;
    int n_dots = 0;
};

struct Slot {                  // 扇叶槽位（5 个，绕 R 72° 均布）
    int id = 0;
    BBox box;
    float init_w = 50.f, init_h = 50.f;   // 初始尺寸（钳制框不因误匹配无限变大）
    bool has = false;
    bool lit = false;
    string state = "unlighted";    // target / shot / unlighted
    float inner_ema = 0;
    float ring_ema = 0;        // 环峰值覆盖的 EMA（时间积分，抗单帧闪烁）
    int missed = 0;
};

static float wrapPi(float a) {
    while (a > (float)CV_PI) a -= 2.f * (float)CV_PI;
    while (a < -(float)CV_PI) a += 2.f * (float)CV_PI;
    return a;
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

// ---------------- 掩膜与块提取 ----------------
static Mat makeMask(const Mat& frame, const Config& cfg) {
    // 原始掩膜（开运算去噪）。R 签名与 inner_density 都在此掩膜上；
    // 扇叶片段发现另用膨胀掩膜（见主循环）。
    Mat hsv, mask;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    inRange(hsv, Scalar(0, cfg.s_min, cfg.v_min), Scalar(cfg.hue_max, 255, 255), mask);
    morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(cfg.open_size, cfg.open_size)));
    return mask;
}

static vector<Blob> extractBlobs(const Mat& mask, const Config& cfg) {
    vector<vector<Point>> contours;
    vector<Vec4i> hier;
    findContours(mask, contours, hier, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    const int W = mask.cols, H = mask.rows;
    vector<Blob> blobs;
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
        b.c = c; b.r = r; b.area = (float)a;
        b.circ = (float)(4 * CV_PI * a / (peri * peri + 1e-6));
        for (int ch = hier[i][2]; ch >= 0; ch = hier[ch][0])
            if (contourArea(contours[ch]) > 8) ++b.holes;
        blobs.push_back(b);
    }
    return blobs;
}

// 0.5R 内侧圆的掩膜占比（点亮双环 0.24 / 未点亮单环 0.07，HSV 掩膜标定）
static float innerDensity(const Mat& mask, Point2f c, float radius) {
    if (radius < 3.f) return 0.f;
    int x0 = max(0, (int)(c.x - radius)), x1 = min(mask.cols, (int)(c.x + radius));
    int y0 = max(0, (int)(c.y - radius)), y1 = min(mask.rows, (int)(c.y + radius));
    if (x1 <= x0 || y1 <= y0) return 0.f;
    Mat roi = mask(Rect(x0, y0, x1 - x0, y1 - y0));
    float inside = 0, total = 0;
    for (int yy = 0; yy < roi.rows; ++yy)
        for (int xx = 0; xx < roi.cols; ++xx) {
            float dx = x0 + xx - c.x, dy = y0 + yy - c.y;
            if (dx * dx + dy * dy > radius * radius) continue;
            ++total;
            if (roi.at<uchar>(yy, xx)) ++inside;
        }
    return total > 0 ? inside / total : 0.f;
}

// 环峰值覆盖：在多个半径上采样 8 扇区的掩膜覆盖，取最大值。
// 双环靶标在某个半径上 8 扇区全有能量（peak ≥0.75）；实心长条/箭头串在任意半径上
// 最多覆盖 2~3 个扇区（peak ≤0.31）——用于把"点亮"判据与长条/箭头区分开（实测）。
static float ringPeakCoverage(const Mat& mask, Point2f c, float rmax) {
    const int sectors = 8, steps = 10;
    float best = 0.f;
    if (rmax < 6.f) return 0.f;
    for (int ri = 0; ri < steps; ++ri) {
        float r = rmax * (0.25f + 0.9f * ri / (steps - 1));
        int hit = 0;
        for (int k = 0; k < sectors; ++k) {
            int cnt = 0, tot = 0;
            for (int t = 0; t < steps; ++t) {
                float ang = 2.f * (float)CV_PI * (k + (float)t / steps) / sectors;
                int x = (int)(c.x + r * cos(ang)), y = (int)(c.y + r * sin(ang));
                if (x < 0 || y < 0 || x >= mask.cols || y >= mask.rows) continue;
                ++tot;
                if (mask.at<uchar>(y, x)) ++cnt;
            }
            if (tot > 0 && cnt / (float)tot > 0.25f) ++hit;
        }
        best = max(best, hit / (float)sectors);
    }
    return best;
}

// R 字形签名（两视频实测一致；用户确认该判据稳定）
static bool rSignature(const Blob& b, const Config& cfg) {
    if (b.r < cfg.r_r_min || b.r > cfg.r_r_max) return false;
    if (b.area < cfg.r_area_min || b.area > cfg.r_area_max) return false;
    if (b.circ < cfg.r_circ_min || b.circ > cfg.r_circ_max) return false;
    float fill = b.area / (float)(CV_PI * b.r * b.r + 1e-6f);
    if (fill < cfg.r_fill_min || fill > cfg.r_fill_max) return false;
    return true;
}

// 仓库的旋转矩形内外沿：minAreaRect 四顶点按到 R 的距离排序，
// p1/p2 = 最远两点（外沿），p3/p4 = 其余（内沿）；返回内/外沿中点到 R 的距离
static void rectInOutDist(const RotatedRect& rr, const Point2f& R, float& dis_in, float& dis_out) {
    Point2f p[4];
    rr.points(p);
    array<pair<float, int>, 4> d;
    for (int i = 0; i < 4; ++i) d[i] = {(float)norm(p[i] - R), i};
    sort(d.begin(), d.end(), [](const pair<float,int>& a, const pair<float,int>& b) {
        return a.first > b.first; });
    Point2f mid_out = (p[d[0].second] + p[d[1].second]) * 0.5f;
    Point2f mid_in  = (p[d[2].second] + p[d[3].second]) * 0.5f;
    dis_out = (float)norm(mid_out - R);
    dis_in  = (float)norm(mid_in - R);
}

// 箭头串检测（精简版）：圆点 → 贪心链 → 共线校验 → 以 R 定向（近端=靠 R 的一端）。
// 被点亮的扇叶旁才有流向灯——基地灯/横幅没有，这是区分它们的最可靠结构信号。
static vector<Strip> detectStrips(const vector<Blob>& blobs, const Config& cfg,
                                  bool r_known, const Point2f& r_pos) {
    vector<Point2f> dots;
    for (auto& b : blobs) {
        if (b.r < cfg.dot_r_min || b.r > cfg.dot_r_max) continue;
        if (b.area < cfg.dot_area_min || b.area > cfg.dot_area_max) continue;
        if (b.circ < 0.15f || b.holes > 0) continue;
        dots.push_back(b.c);
    }
    vector<Strip> out;
    const int n = (int)dots.size();
    if (n < cfg.strip_min_dots) return out;
    vector<bool> used(n, false);
    for (int i = 0; i < n; ++i) {
        if (used[i]) continue;
        vector<int> chain{i};
        used[i] = true;
        while (true) {
            int best = -1; float bd = cfg.dot_link;
            for (int j = 0; j < n; ++j) {
                if (used[j]) continue;
                float d = (float)norm(dots[j] - dots[chain.back()]);
                if (d < bd) { bd = d; best = j; }
            }
            if (best < 0) break;
            chain.push_back(best);
            used[best] = true;
        }
        if ((int)chain.size() < cfg.strip_min_dots) continue;
        Point2f mean(0, 0);
        for (int id : chain) mean += dots[id];
        mean *= 1.f / chain.size();
        float sxx = 0, sxy = 0, syy = 0;
        for (int id : chain) {
            Point2f d = dots[id] - mean;
            sxx += d.x * d.x; sxy += d.x * d.y; syy += d.y * d.y;
        }
        float th = 0.5f * atan2(2 * sxy, sxx - syy);
        Point2f u(cos(th), sin(th));
        float max_perp = 0;
        for (int id : chain) {
            Point2f d = dots[id] - mean;
            max_perp = max(max_perp, fabsf(d.x * u.y - d.y * u.x));
        }
        if (max_perp > cfg.strip_max_perp) continue;
        int i0 = chain[0], i1 = chain[0];
        float t0 = 0, t1 = 0;
        for (int id : chain) {
            float t = (dots[id] - mean).dot(u);
            if (t < t0) { t0 = t; i0 = id; }
            if (t > t1) { t1 = t; i1 = id; }
        }
        if (t1 - t0 < cfg.strip_min_len) continue;
        Strip st;
        st.n_dots = (int)chain.size();
        st.near_end = dots[i0];
        st.far_end = dots[i1];
        st.dir = u;
        if (r_known && norm(st.near_end - r_pos) > norm(st.far_end - r_pos)) {
            swap(st.near_end, st.far_end);
            st.dir = u * -1.f;
        }
        out.push_back(st);
    }
    vector<Strip> uniq;                     // 同一条串的多个子串合并
    for (auto& st : out) {
        bool dup = false;
        for (auto& u2 : uniq) {
            float dotp = st.dir.x * u2.dir.x + st.dir.y * u2.dir.y;
            float ang = acos(min(1.f, fabsf(dotp))) * 180.f / (float)CV_PI;
            Point2f dv = st.near_end - u2.near_end;
            float perp = fabs(dv.x * u2.dir.y - dv.y * u2.dir.x);
            if (ang < 15.f && perp < 18.f) {
                if (st.n_dots > u2.n_dots) u2 = st;
                dup = true;
                break;
            }
        }
        if (!dup) uniq.push_back(st);
    }
    return uniq;
}

static bool stripSupports(const vector<Strip>& strips, const Point2f& p,
                          float ring_radius, const Config& cfg) {
    for (auto& st : strips) {
        Point2f dv = p - st.far_end;
        float along = dv.dot(st.dir);
        float perp = fabs(dv.x * st.dir.y - dv.y * st.dir.x);
        if (perp <= cfg.strip_perp_tol && along >= -ring_radius &&
            along <= cfg.strip_along_max) return true;
    }
    return false;
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
        else if (video_path == "../resources/task_3.mp4" && out_dir.empty() &&
                 args[i].rfind("--", 0) != 0) video_path = args[i];
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
    csv << "frame,r_status,r_x,r_y,t_status,t_id,t_x,t_y,theta_deg,n_blades,n_lit,n_strips\n";
    vector<string> events;

    enum class Phase { INIT_R, INIT_FAN, TRACK };
    Phase phase = Phase::INIT_R;

    // ---- R 标（签名 + 单候选累积 + 速度外推） ----
    Point2f r_est(-1, -1), r_vel(0, 0);
    int r_miss = 0, r_last_hit = -1000;
    deque<int> r_hits;
    bool r_valid = false;
    string r_status = "searching";

    // ---- 扇叶槽位 ----
    Point2f r_prev_center(-1, -1);    // 上一帧 R 中心（ΔR 计算，每帧末更新）
    int r_gap_frames = 0;             // R 连续不可见帧数（重验证触发用）
    array<Slot, 5> slots;
    for (int i = 0; i < 5; ++i) slots[i].id = i;
    float radius = 0;
    float last_radius = 0;            // 上次成功初始化的半径（重初始化合理性校验用）
    int lit_cnt_prev = 0;

    // ---- 目标输出 ----
    int t_id = 0, id_counter = 0, lock_selections = 0;
    int sel_slot = -1, t_lost = 0, sel_lost = 0, sel_dark = 0;
    string t_status = "searching";
    Point2f t_show(-1, -1);
    float t_ring_show = 0;

    // ---- 场景位移（大结构逐帧位移中位数 = 相机运动估计；用于 R 不可见时外推） ----
    vector<Point2f> prev_big;
    Point2f scene_disp(0, 0);

    int frame_idx = 0;
    int n_detected = 0, n_lost = 0, n_lit_frames = 0, n_dual = 0, n_observed = 0;
    int r_detected = 0, r_estimated = 0;
    Mat frame, mask, mask_d;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        ++frame_idx;
        mask = makeMask(frame, cfg);
        dilate(mask, mask_d, getStructuringElement(MORPH_RECT, Size(cfg.dilate_size, cfg.dilate_size)));
        vector<Blob> blobs = extractBlobs(mask, cfg);

        // ================= 0. 场景位移（相机运动估计） =================
        {
            vector<Point2f> big;
            for (auto& b : blobs)
                if (b.area >= 400.f) big.push_back(b.c);
            if (!prev_big.empty() && !big.empty()) {
                vector<float> dxs, dys;
                for (auto& p : prev_big) {
                    float bd = 1e9f; Point2f bc(0, 0);
                    for (auto& q : big) {
                        float d = (float)norm(q - p);
                        if (d < bd) { bd = d; bc = q; }
                    }
                    if (bd < 100.f) { dxs.push_back(bc.x - p.x); dys.push_back(bc.y - p.y); }
                }
                if ((int)dxs.size() >= 2) {
                    size_t mi = dxs.size() / 2;
                    nth_element(dxs.begin(), dxs.begin() + mi, dxs.end());
                    nth_element(dys.begin(), dys.begin() + mi, dys.end());
                    scene_disp = Point2f(dxs[mi], dys[mi]);
                }
            }
            prev_big = big;
        }

        // ================= 1. R 标（签名 + 单候选累积） =================
        const Blob* glyph = nullptr;
        float gd = 1e9f;
        Point2f r_pred = r_est + r_vel;
        for (auto& b : blobs) {
            if (!rSignature(b, cfg)) continue;
            bool isolated = true;                    // 同尺寸块中孤立
            for (auto& o : blobs) {
                if (&o == &b) continue;
                float ratio = max(o.r, b.r) / max(1e-6f, min(o.r, b.r));
                if (ratio <= 1.5f && norm(o.c - b.c) < 40.f) { isolated = false; break; }
            }
            if (!isolated) continue;
            if (r_valid) {
                float d = (float)norm(b.c - r_pred);
                if (d < cfg.r_gate && d < gd) { gd = d; glyph = &b; }
            } else {
                glyph = &b;
            }
        }
        bool r_hit = false;
        if (glyph) {
            Point2f g = glyph->c;
            if (!r_valid) { r_est = g; r_vel = Point2f(0, 0); }
            else {
                Point2f prev = r_est;
                r_est = 0.5f * r_est + 0.5f * g;
                r_vel = 0.8f * r_vel + 0.2f * (r_est - prev);
                r_vel.x = max(-cfg.r_vel_max, min(cfg.r_vel_max, r_vel.x));
                r_vel.y = max(-cfg.r_vel_max, min(cfg.r_vel_max, r_vel.y));
            }
            r_valid = true;
            r_miss = 0;
            r_last_hit = frame_idx;
            r_hit = true;
        } else if (r_valid) {
            ++r_miss;
            // 按场景位移外推（相机平移时所有大结构同动，比 R 自身陈旧速度可靠）
            r_est += scene_disp;
        }
        r_hits.push_back(r_hit ? 1 : 0);
        while ((int)r_hits.size() > cfg.r_window) r_hits.pop_front();
        int rhits = 0;
        for (int v : r_hits) rhits += v;
        if (!r_valid && rhits >= cfg.r_init_hits) {
            r_valid = true;
            events.push_back("frame " + to_string(frame_idx) + ": R locked at (" +
                             fmtf(r_est.x, 0) + "," + fmtf(r_est.y, 0) + ")");
        }
        if (r_valid && frame_idx - r_last_hit > cfg.r_hold) {
            r_valid = false; r_est = Point2f(-1, -1); r_vel = Point2f(0, 0);
            r_hits.clear();
            events.push_back("frame " + to_string(frame_idx) + ": R lost, re-searching");
        }
        if (r_valid && frame_idx == r_last_hit) { r_status = "detected"; ++r_detected; }
        else if (r_valid) { r_status = "estimated"; ++r_estimated; }
        else r_status = "searching";
        if (phase == Phase::INIT_R && r_valid) phase = Phase::INIT_FAN;

        Point2f R = r_est;
        bool r_ok = r_valid && R.x > 0;
        bool r_detected_now = (r_status == "detected");
        bool r_live = r_ok && r_detected_now;   // v6.3 冻结原则：R 没有实测→一切以 R 为锚的操作冻结
        int r_gap_prev = r_gap_frames;          // 上一帧的 R 连续不可见数（重验证触发用）
        r_gap_frames = r_detected_now ? 0 : min(r_miss, 999);
        Point2f r_delta = (r_ok && r_prev_center.x > 0) ? R - r_prev_center : Point2f(0, 0);
        r_prev_center = R;              // 供下一帧计算 ΔR

        vector<Strip> strips = detectStrips(blobs, cfg, r_ok, R);

        // ================= 2. 双圆遮挡（仓库核心） =================
        // 仅 R 实测时做遮挡：外推 R 有误差，涂错位置会粉碎真片段
        if (r_live && phase == Phase::TRACK && radius > 10.f) {
            circle(mask_d, R, (int)(radius * cfg.inside_rate), Scalar(0), -1);   // 抹掉 R+流水灯
            circle(mask_d, R, (int)(radius * cfg.outside_rate), Scalar(0), 3);   // 隔断外部光源
        }

        // ================= 3. 扇叶片段（环带内轮廓） =================
        vector<Fragment> frags;
        {
            vector<vector<Point>> contours;
            findContours(mask_d, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            for (auto& cont : contours) {
                double a = contourArea(cont);
                if (a < 60) continue;
                Rect br = boundingRect(cont);
                Fragment f;
                f.box = BBox{(float)br.x, (float)br.y,
                             (float)(br.x + br.width), (float)(br.y + br.height)};
                if (f.box.center().x < 20 || f.box.center().y < 20 ||
                    f.box.center().x > W - 20 || f.box.center().y > H - 20) continue;
                f.area = (float)a;
                if (r_ok && radius > 10.f) {
                    RotatedRect rr = minAreaRect(cont);
                    rectInOutDist(rr, R, f.dis_in, f.dis_out);
                }
                frags.push_back(f);
            }
        }

        // ================= 4. 初始化 / 跟踪 =================
        if (phase == Phase::INIT_FAN && r_ok) {
            // 首个击打框自动发现：距 R 合理 + 面积达标 + inner 点亮，要求跨帧稳定
            // 重初始化时半径必须在上次稳定半径的 [0.65,1.55] 倍内：
            // task_3 f540 曾把 R 旁流水灯簇当扇叶（radius 102 vs 真实 213），后半段全程错位
            float dmin = cfg.fan_dist_min, dmax = cfg.fan_dist_max;
            if (last_radius > 0) {
                dmin = max(dmin, 0.65f * last_radius);
                dmax = min(dmax, 1.55f * last_radius);
            }
            const Fragment* best = nullptr;
            for (auto& f : frags) {
                float d = (float)norm(f.box.center() - R);
                if (d < dmin || d > dmax) continue;
                if (f.box.area() < cfg.fan_box_area_min) continue;
                float ir = 0.25f * min(f.box.width(), f.box.height());
                float inner = innerDensity(mask, f.box.center(), ir);
                if (inner < cfg.lit_inner_min) continue;
                // 初始化必须箭头串指向（基地灯无箭头指向，实测 task_4 f8 曾误锁）
                if (!stripSupports(strips, f.box.center(), 40.f, cfg)) continue;
                if (!best || f.box.area() > best->box.area()) best = &f;
            }
            static int fan_hits = 0;
            static int fan_last_hit = -1000;
            static Point2f fan_last_pos(-1, -1);
            if (best) {
                if (frame_idx - fan_last_hit <= 3 && fan_last_pos.x > 0 &&
                    norm(best->box.center() - fan_last_pos) > 40.f)
                    fan_hits = 0;                        // 位置跳变，重新计
                ++fan_hits;
                fan_last_hit = frame_idx;
                fan_last_pos = best->box.center();
            } else if (frame_idx - fan_last_hit > 3) {
                fan_hits = 0;
            }
            if (best && fan_hits >= cfg.fan_init_hits) {
                radius = (float)norm(best->box.center() - R);
                last_radius = radius;
                slots[0].box = best->box;
                slots[0].init_w = best->box.width();
                slots[0].init_h = best->box.height();
                slots[0].has = true;
                slots[0].lit = true;
                slots[0].state = "target";
                slots[0].inner_ema = innerDensity(mask, best->box.center(),
                                                  0.25f * min(best->box.width(), best->box.height()));
                for (int k = 1; k < 5; ++k) {            // 其余 4 槽位：绕 R 旋转 72°*k
                    float ang = (float)(72.0 * k * CV_PI / 180.0);
                    Point2f rel = best->box.center() - R;
                    Point2f cc(R.x + rel.x * cos(ang) - rel.y * sin(ang),
                               R.y + rel.x * sin(ang) + rel.y * cos(ang));
                    slots[k].box = BBox::fromCenter(cc, best->box.width(), best->box.height());
                    slots[k].has = true;
                }
                phase = Phase::TRACK;
                t_id = ++id_counter;
                sel_slot = 0;
                fan_hits = 0;
                events.push_back("frame " + to_string(frame_idx) +
                                 ": initialized (R + fan), radius=" + fmtf(radius, 0));
            }
        }

        if (phase == Phase::TRACK && r_live) {
            // R 实测帧才做关联/更新；estimated 帧走下方冻结分支
            Point2f delta = r_delta;
            // R 重现重验证：盲窗 ≥3 帧后，用正确的 R 重查所有槽位径向门限，
            // 不合格的槽位重置（清 has），让正确片段重新播种/关联
            if (r_gap_prev >= 3 && radius > 10.f) {
                for (auto& s : slots) {
                    if (!s.has) continue;
                    float d = norm(s.box.center() - R);
                    if (d < 0.5f * radius || d > cfg.blade_dis_out * radius + 20.f) {
                        s.has = false;
                        events.push_back("frame " + to_string(frame_idx) +
                                         ": slot re-validated out of range, reset");
                    }
                }
            }
            // 片段 ↔ 槽位关联（IoU > 0）
            vector<vector<const Fragment*>> by_slot(5);
            vector<float> best_iou(5, 0.f);
            vector<bool> frag_used(frags.size(), false);
            vector<bool> matched(slots.size(), false);
            for (int si = 0; si < 5; ++si) {
                if (!slots[si].has) continue;
                BBox pb = slots[si].box.translated(delta);
                for (size_t fi = 0; fi < frags.size(); ++fi) {
                    if (frag_used[fi]) continue;
                    const Fragment& f = frags[fi];
                    if (radius > 10.f &&
                        !(f.dis_in > 0.5f * radius &&
                          f.dis_out < cfg.blade_dis_out * radius + 20.f)) continue;
                    float fiou = pb.iou(f.box);
                    if (fiou > cfg.iou_min) {
                        by_slot[si].push_back(&f);
                        frag_used[fi] = true;
                        best_iou[si] = max(best_iou[si], fiou);
                    }
                }
            }
            // 槽位更新（P1 语义）：所有匹配更新框（跟随真实结构）；
            // lit = inner 滞回 && 环峰值覆盖 ≥0.45（双环 0.75~1.0，长条/箭头 ≤0.31）
            // && 到 R 距离 ≤1.5×radius+30（底部灯排除）。验证失败 = 不点亮。
            int lit_now = 0;
            for (int si = 0; si < 5; ++si) {
                Slot& s = slots[si];
                if (!s.has) continue;
                if (!by_slot[si].empty()) {
                    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
                    for (auto* f : by_slot[si]) {
                        x0 = min(x0, f->box.xmin); y0 = min(y0, f->box.ymin);
                        x1 = max(x1, f->box.xmax); y1 = max(y1, f->box.ymax);
                    }
                    s.box = BBox{x0, y0, x1, y1};
                    // 尺寸钳制：框不能因误匹配无限变大/变小
                    {
                        float wlim = 1.8f * s.init_w + 20.f, hlim = 1.8f * s.init_h + 20.f;
                        float wmin = 0.35f * s.init_w, hmin = 0.35f * s.init_h;
                        Point2f c = s.box.center();
                        s.box = BBox::fromCenter(c,
                            max(wmin, min(wlim, s.box.width())),
                            max(hmin, min(hlim, s.box.height())));
                    }
                    s.missed = 0;
                    matched[si] = true;
                    float ir = 0.25f * min(s.box.width(), s.box.height());
                    float inner = innerDensity(mask, s.box.center(), ir);
                    s.inner_ema = 0.8f * s.inner_ema + 0.2f * inner;
                    // 点亮：inner 滞回 + 环峰值覆盖（双环 0.75~1.0，长条/箭头 ≤0.31）
                    s.lit = s.lit ? (s.inner_ema >= cfg.lit_inner_exit)
                                  : (s.inner_ema >= cfg.lit_inner_min);
                    float peak = ringPeakCoverage(mask, s.box.center(),
                                                  0.6f * min(s.box.width(), s.box.height()));
                    if (peak < cfg.lit_ring_peak_min) s.lit = false;
                    // 点亮到 R 的距离约束（底部灯等远距环状干扰）
                    if (r_ok && radius > 10.f &&
                        norm(s.box.center() - R) > cfg.lit_max_dist_ratio * radius + 30.f)
                        s.lit = false;
                    // 槽位中心径向下限：真靶标中心实测 ≥0.65×radius（p10），
                    // R 旁小方块/箭头串中部 <0.55×radius。条带检测仅 32% 帧可用，
                    // 不能作 lit 硬门限（v6.3 实测把检出打到 2.5%），改用几何门限。
                    if (r_ok && radius > 10.f &&
                        norm(s.box.center() - R) < 0.55f * radius)
                        s.lit = false;
                } else {
                    // 未匹配：随 R 平移保持，不点亮
                    s.box = s.box.translated(delta);
                    ++s.missed;
                    s.lit = false;
                }
                // 槽位中心保持在画面内
                s.box.xmin = max(30.f, min((float)W - 30.f, s.box.xmin));
                s.box.xmax = max(30.f, min((float)W - 30.f, s.box.xmax));
                s.box.ymin = max(30.f, min((float)H - 30.f, s.box.ymin));
                s.box.ymax = max(30.f, min((float)H - 30.f, s.box.ymax));
                if (s.lit) ++lit_now;
            }
            // 槽位重锚定：未匹配的强片段（面积+点亮证据）回收漂移/坍缩的最近槽位
            for (size_t fi = 0; fi < frags.size(); ++fi) {
                if (frag_used[fi]) continue;
                const Fragment& f = frags[fi];
                if (f.area < cfg.fan_box_area_min) continue;
                float ir = 0.25f * min(f.box.width(), f.box.height());
                if (innerDensity(mask, f.box.center(), ir) < cfg.lit_inner_min) continue;
                if (radius > 10.f &&
                    !(f.dis_in > 0.5f * radius &&
                      f.dis_out < cfg.blade_dis_out * radius + 30.f)) continue;
                int near = -1; float nd = 1e9f;
                float adopt_gate = max(0.45f * radius, 120.f);
                for (int si = 0; si < 5; ++si) {
                    if (!slots[si].has) continue;
                    float d = (float)norm(slots[si].box.center() - f.box.center());
                    if (d < adopt_gate && d < nd) { nd = d; near = si; }
                }
                if (near >= 0) {
                    slots[near].box = f.box;
                    slots[near].missed = 0;
                    slots[near].lit = true;
                    slots[near].inner_ema = 0.5f * slots[near].inner_ema +
                                            0.5f * innerDensity(mask, f.box.center(), ir);
                    frag_used[fi] = true;
                    matched[near] = true;
                    if (near == sel_slot) { sel_lost = 0; t_status = "detected"; }
                }
            }
            lit_cnt_prev = lit_now;

            // 目标选择与身份保持：有效目标 = 点亮槽位；
            // 已选槽位保持身份，直到它连续 dark_switch 帧不点亮才切换
            int best = -1; float bs = -1.f;
            for (int si = 0; si < 5; ++si) {
                Slot& s = slots[si];
                if (!s.has || !s.lit) continue;
                if (si == sel_slot) { best = si; break; }
                if (s.inner_ema > bs) { bs = s.inner_ema; if (best < 0) best = si; }
            }
            // 身份保持：已选槽位点亮 → 锁定；不点亮时计数，连续 dark_switch 帧
            // （0.5s）才允许切换到其他点亮槽位——防止 inner 短暂跌破造成乒乓切换
            if (sel_slot >= 0 && slots[sel_slot].has && slots[sel_slot].lit) {
                best = sel_slot;
                sel_dark = 0;
            } else if (sel_slot >= 0) {
                ++sel_dark;
                if (sel_dark < cfg.dark_switch) best = sel_slot;   // 宽限期保留身份
            }
            if (best < 0) {
                // 已选失效达到宽限：在其他点亮槽位中选 inner 最高者
                for (int si = 0; si < 5; ++si) {
                    Slot& s = slots[si];
                    if (!s.has || !s.lit) continue;
                    if (best < 0 || s.inner_ema > slots[best].inner_ema) best = si;
                }
            }
            if (best >= 0) {
                if (sel_slot != best) {
                    ++lock_selections;
                    events.push_back("frame " + to_string(frame_idx) + ": target -> slot " +
                                     to_string(best) + " (new ID " + to_string(id_counter + 1) + ")");
                    sel_slot = best;
                    t_id = ++id_counter;
                }
                if (slots[best].lit) {
                    sel_lost = 0;
                    sel_dark = 0;
                    t_lost = 0;          // 有实测点亮证据：连续丢失计数清零
                    t_status = "detected";
                    ++n_observed;
                } else {
                    // 宽限期内槽位无实测点亮证据：诚实标 LOST（身份仍保留）
                    ++sel_lost; ++t_lost;
                    t_status = (sel_lost >= cfg.lost_confirm) ? "lost" : "detected";
                }
                t_show = slots[best].box.center();
                t_ring_show = 0.5f * min(slots[best].box.width(), slots[best].box.height());
            } else {
                ++sel_lost;
                ++t_lost;
                t_status = (sel_lost >= cfg.lost_confirm) ? "lost" : "detected";
                if (sel_slot >= 0 && slots[sel_slot].has)
                    t_show = slots[sel_slot].box.center();
                if (sel_lost > cfg.lost_tolerance) {
                    events.push_back("frame " + to_string(frame_idx) + ": target released after " +
                                     to_string(sel_lost) + " lost frames");
                    sel_slot = -1; t_status = "searching"; sel_lost = 0; t_lost = 0;
                }
            }
        } else if (phase == Phase::TRACK && !r_live) {
            // R 非实测（estimated/丢失）：槽位整体冻结——不关联、不更新、不重算 lit，
            // 身份保持；blind > lost_tolerance 才释放重初始化（v6.3 冻结原则）
            ++t_lost;
            t_status = (t_lost >= cfg.lost_confirm) ? "lost" : "detected";
            if (sel_slot >= 0 && slots[sel_slot].has) t_show = slots[sel_slot].box.center();
            if (t_lost > cfg.lost_tolerance) {
                events.push_back("frame " + to_string(frame_idx) + ": R lost too long, re-init");
                phase = Phase::INIT_R;
                r_valid = false;
                sel_slot = -1; t_status = "searching"; t_lost = 0;
                for (auto& s : slots) s.has = false;
            }
        }
        if (t_status == "detected") ++n_detected;
        else if (sel_slot >= 0 && t_status == "lost") ++n_lost;
        int n_lit = 0;
        for (auto& s : slots) if (s.has && s.lit) ++n_lit;
        if (n_lit > 0) ++n_lit_frames;
        bool has_second = false;
        if (sel_slot >= 0)
            for (int si = 0; si < 5; ++si) {
                if (si == sel_slot || !slots[si].has || !slots[si].lit) continue;
                // 第二目标也需箭头串支撑（基地灯的同心圆会偶然通过点亮判据）
                if (!stripSupports(strips, slots[si].box.center(),
                                   0.5f * min(slots[si].box.width(), slots[si].box.height()), cfg))
                    continue;
                has_second = true;
            }
        if (has_second) ++n_dual;

        float theta_deg = 0;
        bool theta_ok = false;
        if (r_ok && sel_slot >= 0 && t_status == "detected") {
            Point2f tp = slots[sel_slot].box.center();
            theta_deg = atan2(R.y - tp.y, tp.x - R.x) * 180.f / (float)CV_PI;
            if (theta_deg < 0) theta_deg += 360.f;
            theta_ok = true;
        }

        if (debug) {
            cout << "f" << frame_idx << " phase="
                 << (phase == Phase::INIT_R ? "INIT_R" : phase == Phase::INIT_FAN ? "INIT_FAN" : "TRACK")
                 << " R[" << r_status << "](" << fmtf(R.x, 0) << "," << fmtf(R.y, 0) << ")"
                 << " frags=" << frags.size() << " lit=" << n_lit
                 << " " << t_status << " id=" << t_id;
            for (auto& s : slots)
                if (s.has)
                    cout << " #" << s.id << "(" << fmtf(s.box.center().x, 0) << ","
                         << fmtf(s.box.center().y, 0) << ";in" << fmtf(s.inner_ema, 2)
                         << (s.lit ? "L" : "") << "m" << s.missed << ")";
            cout << endl;
        }

        // ================= 5. 可视化 =================
        Mat annotated = frame.clone();
        Mat binary;
        cvtColor(mask_d, binary, COLOR_GRAY2BGR);
        if (r_ok && phase == Phase::TRACK && radius > 10.f) {
            circle(binary, R, (int)(radius * cfg.inside_rate), Scalar(0, 140, 0), 1, LINE_AA);
            circle(binary, R, (int)(radius * cfg.outside_rate), Scalar(0, 140, 0), 1, LINE_AA);
        }
        for (auto& s : slots) {                     // 槽位框（绿=点亮，灰=未点亮）
            if (!s.has) continue;
            Scalar c = s.lit ? Scalar(0, 255, 0) : Scalar(110, 110, 110);
            rectangle(binary, Point((int)s.box.xmin, (int)s.box.ymin),
                      Point((int)s.box.xmax, (int)s.box.ymax), c, 1, LINE_AA);
        }
        if (r_ok) {
            Scalar rc = (r_status == "detected") ? Scalar(0, 255, 0) : Scalar(0, 180, 255);
            circle(annotated, R, 6, rc, FILLED, LINE_AA);
            circle(annotated, R, 12, rc, 2, LINE_AA);
            drawLabel(annotated, "R (" + r_status + ")", Point((int)R.x + 14, (int)R.y - 10), rc);
            circle(binary, R, 12, rc, 2, LINE_AA);
        } else {
            drawLabel(annotated, "R searching", Point(20, 70), Scalar(0, 165, 255), 0.6, 1);
        }
        if (has_second) {
            for (int si = 0; si < 5; ++si) {
                if (si == sel_slot || !slots[si].has || !slots[si].lit ||
                    !stripSupports(strips, slots[si].box.center(),
                                   0.5f * min(slots[si].box.width(), slots[si].box.height()), cfg))
                    continue;
                Point2f p = slots[si].box.center();
                float tr = 0.5f * min(slots[si].box.width(), slots[si].box.height());
                circle(annotated, p, (int)max(8.f, tr), Scalar(255, 200, 0), 2, LINE_AA);
                drawMarker(annotated, p, Scalar(255, 200, 0), MARKER_TILTED_CROSS, 10, 1);
                drawLabel(annotated, "target 2", Point((int)p.x + 10, (int)p.y - 10),
                          Scalar(255, 200, 0), 0.5, 1);
            }
        }
        if (sel_slot >= 0) {
            Scalar tc = (t_status == "detected") ? Scalar(0, 255, 255) : Scalar(0, 0, 255);
            float tr = (t_ring_show >= 6.f) ? t_ring_show : 20.f;
            circle(annotated, t_show, (int)tr, tc, 2, LINE_AA);
            if (t_status == "detected") {
                drawMarker(annotated, t_show, tc, MARKER_CROSS, 14, 2);
                if (r_ok) line(annotated, R, t_show, Scalar(255, 0, 255), 2, LINE_AA);
                string info = "ID " + to_string(t_id) + " detected";
                if (theta_ok) info += " th=" + fmtf(theta_deg);
                drawLabel(annotated, info,
                          Point((int)t_show.x + (int)tr + 8, (int)t_show.y - (int)tr), tc, 0.6, 1);
            } else {
                drawLabel(annotated, "ID " + to_string(t_id) + " LOST " + to_string(t_lost),
                          Point((int)t_show.x - 60, (int)t_show.y - 32), tc, 0.6, 1);
            }
        } else {
            drawLabel(annotated, "target searching", Point(20, 95), Scalar(0, 255, 255), 0.6, 1);
        }
        drawLabel(annotated, "frame " + to_string(frame_idx) + "  lit=" + to_string(n_lit),
                  Point(20, 28), Scalar(255, 255, 255), 0.6, 1);
        w_overlay.write(annotated);
        w_binary.write(binary);

        csv << frame_idx << "," << r_status << ","
            << (r_ok ? fmtf(R.x) : "") << "," << (r_ok ? fmtf(R.y) : "") << ","
            << t_status << "," << (sel_slot >= 0 ? to_string(t_id) : "") << ","
            << (t_show.x > 0 ? fmtf(t_show.x) : "") << "," << (t_show.x > 0 ? fmtf(t_show.y) : "") << ","
            << (theta_ok ? fmtf(theta_deg) : "") << "," << slots.size() << "," << n_lit
            << "," << strips.size() << "\n";

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
    cout << "target observed(matched) " << n_observed << " ("
         << 100.0 * n_observed / max(total, 1) << "%)" << endl;
    cout << "target displayed detected " << n_detected << " ("
         << 100.0 * n_detected / max(total, 1) << "%), lost " << n_lost << " ("
         << 100.0 * n_lost / max(total, 1) << "%)" << endl;
    cout << "frames with lit slot: " << n_lit_frames << " ("
         << 100.0 * n_lit_frames / max(total, 1) << "%)" << endl;
    cout << "frames with second target: " << n_dual << " ("
         << 100.0 * n_dual / max(total, 1) << "%)" << endl;
    cout << "R detected " << r_detected << " (" << 100.0 * r_detected / max(total, 1) << "%)"
         << ", estimated " << r_estimated << " (" << 100.0 * r_estimated / max(total, 1) << "%)" << endl;
    cout << "lock selections = " << lock_selections << endl;
    cout << "events:" << endl;
    for (auto& e : events) cout << "  " << e << endl;

    ofstream summary((fs::path(out_dir) / "summary.txt").string());
    summary << "frames=" << total << "\nfps=" << fps << "\nsize=" << W << "x" << H << "\n";
    summary << "target_detected=" << n_detected << "\ntarget_lost=" << n_lost
            << "\ntarget_observed=" << n_observed << "\n";
    summary << "lit_frames=" << n_lit_frames << "\ndual_frames=" << n_dual << "\n";
    summary << "r_detected=" << r_detected << "\nr_estimated=" << r_estimated << "\n";
    summary << "lock_selections=" << lock_selections << "\n";
    for (auto& e : events) summary << "event: " << e << "\n";

    csv.close(); summary.close();
    w_overlay.release(); w_binary.release();
    cap.release();
    if (show) destroyAllWindows();
    return 0;
}
