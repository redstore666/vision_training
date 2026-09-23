#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    string video_path = "../resources/task_3.mp4";
    if (argc > 1) video_path = argv[1];

    VideoCapture cap(video_path);
    if (!cap.isOpened()) { cerr << "Cannot open video!" << endl; return 1; }

    int W = cap.get(CAP_PROP_FRAME_WIDTH);
    int H = cap.get(CAP_PROP_FRAME_HEIGHT);
    Point2f img_center(W / 2.0, H / 2.0);
    cout << "Video: " << W << "x" << H << endl;

    string debug_dir = "../result/task3_windmill/debug/";
    fs::create_directories(debug_dir);
    ofstream log_file(debug_dir + "debug_log.txt");
    if (!log_file.is_open()) { cerr << "Warning: cannot open log file!" << endl; }

    Mat frame, hsv, mask;
    int frame_idx = 0;
    bool auto_save = false;

    // ============ R标追踪器 ============
    Point2f r_center(-1, -1);       // 当前的 R 标位置
    int r_lost_frames = 0;          // R 标连续丢失帧数
    const int R_MAX_LOST = 30;      // 连续丢失超过 30 帧，才认为 R 标彻底丢失

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frame_idx++;

        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, Scalar(0, 100, 80), Scalar(25, 255, 255), mask);
        morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));

        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // ============ 1. 寻找 R 标：在画面中心附近 ============
        Point2f r_candidate(-1, -1);
        double r_best_dist = 1e9;
        for (auto& c : contours) {
            double area = contourArea(c);
            if (area < 100 || area > 1500) continue;
            Moments M = moments(c);
            if (M.m00 == 0) continue;
            Point2f center(M.m10 / M.m00, M.m01 / M.m00);
            double d = norm(center - img_center);
            // R 标一定在画面中心附近（距离 < 200 像素）
            if (d < 200 && d < r_best_dist) {
                r_best_dist = d;
                r_candidate = center;
            }
        }

        // 更新 R 标追踪器
        if (r_candidate.x > 0) {
            r_center = r_candidate;
            r_lost_frames = 0;
        } else {
            r_lost_frames++;
            if (r_lost_frames > R_MAX_LOST) {
                r_center = Point2f(-1, -1); // 超过 30 帧，放弃追踪
            }
            // 否则沿用上一帧的 r_center，保证不丢失
        }

        // ============ 2. 寻找靶标：在 R 标周围的环形轨道上 ============
        vector<pair<Point2f, float>> targets;
        for (auto& c : contours) {
            double area = contourArea(c);
            if (area < 150 || area > 1500) continue;

            Point2f circle_center;
            float circle_radius;
            minEnclosingCircle(c, circle_center, circle_radius);

            // 靶标半径约 10~25 像素
            if (circle_radius < 8 || circle_radius > 25) continue;

            // 圆形度（越接近 1 越圆）
            double perimeter = arcLength(c, true);
            double circularity = 4 * M_PI * area / (perimeter * perimeter + 1e-6);
            if (circularity < 0.4) continue;

            // 填充率（环形约 0.3~0.8）
            double fill_ratio = area / (M_PI * circle_radius * circle_radius + 1e-6);
            if (fill_ratio < 0.3 || fill_ratio > 0.85) continue;

            // 如果 R 标找到了，只保留距离 R 标 150~400 像素的靶标
            if (r_center.x > 0) {
                double d_to_r = norm(circle_center - r_center);
                if (d_to_r < 150 || d_to_r > 400) continue;
            } else {
                // 如果 R 标丢了，暂时不检测靶标，避免误检
                continue;
            }

            targets.push_back({circle_center, circle_radius});
        }

        // ============ 3. 靶标去重：同一个靶标可能被检测出内外两个圆 ============
        vector<pair<Point2f, float>> unique_targets;
        for (auto& t : targets) {
            bool is_duplicate = false;
            for (auto& u : unique_targets) {
                if (norm(t.first - u.first) < 30) { // 圆心距离 < 30 像素认为是同一个
                    is_duplicate = true;
                    break;
                }
            }
            if (!is_duplicate) unique_targets.push_back(t);
        }
        targets = unique_targets;

        // ============ 4. 可视化 ============
        Mat annotated = frame.clone();

        if (r_center.x > 0) {
            circle(annotated, r_center, 8, Scalar(0, 255, 0), 2);
            putText(annotated, "R", Point(r_center.x + 12, r_center.y),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        } else {
            putText(annotated, "R LOST", Point(W/2 - 50, 50),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
        }

        for (auto& t : targets) {
            circle(annotated, t.first, t.second, Scalar(0, 255, 255), 2);
            circle(annotated, t.first, 3, Scalar(0, 0, 255), -1);
            if (r_center.x > 0) {
                line(annotated, r_center, t.first, Scalar(255, 0, 255), 2);
            }
        }

        drawMarker(annotated, img_center, Scalar(255, 0, 0), MARKER_CROSS, 30, 1);

        // ============ 5. 写日志 ============
        if (log_file.is_open()) {
            log_file << "Frame " << frame_idx
                     << " | R=(" << r_center.x << "," << r_center.y << ")"
                     << " | targets=" << targets.size();
            for (auto& t : targets) {
                log_file << " [(" << t.first.x << "," << t.first.y
                         << ") r=" << t.second << "]";
            }
            log_file << endl;
        }

        // ============ 6. 显示与按键处理 ============
        imshow("Original", annotated);
        imshow("Mask", mask);

        int key = waitKey(30);
        if (key == 'q' || key == 27) break;

        // 处理暂停时的按键
        if (key == ' ') {
            cout << "--- Paused at frame " << frame_idx << " ---" << endl;
            while (true) {
                int pause_key = waitKey(0);
                if (pause_key == 's' || pause_key == 'S') {
                    string base = debug_dir + "manual_" + to_string(frame_idx);
                    imwrite(base + "_annotated.png", annotated);
                    imwrite(base + "_mask.png", mask);
                    cout << "Manual saved: " << base << "_annotated.png" << endl;
                } else if (pause_key == ' ' || pause_key == 'q' || pause_key == 27) {
                    break; // 继续播放或退出
                }
            }
        }

        // 正常播放时按 's' 保存
        if (key == 's' || key == 'S') {
            string base = debug_dir + "manual_" + to_string(frame_idx);
            imwrite(base + "_annotated.png", annotated);
            imwrite(base + "_mask.png", mask);
            cout << "Manual saved: " << base << "_annotated.png" << endl;
        }

        if (key == 'a' || key == 'A') {
            auto_save = !auto_save;
            cout << "Auto-save mode: " << (auto_save ? "ON" : "OFF") << endl;
        }

        if (auto_save && (frame_idx % 100 == 0)) {
            string fname = debug_dir + "auto_" + to_string(frame_idx) + ".png";
            imwrite(fname, annotated);
            cout << "Auto-saved: " << fname << endl;
        }
    }

    if (log_file.is_open()) log_file.close();
    cap.release();
    destroyAllWindows();
    return 0;
}