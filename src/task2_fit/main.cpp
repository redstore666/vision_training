#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream> 
#include <ceres/ceres.h>

using namespace cv;
using namespace std;

// 定义残差计算器(main上面)
    struct SineResidual {
        SineResidual(double t, double omega) : t_(t), omega_(omega) {}
        template <typename T>
        bool operator()(const T* const params, T* residual) const {
            T prediction = params[1] + params[0] * ceres::sin(params[2] * T(t_) + params[3]);
            residual[0] = T(omega_) - prediction;
            return true;
        }
    private:
        const double t_;
        const double omega_;
    };

int main() {
    // 1. 打开视频
    VideoCapture cap("../resources/task_2.mp4"); 
    if (!cap.isOpened()) {
        cerr << "Cannot open video!" << endl;
        return 1;
    }

    // 视频参数：960x720, 60FPS
    double fps = cap.get(CAP_PROP_FPS);
    int total_frames = cap.get(CAP_PROP_FRAME_COUNT);
    double dt = 1.0 / fps;  // 每帧时间间隔（秒）
    cout << "FPS: " << fps << ", Total Frames: " << total_frames << endl;

    // 2. 循环读取并处理
    Mat frame, hsv, mask;
    // 用 vector 存储每帧的数据（因为帧数不确定，用动态数组）
    vector<double> time_data;   // 时间 t
    vector<double> theta_data;  // 原始的 wrapped 角度

    for (int i = 0; i < total_frames; i++) {
        cap >> frame;
        if (frame.empty()) break;

        // 3. 提取青色目标（HSV范围参考：H约85-95, S>100, V>100）
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, Scalar(85, 100, 100), Scalar(95, 255, 255), mask);

        // 4. 找轮廓，计算中心
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        double best_area = 0;
        double cx = -1, cy = -1;

        for (auto& c : contours) {
            double area = contourArea(c);
            if (area < 50) continue; 
            // 选最大的轮廓（避免误检测到多个小青点）
            if (area > best_area) {
                best_area = area;
                Moments M = moments(c);
                if (M.m00 > 0) {
                    cx = M.m10 / M.m00;
                    cy = M.m01 / M.m00;
                }
            }
        }

        // 如果本帧检测到了有效目标，保存数据
        if (cx > 0) {
            double center_x = 480.0, center_y = 360.0;
            double theta = atan2(center_y - cy, cx - center_x);
            time_data.push_back(i * dt);
            theta_data.push_back(theta);
        } else {
            // 漏检！这是一个非常重要的问题（见下面说明）
            cout << "Frame " << i << ": No target detected!" << endl;
        }


    }

    cout << "Valid samples: " << time_data.size() << endl;

    // ================= 3. 角度展开 =================
    vector<double> theta_unwrapped(theta_data.size());
    theta_unwrapped[0] = theta_data[0];
    for (size_t i = 1; i < theta_data.size(); i++) {
        double delta = theta_data[i] - theta_data[i-1];
        // 处理回绕：如果相邻帧的差值超过 pi，说明发生了跳变
        if (delta > M_PI)  delta -= 2 * M_PI;
        if (delta < -M_PI) delta += 2 * M_PI;
        theta_unwrapped[i] = theta_unwrapped[i-1] + delta;
    }

    // ================= 4. 计算角速度 =================
    // omega[i] = (theta[i+1] - theta[i]) / dt
    vector<double> omega_data;
    vector<double> omega_time;
    for (size_t i = 0; i + 1 < theta_unwrapped.size(); i++) {
        double omega = (theta_unwrapped[i+1] - theta_unwrapped[i]) / dt;
        omega_data.push_back(omega);
        // 时间取两帧的中点
        omega_time.push_back((time_data[i] + time_data[i+1]) / 2.0);
    }

    cout << "Omega samples: " << omega_data.size() << endl;

    // ================= 5. 保存数据到文件（方便调试和后续拟合）=================
    ofstream fout("../result/task2_fit/observations.csv");
    fout << "t,theta,omega\n";
    for (size_t i = 0; i < theta_unwrapped.size(); i++) {
        fout << time_data[i] << "," << theta_unwrapped[i];
        if (i < omega_data.size()) fout << "," << omega_data[i];
        fout << "\n";
    }
    fout.close();
    cout << "Saved to result/task2_fit/observations.csv" << endl;

    // ================= 6. 打印前 10 个角速度看看 =================
    for (size_t i = 0; i < 10 && i < omega_data.size(); i++) {
        cout << "t=" << omega_time[i] << ", omega=" << omega_data[i] << endl;
    }

    // ================= 7. 使用 Ceres 进行非线性拟合 =================
    // 定义残差计算器(在main上面)

    // ========== 第一步：对 Omega 进行粗扫，找到一个好的初值 ==========
    double best_omega_init = 0.1;
    double best_cost = 1e30;
    
    // Omega 从 0.1 到 3.0，步长 0.1 粗扫（假设周期至少 2 秒）
    for (double omega_test = 0.1; omega_test <= 3.0; omega_test += 0.1) {
        double test_params[4] = {0.5, 1.5, omega_test, 0.0};
        // 手动算一遍所有残差的平方和
        double cost = 0;
        for (size_t i = 0; i < omega_data.size(); i++) {
            double pred = test_params[1] + test_params[0] * sin(test_params[2] * omega_time[i] + test_params[3]);
            double r = omega_data[i] - pred;
            cost += r * r;
        }
        if (cost < best_cost) {
            best_cost = cost;
            best_omega_init = omega_test;
        }
    }
    std::cout << "粗扫得到的最佳 Omega 初值: " << best_omega_init 
              << " (cost=" << best_cost << ")" << std::endl;

    // ========== 第二步：用这个初值，交给 Ceres 精修 ==========
    // 用粗扫的结果作为 Omega 的初值，A 和 b 也可以根据数据给个合理的初值
    double params[4] = {0.5, 1.5, best_omega_init, 0.0};

    ceres::Problem problem;
    for (size_t i = 0; i < omega_data.size(); ++i) {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<SineResidual, 1, 4>(
                new SineResidual(omega_time[i], omega_data[i]));
        problem.AddResidualBlock(cost_function, nullptr, params);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // ========== 第三步：报告结果 ==========
    std::cout << summary.BriefReport() << "\n";
    std::cout << "估计的 A     = " << params[0] << "\n";
    std::cout << "估计的 b     = " << params[1] << "\n";
    std::cout << "估计的 Omega = " << params[2] << "\n";
    // 把 phi 归一化到 [-pi, pi)
    double phi = params[3];
    while (phi >= M_PI) phi -= 2 * M_PI;
    while (phi < -M_PI) phi += 2 * M_PI;
    std::cout << "估计的 phi   = " << phi << " (已归一化)\n";



        // ================= 8. 画图：观测点 vs 拟合曲线 =================
    // 8.1 准备画布
    int W = 1200, H = 500;
    Mat canvas(H, W, CV_8UC3, Scalar(255, 255, 255)); // 白色画布
    
    // 8.2 定义坐标映射参数
    // 边距：左100，右100，上50，下50，中间是绘图区
    int margin_left = 100, margin_right = 100, margin_top = 50, margin_bottom = 50;
    int plot_w = W - margin_left - margin_right;  // 1000
    int plot_h = H - margin_top - margin_bottom;  // 400

    // 数据范围（根据你的数据调整）
    double t_min = 0.0, t_max = 24.0;
    double w_min = 0.0, w_max = 2.5;

    // 8.3 画坐标轴（黑色）
    line(canvas, Point(margin_left, margin_top), Point(margin_left, H - margin_bottom), Scalar(0,0,0), 2); // y轴
    line(canvas, Point(margin_left, H - margin_bottom), Point(W - margin_right, H - margin_bottom), Scalar(0,0,0), 2); // x轴
    putText(canvas, "t (s)", Point(W/2, H-10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0), 2);
    putText(canvas, "omega (rad/s)", Point(10, H/2), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0), 2);

    // 8.4 定义映射 lambda 表达式（这是关键！）
    // 输入物理量，返回像素坐标
    auto toPixel = [&](double t, double omega) {
        int x = margin_left + (int)((t - t_min) / (t_max - t_min) * plot_w);
        int y = H - margin_bottom - (int)((omega - w_min) / (w_max - w_min) * plot_h);
        return Point(x, y);
    };

    // 8.5 画观测点（蓝色小圆点）
    // 如果点太多，可以每隔几个点画一次，避免重叠
    for (size_t i = 0; i < omega_data.size(); i += 5) {
        Point p = toPixel(omega_time[i], omega_data[i]);
        circle(canvas, p, 3, Scalar(255, 0, 0), -1); // BGR：(255,0,0) 是蓝色
    }

    // 8.6 画拟合曲线（红色折线）
    // 用拟合参数重新计算整条曲线
    vector<Point> fit_points;
    for (double t = t_min; t <= t_max; t += 0.05) {
        double pred = params[1] + params[0] * sin(params[2] * t + params[3]);
        fit_points.push_back(toPixel(t, pred));
    }
    // 用 polylines 一次性画整条折线
    polylines(canvas, fit_points, false, Scalar(0, 0, 255), 2); // BGR：(0,0,255) 是红色

    // 8.7 标注参数信息
    string info = "A=" + to_string(params[0]) + ", b=" + to_string(params[1]) 
                + ", Omega=" + to_string(params[2]) + ", phi=" + to_string(params[3]);
    putText(canvas, info, Point(margin_left, margin_top - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,0), 1);

    // 8.8 保存
    imwrite("../result/task2_fit/fit_comparison.png", canvas);
    cout << "Saved fit_comparison.png" << endl;
    imshow("Fit Comparison", canvas);
    waitKey(0);



        // ================= 9. 画残差图 =================
    Mat canvas_res(H, W, CV_8UC3, Scalar(255, 255, 255));
    
    // 残差范围：假设在 -0.2 到 0.2 之间
    double r_min = -0.2, r_max = 0.2;
    
    // 画轴
    line(canvas_res, Point(margin_left, margin_top), Point(margin_left, H-margin_bottom), Scalar(0,0,0), 2);
    line(canvas_res, Point(margin_left, H-margin_bottom), Point(W-margin_right, H-margin_bottom), Scalar(0,0,0), 2);
    putText(canvas_res, "t (s)", Point(W/2, H-10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0), 2);
    putText(canvas_res, "Residual", Point(10, H/2), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0), 2);

    // 画 y=0 基准线（红色虚线）
    int zero_y = H - margin_bottom - (int)((0 - r_min) / (r_max - r_min) * plot_h);
    line(canvas_res, Point(margin_left, zero_y), Point(W-margin_right, zero_y), Scalar(0,0,255), 1);

    // 映射函数（只改 y 的映射）
    auto toPixelRes = [&](double t, double res) {
        int x = margin_left + (int)((t - t_min) / (t_max - t_min) * plot_w);
        int y = H - margin_bottom - (int)((res - r_min) / (r_max - r_min) * plot_h);
        return Point(x, y);
    };

    // 画残差点
    for (size_t i = 0; i < omega_data.size(); i++) {
        double pred = params[1] + params[0] * sin(params[2] * omega_time[i] + params[3]);
        double res = omega_data[i] - pred;
        circle(canvas_res, toPixelRes(omega_time[i], res), 3, Scalar(255, 0, 0), -1);
    }
    imwrite("../result/task2_fit/residuals.png", canvas_res);
    cout << "Saved residuals.png" << endl;


        // ================= 10. 画角速度曲线图 =================
    Mat canvas_omega(H, W, CV_8UC3, Scalar(255, 255, 255));
    
    // 画轴（复用之前的坐标）
    line(canvas_omega, Point(margin_left, margin_top), Point(margin_left, H-margin_bottom), Scalar(0,0,0), 2);
    line(canvas_omega, Point(margin_left, H-margin_bottom), Point(W-margin_right, H-margin_bottom), Scalar(0,0,0), 2);

    // 映射函数
    auto toPixelOmega = [&](double t, double w) {
        int x = margin_left + (int)((t - t_min) / (t_max - t_min) * plot_w);
        int y = H - margin_bottom - (int)((w - w_min) / (w_max - w_min) * plot_h);
        return Point(x, y);
    };

    // 画观测角速度的折线（蓝色）
    vector<Point> obs_line;
    for (size_t i = 0; i < omega_data.size(); i++) {
        obs_line.push_back(toPixelOmega(omega_time[i], omega_data[i]));
    }
    polylines(canvas_omega, obs_line, false, Scalar(255, 0, 0), 1);

    // 叠加拟合曲线（红色）
    vector<Point> fit_line;
    for (double t = t_min; t <= t_max; t += 0.05) {
        double pred = params[1] + params[0] * sin(params[2] * t + params[3]);
        fit_line.push_back(toPixelOmega(t, pred));
    }
    polylines(canvas_omega, fit_line, false, Scalar(0, 0, 255), 2);

    imwrite("../result/task2_fit/angular_velocity.png", canvas_omega);
    cout << "Saved angular_velocity.png" << endl;


        // ================= 11. 生成带识别标记的视频 =================
    VideoCapture cap2("../resources/task_2.mp4");
    if (!cap2.isOpened()) return 1;
    
    // 设置视频写入器，编码格式用 MJPG，帧率 60
    VideoWriter writer("../result/task2_fit/tracking_overlay.mp4",
                       VideoWriter::fourcc('m','p','4','v'),
                       60, Size(960, 720));
    if (!writer.isOpened()) {
        cerr << "Failed to open VideoWriter!" << endl;
        return 1;
    }

    Mat frame2, hsv2, mask2;
    for (int i = 0; i < total_frames; i++) {
        cap2 >> frame2;
        if (frame2.empty()) break;

        cvtColor(frame2, hsv2, COLOR_BGR2HSV);
        inRange(hsv2, Scalar(85, 100, 100), Scalar(95, 255, 255), mask2);

        vector<vector<Point>> contours2;
        findContours(mask2, contours2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        double best_area = 0, cx = -1, cy = -1;
        for (auto& c : contours2) {
            double area = contourArea(c);
            if (area < 50) continue;
            if (area > best_area) {
                best_area = area;
                Moments M = moments(c);
                if (M.m00 > 0) { cx = M.m10 / M.m00; cy = M.m01 / M.m00; }
            }
        }

        // 画检测结果
        if (cx > 0) {
            double center_x = 480.0, center_y = 360.0;
            // 画青色目标点
            circle(frame2, Point(cx, cy), 10, Scalar(255, 255, 0), -1);
            // 画旋转中心
            circle(frame2, Point(center_x, center_y), 5, Scalar(0, 0, 255), -1);
            // 画目标与中心的连线
            line(frame2, Point(center_x, center_y), Point(cx, cy), Scalar(0, 255, 0), 2);
            // 写上 "Tracking"
            putText(frame2, "Tracking", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        } else {
            putText(frame2, "Lost", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }

        writer.write(frame2);
    }
    cap2.release();
    writer.release();
    cout << "Saved tracking_overlay.mp4" << endl;


    





    return 0;
}