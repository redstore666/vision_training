# Vision Training — OpenCV / Eigen / Ceres 综合实践

本工程为《第二次培训：OpenCV C++ 图像处理基础》配套作业，包含三个任务：
- **任务1**：郁金香图片 OpenCV 图像处理
- **任务2**：合成旋转视频的参数拟合
- **任务3**：真实能量机关视频的识别与稳定跟踪

三个任务统一使用 C++、OpenCV、Eigen 与 Ceres，采用 CMake 构建。

---

## 1. 环境依赖

| 依赖 | 版本要求 | 安装方式（Ubuntu 22.04） |
|:---|:---|:---|
| CMake | ≥ 3.16 | `sudo apt install cmake` |
| GCC / G++ | ≥ 9（支持 C++17） | `sudo apt install build-essential` |
| OpenCV | ≥ 4.5 | `sudo apt install libopencv-dev` |
| Eigen3 | ≥ 3.3 | `sudo apt install libeigen3-dev` |
| Ceres Solver | ≥ 2.0 | `sudo apt install libceres-dev` |

> 验证 OpenCV：`pkg-config --modversion opencv4`  
> 验证 Eigen：`ls /usr/include/eigen3`  
> 验证 Ceres：`pkg-config --modversion ceres`（若可用）

---

## 2. 构建与运行

```bash
# 在工程根目录下执行
mkdir -p build
cd build
cmake ..
make

# 运行任务1
./task1_image

# 运行任务2
./task2_fit

# 运行任务3（默认处理 task_3.mp4）
./task3_windmill
# 或指定视频
./task3_windmill ../resources/task_4.mp4
```
---

## 3.目录结构
vision_training/
├── CMakeLists.txt
├── README.md
├── include/                    # 公共头文件（预留）
├── src/
│   ├── common/                 # 公共源码（预留）
│   ├── task1_image/
│   │   └── main.cpp            # 任务1：图片处理
│   ├── task2_fit/
│   │   ├── main.cpp            # 任务2：视频拟合
│   │   └── test_eigen.cpp      # Eigen 测试（可选）
│   └── task3_windmill/
│       └── main.cpp            # 任务3：能量机关识别与跟踪
├── config/                     # 配置目录（预留）
├── resources/
│   ├── test_image.jpg          # 任务1输入图片
│   ├── task_2.mp4              # 任务2输入视频
│   ├── task_3.mp4              # 任务3输入视频（小能量机关）
│   └── task_4.mp4              # 任务3输入视频（大能量机关）
└── result/
    ├── task1_images/           # 任务1全部输出图（16张）
    ├── task2_fit/              # 任务2输出：视频、曲线、残差、CSV
    ├── task2_fit_result.md     # 任务2说明文件
    ├── task3_windmill/         # 任务3输出（按视频分目录）
    │   ├── task_3/
    │   └── task_4/
    └── task3_tracking_result.md # 任务3说明文件

---

## 4. 输入输出路径
任务	输入	输出
任务1	resources/test_image.jpg	result/task1_images/*.png（16张）
任务2	resources/task_2.mp4	result/task2_fit/ 下视频、图片、CSV、task2_fit_result.md
任务3	resources/task_3.mp4、task_4.mp4	result/task3_windmill/task_3/、task_4/ 下识别视频、二值化视频、track_log.csv、summary.txt，说明见 result/task3_tracking_result.md

---

## 5. 关键参数
5.1 任务1：图片处理
HSV 红色双区间：H ∈ [0,10] ∪ [170,179]，S ≥ 100，V ≥ 100

滤波核尺寸：均值 Size(15,15)；高斯 Size(5,5), σ=1.5；中值核 5

形态学核：Size(3,3) 或 Size(5,5)，矩形结构元素

轮廓筛选：面积 ≥ 500 像素，长宽比 ∈ [0.2, 5.0]

旋转：绕图像中心旋转 35°

裁剪：左上角 1/4（原宽、高各取一半）

5.2 任务2：视频拟合
青色 HSV 范围：H ∈ [85,95]，S ≥ 100，V ≥ 100

旋转中心：(480, 360)

角速度模型：ω(t) = b + A·sin(Ωt + φ)

拟合方法：Ceres 自动求导 + Ω 初值网格粗扫

最终参数：A ≈ 0.55，b ≈ 1.35，Ω ≈ 1.65，φ ≈ 0.70（归一化后）

5.3 任务3：识别与跟踪
橙红色 HSV 范围（实测标定）：H ∈ [0,17]，S ≥ 55，V ≥ 80

目标簇：核心块半径 ≥ 8px，48px 链式聚类；外接半径 15~95px，填充率 ≤ 0.55，长宽比 ≤ 4，面积 ≤ 3200

点亮判别 inner_density（0.5R 内侧掩膜占比）：进入 ≥ 0.13，退出 < 0.09（滞回；点亮双环 ≈0.24，未点亮单环 ≈0.07）

R 标字形签名：半径 11~14px，面积 215~325 像素，圆形度 0.33~0.62，填充率 0.42~0.75，且在同尺寸块中孤立

R 跟踪：链式累积命中（30 帧窗口 ≥ 9 次），不可见时按场景漂移外推，>90 帧无命中标记 lost

丢失容忍：45 帧；重选条件：当前目标熄灭 15 帧（0.5s）且有其他点亮稳定目标，或持续丢失超过 45 帧

同时双目标（task_4）：另一目标只作候选标注（青色 ID），不夺走已选身份；同扇叶重获时继承原 ID

---

## 6. 任务1 分析
任务1使用 test_image.jpg 完成以下 OpenCV 基础操作，输出 16 张结果图：

灰度化：cvtColor(img, gray, COLOR_BGR2GRAY)

HSV 单通道：split 分离 H、S、V

三种滤波：均值、高斯、中值，核尺寸与 σ 见参数表

红色提取：HSV 双区间 inRange + bitwise_or 合并

形态学：腐蚀、膨胀、开运算、闭运算

轮廓与筛选：findContours + contourArea + boundingRect + 长宽比过滤

绘制与变换：circle、rectangle、putText，旋转 35°，裁剪左上角 1/4

分析结论：

均值滤波对边缘的模糊最明显；高斯滤波在相同核尺寸下保留细节更好；中值滤波对椒盐噪声最有效。

红色提取能覆盖大部分花瓣区域，但在强反光/阴影处会出现空洞，闭运算可有效填补。

轮廓筛选后保留了主要郁金香区域，面积与长宽比约束能有效去除噪点。

---

## 7. 全部结果索引
7.1 任务1 结果图（result/task1_images/）
文件名	说明
gray.png	灰度图
hsv_h.png / hsv_s.png / hsv_v.png	HSV 三通道
mean_filter.png / gaussian_filter.png / median_filter.png	三种滤波结果
red_mask.png	红色掩膜
erode.png / dilate.png / open.png / close.png	形态学操作
contours_boxes.png	轮廓筛选与外接矩形
drawing.png	绘制圆、矩形、文字
rotated_35deg.png	旋转 35°
crop_top_left.png	裁剪左上角 1/4
7.2 任务2 结果（result/task2_fit/）
文件名	说明
tracking_overlay.mp4	带识别标记的视频
fit_comparison.png	观测点与拟合曲线对比
angular_velocity.png	角速度曲线
residuals.png	残差图
observations.csv	原始观测数据（t, theta, omega）
../task2_fit_result.md	任务2参数、方法与误差指标说明
7.3 任务3 结果（result/task3_windmill/）
路径	说明
task_3/recognition_overlay.mp4	小能量机关识别视频（796 帧，30 FPS）
task_4/recognition_overlay.mp4	大能量机关识别视频（1800 帧，30 FPS）
task_3/binary_process.mp4（建议项）	二值化过程
task_4/binary_process.mp4（建议项）	二值化过程
task_3/track_log.csv、task_4/track_log.csv	逐帧跟踪结果（R 状态/位置、目标状态/ID/角度、点亮数）
task_3/summary.txt、task_4/summary.txt	运行统计与锁定/重选事件日志
../task3_tracking_result.md	任务3检测、锁定、丢失与重选规则说明
