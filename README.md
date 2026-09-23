# 任务1：OpenCV 图片处理
滤波参数：“均值滤波核大小 15x15，高斯滤波核大小 5x5、sigma=1.5，中值滤波核大小 5”

红色提取参数：“HSV 范围 H: 0-10 与 170-179，S: 100-255，V: 100-255。合并两段掩膜得到完整红色区域”

形态学与轮廓：“使用 3x3 核进行闭运算缝合断裂边缘，筛选面积大于 500 像素、长宽比在 0.2 到 5.0 之间的轮廓”

结果分析：“由于光照不均，红色花瓣反光处出现空洞，通过闭运算有效填补。阴影处的红色饱和度低，被部分滤除，符合预期”
# 任务2：合成旋转视频的参数拟合结果

## 1. 模型与参数
角速度模型：ω(t) = b + A·sin(Ωt + φ)

通过 Ceres 非线性最小二乘拟合，得到以下参数（时间原点为视频第0帧）：
- 振幅 A = 0.5499 rad/s
- 平均角速度 b = 1.3500 rad/s
- 频率参数 Ω = 1.6498 rad/s
- 相位 φ = 0.7026 rad（已归一化到 [-π, π)）

速度变化周期 T = 2π / Ω ≈ 3.81 秒。

## 2. 方法与流程
1. 使用 OpenCV 逐帧读取 task_2.mp4，通过 HSV 颜色空间提取青色目标；
2. 计算每帧目标的质心坐标，使用 atan2 计算相对旋转中心 (480, 360) 的角度；
3. 对角度进行展开（unwrap），消除回绕跳变；
4. 计算角速度 ω = Δθ/Δt；
5. 使用 Ceres Solver 对 ω(t) = b + A·sin(Ωt + φ) 进行自动求导非线性拟合。

## 3. 误差指标
- 有效样本数：1439 个
- 参与计算的帧范围：第 0 帧 到 第 1439 帧
- Ceres 最终代价：2.9769
- 残差平方和 (SSE)：5.9537
- 均方根误差 (RMSE)：√(SSE / N) = √(5.9537 / 1439) ≈ 0.0643 rad/s

## 4. 结果分析
从 fit_comparison.png 和 residuals.png 可以看出，观测点均匀分布在拟合曲线两侧，残差无明显系统偏差。
说明模型选择合理，Ceres 求解收敛良好。

# 任务3：真实能量机关视频的识别与稳定跟踪

详细方法、锁定/丢失/重选规则与已知失败情况见 [result/task3_tracking_result.md](result/task3_tracking_result.md)。

## 场景对应关系与视频参数
| 素材 | 场景 | 分辨率 | 帧率 | 帧数 |
|---|---|---|---|---|
| task_3.mp4 | 小能量机关（同时最多 1 个目标） | 1440×1080 | 30 FPS | 796 |
| task_4.mp4 | 大能量机关（同时最多 2 个目标） | 1440×1080 | 30 FPS | 1800 |

## 关键参数
- 掩膜：`H∈[0,17], S≥55, V≥80`（能量机关发光为深橙红；基地灯偏黄、横幅偏绿，均在掩膜外）
- 点亮判别：`inner_density = 0.5R 内侧掩膜占比 ≥ 0.13`（点亮双环靶标 ≈0.24，未点亮单环 ≈0.07）
- 丢失容忍 45 帧；熄灭 15 帧且有其他点亮目标时重选；ID 自增分配，不使用轮廓顺序
- R 标 = 字形检测（脉冲 LED，可见率 18%~42%）+ 场景漂移外推，状态分 detected / estimated

## 结果概览
| 指标 | task_3 | task_4 |
|---|---|---|
| 目标 detected 帧 | 436（54.8%） | 959（53.3%） |
| R 标覆盖（检出+外推） | 311（39.1%） | 1731（96.2%） |
| 使用的 ID 数 | 7 | 20 |

## 构建与运行（三个任务）
```bash
cd vision_training
cmake -S . -B build
cmake --build build -j4
cd build
./task1_image        # 任务1：图片处理 → ../result/task1_images/
./task2_fit          # 任务2：合成视频拟合 → ../result/task2_fit/
./task3_windmill ../resources/task_3.mp4   # 任务3 → ../result/task3_windmill/task_3/
./task3_windmill ../resources/task_4.mp4   # 任务3 → ../result/task3_windmill/task_4/
```
任务3 可选参数：`--show` 实时显示，`--debug` 打印逐帧跟踪状态。

## 结果索引
- 任务1：`result/task1_images/`（16 张处理结果图）
- 任务2：`result/task2_fit/`（tracking_overlay.mp4、fit_comparison.png、angular_velocity.png、residuals.png、task2_fit_result.md）
- 任务3：`result/task3_windmill/task_3/`、`result/task3_windmill/task_4/`（recognition_overlay.mp4、binary_process.mp4、track_log.csv、summary.txt）、`result/task3_tracking_result.md`