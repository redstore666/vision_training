# 任务2：合成旋转视频的参数拟合结果

## 1. 任务简介
处理 `resources/task_2.mp4`（960×720, 60FPS, 24秒, 1440帧）。
相机固定，白色点为旋转中心，青色圆点为跟踪目标。
已知旋转中心为 (480, 360)，半径为 220 像素。

## 2. 角速度模型
已知角速度遵循：

    ω(t) = b + A·sin(Ωt + φ)

待估计的四个参数：振幅 A、平均角速度 b、频率参数 Ω、相位 φ。
约束：A > 0，b > A，Ω > 0。
时间单位为秒，角度单位为弧度。
相位 φ 报告时统一到 [-π, π)。

## 3. 实现流程

### 3.1 目标提取（OpenCV）
1. 逐帧读取视频，转换到 HSV 颜色空间；
2. 使用 inRange 提取青色目标（H: 85~95, S: 100~255, V: 100~255）；
3. 使用 findContours 找轮廓，面积过滤（<50 丢弃）；
4. 使用 moments 计算质心 (cx, cy)。

### 3.2 角度计算
角度定义：从右方起算、逆时针为正：

    θ_i = atan2(c_y - y_i, x_i - c_x)

其中 (c_x, c_y) = (480, 360)。

### 3.3 角度展开（Unwrap）
因为 atan2 输出在 [-π, π]，目标转过一圈后会跳变。
通过相邻帧角度差判断，若差值超出 ±π 则补偿 2π：

    delta = θ[i] - θ[i-1]
    if (delta > π)  delta -= 2π
    if (delta < -π) delta += 2π
    θ_unwrapped[i] = θ_unwrapped[i-1] + delta

### 3.4 角速度计算
角速度 = 相邻两帧角度差 / 时间间隔：

    ω_i = (θ[i+1] - θ[i]) / (1/60)

### 3.5 Ceres 非线性最小二乘拟合
使用 Ceres Solver 自动求导拟合。残差定义为：

    residual = ω_observed - (b + A·sin(Ω·t + φ))

**关键：Ω 的初值粗扫（Grid Search）**
因为 Ceres 对 Ω 的初值极其敏感，先对 Ω 在 0.1~3.0 范围内以步长 0.1 粗扫，
找到使残差平方和最小的 Ω 作为 Ceres 的初值。

## 4. 拟合结果

| 参数 | 估计值 | 单位 |
|:---|:---|:---|
| A（振幅） | 0.5499 | rad/s |
| b（平均角速度） | 1.3500 | rad/s |
| Ω（频率参数） | 1.6498 | rad/s |
| φ（相位，已归一化） | 0.7026 | rad |

速度变化周期：T = 2π/Ω ≈ 3.81 秒。

## 5. 误差指标
- 有效样本数：1439
- 参与计算帧范围：第 0 帧 ~ 第 1439 帧
- Ceres 最终代价：2.9769
- 残差平方和 (SSE)：5.9537
- 均方根误差 (RMSE)：√(SSE/N) ≈ 0.0643 rad/s

## 6. 结果分析
从 `fit_comparison.png` 和 `residuals.png` 可以看出：
观测点均匀分布在拟合曲线两侧，残差无明显系统偏差，
说明模型选择合理，Ceres 求解收敛良好。

## 7. 输出文件
- `result/task2_fit/tracking_overlay.mp4`：标注视频
- `result/task2_fit/fit_comparison.png`：观测点与拟合曲线对比
- `result/task2_fit/angular_velocity.png`：角速度曲线
- `result/task2_fit/residuals.png`：残差图
- `result/task2_fit/observations.csv`：原始观测数据