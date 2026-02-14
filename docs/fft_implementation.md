# FFT 图表实现文档

## 概述

本文档描述 LogCortex 中 FFT (快速傅里叶变换) 图表的实现，严格对齐 Flight Review 标准。

---

## 1. 支持的 FFT 图表

| 图表 | 数据源 Topic | 数据字段 | 用途 |
|------|-------------|----------|------|
| **Actuator Controls FFT** | `actuator_controls_0` 或 `vehicle_torque_setpoint` | Roll, Pitch, Yaw 控制量 | 分析执行器控制信号的频域特性 |
| **Angular Velocity FFT** | `vehicle_angular_velocity` | `xyz[0]`, `xyz[1]`, `xyz[2]` | 分析角速度的频域特性 |

---

## 2. 数据源选择逻辑

### 2.1 Actuator Controls FFT

```python
# 检测是否使用动态控制分配
if 'actuator_motors' in topics or 'actuator_servos' in topics:
    # 动态控制分配模式
    topic = 'vehicle_torque_setpoint'
    fields = ['xyz[0]', 'xyz[1]', 'xyz[2]']  # Roll, Pitch, Yaw
else:
    # 传统模式
    topic = 'actuator_controls_0'
    fields = ['control[0]', 'control[1]', 'control[2]']  # Roll, Pitch, Yaw
```

### 2.2 Angular Velocity FFT

```python
topic = 'vehicle_angular_velocity'
fields = ['xyz[0]', 'xyz[1]', 'xyz[2]']  # Rollspeed, Pitchspeed, Yawspeed
```

---

## 3. FFT 计算核心

### 3.1 频率轴计算 (关键)

**问题**：降采样会导致频率轴计算错误。

**错误方式**：
```python
# ❌ 降采样后计算 delta_t
total_time = t[-1] - t[0]
delta_t = total_time / n  # n 是降采样后的点数，导致 delta_t 变大
```

**正确方式**：
```python
# ✅ 使用相邻时间差的中位数
delta_t = np.median(np.diff(t.astype(float))) * 1e-6  # 微秒转秒
```

### 3.2 完整实现

```python
def compute_fft(topic, fields, t0, t1):
    # 1. 获取原始微秒 timestamp
    multi = get_multi_axis_signal(topic, fields, t0, t1, use_raw_timestamp=True)
    t = multi["timestamp"].to_numpy(dtype=np.int64)

    # 2. 计算采样间隔 (使用中位数，避免降采样影响)
    delta_t = np.median(np.diff(t.astype(float))) * 1e-6  # 微秒 → 秒

    # 3. 降采样 (仅减少计算量，不影响频率轴)
    if len(t) > 32768:
        step = len(t) // 32768
        multi = multi.iloc[::step, :]

    # 4. 计算 FFT
    for field in fields:
        y = multi[field].values
        y = y - np.mean(y)  # 去除直流分量
        n = len(y)

        # 频率轴
        freq = np.fft.rfftfreq(n, d=delta_t)

        # FFT 归一化 (Flight Review 标准)
        amp = (2.0 / n) * np.abs(np.fft.rfft(y))
```

---

## 4. 滤波器参数标记

### 4.1 Actuator Controls FFT

| 参数名 | 说明 |
|--------|------|
| `MC_DTERM_CUTOFF` | D 项滤波器截止频率 |
| `IMU_DGYRO_CUTOFF` | 角速度微分滤波器截止频率 |
| `IMU_GYRO_CUTOFF` | 陀螺仪滤波器截止频率 |

### 4.2 Angular Velocity FFT

| 参数名 | 说明 | 条件 |
|--------|------|------|
| `IMU_GYRO_CUTOFF` | 陀螺仪滤波器截止频率 | 始终标记 |
| `IMU_GYRO_NF_FREQ` | 陀螺仪陷波滤波器频率 | 仅当 > 0 时标记 |

---

## 5. 关键公式

### 5.1 采样间隔

```
delta_t = median(diff(timestamp)) × 1e-6  (秒)
```

### 5.2 采样频率

```
sampling_freq = 1 / delta_t  (Hz)
```

### 5.3 Nyquist 频率

```
nyquist = sampling_freq / 2  (Hz)
```

### 5.4 FFT 归一化

```
amplitude = (2/N) × |FFT(signal)|
```

### 5.5 频率轴

```python
freq = np.fft.rfftfreq(N, d=delta_t)
```

---

## 6. 常见问题排查

### 6.1 横轴频率范围过小 (如 0-10 Hz)

**原因**：降采样后使用 `总时间/点数` 计算 delta_t

**解决**：使用 `np.median(np.diff(t))` 计算采样间隔

### 6.2 timestamp 单位错误

**原因**：忘记将微秒转换为秒

**解决**：`delta_t = delta_t_us * 1e-6`

### 6.3 cutoff 标记位置与曲线不匹配

**原因**：频率轴计算错误

**解决**：确保使用正确的 delta_t 计算

---

## 7. 调试信息

代码中包含调试输出，格式如下：

```
[FFT Debug] topic=vehicle_angular_velocity, original_n=489298, resampled_n=32620, delta_t=3518.38us, sampling_freq=284.2Hz, nyquist=142.1Hz
[FFT Debug] field=xyz[0], freq_range=[0.00, 142.11] Hz
```

---

## 8. 参考文件

- `modules/analyzer.py`: FFT 计算核心逻辑
- `modules/flight_review_layout.py`: FFT 数据源配置
- `modules/flight_review_views.py`: FFT 图表渲染

---

## 9. 修改历史

| 日期 | 修改内容 |
|------|----------|
| 2026-02-15 | 修复频率轴计算错误，使用中位数计算 delta_t |
| 2026-02-15 | 添加动态控制分配检测 |
| 2026-02-15 | 完善滤波器参数标记 |
