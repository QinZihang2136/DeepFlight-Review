# Spectrogram 频谱图实现文档

## 概述

本文档描述 LogCortex 中 Spectrogram（时频谱图）的实现，严格对齐 Flight Review 标准。

---

## 1. 支持的频谱图

| 图表 | 数据源 Topic | 数据字段 | 用途 |
|------|-------------|----------|------|
| **Acceleration Power Spectral Density** | `sensor_combined` | `accelerometer_m_s2[0/1/2]` | 分析三轴加速度的时频特性 |
| **Angular Velocity Power Spectral Density** | `vehicle_angular_velocity` | `xyz[0/1/2]` | 分析三轴角速度的时频特性 |

---

## 2. 核心实现原理

### 2.1 三轴 PSD 求和（关键）

Flight Review 标准是对三轴数据分别计算 spectrogram，然后将 PSD **求和**：

```python
# 对每个轴计算 spectrogram
for field in fields:  # [0], [1], [2]
    f, t_spec, pxx = spectrogram(y, fs=fs, ...)
    if sum_pxx is None:
        sum_pxx = pxx.copy()
    else:
        sum_pxx += pxx  # 累加 PSD

# 转换为 dB
power_db = 10.0 * np.log10(sum_pxx)
```

### 2.2 采样频率计算（Flight Review 标准）

**关键**：使用总时间除以点数，而不是中位数。

```python
# ✅ Flight Review 标准方式
delta_t = ((timestamp[-1] - timestamp[0]) * 1e-6) / len(timestamp)
sampling_frequency = int(1.0 / delta_t)  # 取整
```

**注意**：Logging dropouts 不在此计算中考虑。

### 2.3 采样频率要求

```python
if sampling_frequency < 100:  # Hz
    # 不绘制频谱图，采样频率过低
    return
```

---

## 3. 完整实现

```python
def compute_spectrogram(topic, fields, t0, t1, nperseg=256, noverlap=128):
    """
    Flight Review 标准参数:
    - nperseg=256 (窗口长度)
    - noverlap=128 (50% 重叠)
    - window='hann' (汉宁窗)
    - scaling='density' (功率谱密度)
    """
    # 1. 获取原始微秒 timestamp（不降采样）
    multi = get_multi_axis_signal(topic, fields, t0, t1, use_raw_timestamp=True)
    t = multi["timestamp"].to_numpy(dtype=np.int64)

    # 2. Flight Review 标准计算 delta_t
    delta_t = ((t[-1] - t[0]) * 1.0e-6) / len(t)
    sampling_frequency = int(1.0 / delta_t)

    # 3. 检查采样频率
    if sampling_frequency < 100:
        return DataFrame()  # 采样频率过低

    # 4. 对每个轴计算 spectrogram 并求和
    from scipy.signal import spectrogram
    sum_pxx = None

    for field in fields:
        y = multi[field].to_numpy(dtype=float)
        f, t_spec, pxx = spectrogram(
            y,
            fs=sampling_frequency,
            window='hann',
            nperseg=256,
            noverlap=128,
            scaling='density',
        )
        if sum_pxx is None:
            sum_pxx = pxx.copy()
            f_hz, t_rel = f, t_spec
        else:
            sum_pxx += pxx

    # 5. 转换为 dB
    power_db = 10.0 * np.log10(sum_pxx)

    # 6. 处理 -inf 值（Bokeh/JSON 不支持）
    if -np.inf in power_db:
        finite_min = np.min(np.ma.masked_invalid(power_db))
        power_db[power_db == -np.inf] = finite_min

    # 7. 时间轴转换：加上数据起始时间
    t_start_sec = t[0] * 1e-6
    t_abs = t_rel + t_start_sec

    return DataFrame({"time_s": ..., "freq_hz": ..., "power_db": ...})
```

---

## 4. 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `nperseg` | 256 | 窗口长度（样本数） |
| `noverlap` | 128 | 重叠样本数（50%） |
| `window` | `'hann'` | 汉宁窗 |
| `scaling` | `'density'` | 功率谱密度 |

---

## 5. 常见问题排查

### 5.1 频率范围错误

**症状**：频率范围与 Flight Review 不一致

**原因**：delta_t 计算方式错误

**解决**：确保使用 `((t[-1] - t[0]) * 1e-6) / len(t)` 而不是 `np.median(np.diff(t))`

### 5.2 颜色分布差异大

**症状**：热力图颜色分布与 Flight Review 差异明显

**原因**：
1. 未使用三轴 PSD 求和
2. 功率值范围不同

**解决**：
1. 确保对三轴数据求和
2. 使用 `10 * np.log10(sum_pxx)` 转换为 dB

### 5.3 时间轴不对齐

**症状**：频谱图时间范围与其他图表不一致

**原因**：未加上数据起始时间

**解决**：`t_abs = t_rel + t[0] * 1e-6`

---

## 6. 调试信息

代码中包含调试输出：

```
[Spectrogram Debug] topic=sensor_combined, fields=['accelerometer_m_s2[0]', ...],
n=19986, delta_t=5000.25us, fs=199Hz, freq_range=[0.00, 99.50]Hz
```

关键指标：
- `n`: 数据点数
- `delta_t`: 采样间隔（微秒）
- `fs`: 采样频率（Hz）
- `freq_range`: 频率范围（应接近 Nyquist 频率 = fs/2）

---

## 7. 参考文件

| 文件 | 说明 |
|------|------|
| `modules/analyzer.py` | `compute_spectrogram()` 方法 |
| `modules/flight_review_views.py` | `_render_single_spectrogram()` 渲染 |
| `modules/flight_review_layout.py` | 数据源配置 `ACCEL_AXIS_CANDIDATES`, `GYRO_AXIS_CANDIDATES` |

---

## 8. 与 FFT 实现的区别

| 特性 | FFT | Spectrogram |
|------|-----|-------------|
| delta_t 计算 | `np.median(np.diff(t))` | `((t[-1] - t[0]) / len(t))` |
| 降采样 | 支持（不影响频率轴） | 不支持（使用原始数据） |
| 多轴处理 | 分别显示 | PSD 求和后显示 |
| 输出 | 频域幅度 | 时频功率谱 |

---

## 9. 修改历史

| 日期 | 修改内容 |
|------|----------|
| 2026-02-15 | 初始实现 |
| 2026-02-15 | 修复三轴 PSD 求和 |
| 2026-02-15 | 修复 delta_t 计算方式，严格对齐 Flight Review |
| 2026-02-15 | 修复参数：nperseg=256, noverlap=128 |
