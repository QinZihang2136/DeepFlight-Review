# Thrust & Magnetic Field 图表实现说明

## 概述

本文档描述 DeepFlight-Review 中 **Thrust & Magnetic Field** 图表的实现细节，确保与 PX4 Flight Review (https://review.px4.io) 的行为一致。

---

## 1. 磁场数据 (Magnetic Field)

### 1.1 Topic 优先级

严格按照 Flight Review 标准实现：

| 优先级 | Topic 名称 | 说明 |
|--------|------------|------|
| 1 | `vehicle_magnetometer` | 新版日志（推荐） |
| 2 | `sensor_combined` | 旧版日志 |

### 1.2 数据字段

只使用 `magnetometer_ga[0/1/2]` 字段：

| 字段名 | 描述 | 单位 |
|--------|------|------|
| `magnetometer_ga[0]` | X 轴磁场强度 | gauss |
| `magnetometer_ga[1]` | Y 轴磁场强度 | gauss |
| `magnetometer_ga[2]` | Z 轴磁场强度 | gauss |

### 1.3 磁场范数计算

```python
mag_norm = sqrt(magnetometer_ga[0]² + magnetometer_ga[1]² + magnetometer_ga[2]²)
```

**重要**：
- **不做归一化处理**，直接显示原始 gauss 值
- 地球磁场范数通常在 **0.25 - 0.65 gauss** 范围内
- 如果磁场范数随推力变化，说明存在电磁干扰 (EMI)

### 1.4 代码实现

```python
def _get_magnetic_series(self):
    # Topic 优先级: vehicle_magnetometer > sensor_combined
    topic = self._pick_topic_by_prefix(["vehicle_magnetometer", "sensor_combined"])

    # 只使用 magnetometer_ga[0/1/2] 字段
    mag_fields = ["magnetometer_ga[0]", "magnetometer_ga[1]", "magnetometer_ga[2]"]

    # 计算范数，不做归一化
    mag = np.sqrt(
        df["magnetometer_ga[0]"]**2 +
        df["magnetometer_ga[1]"]**2 +
        df["magnetometer_ga[2]"]**2
    )
```

---

## 2. 推力数据 (Thrust)

### 2.1 控制模式检测

根据日志中是否存在 `actuator_motors` 或 `actuator_servos` topic 来判断控制模式：

| 模式 | 检测条件 |
|------|----------|
| 动态控制分配 | 存在 `actuator_motors` 或 `actuator_servos` |
| 传统控制 | 不存在上述 topic |

### 2.2 动态控制分配模式

| 项目 | 值 |
|------|-----|
| Topic | `vehicle_thrust_setpoint` |
| 字段 | `xyz[0]`, `xyz[1]`, `xyz[2]` |
| 推力计算 | `sqrt(xyz[0]² + xyz[1]² + xyz[2]²)` |

### 2.3 传统控制模式

| 项目 | 值 |
|------|-----|
| Topic | `actuator_controls_0` |
| 字段 | `control[3]` |
| 推力值 | 直接使用 `control[3]` |

### 2.4 代码实现

```python
def _get_thrust_series(self):
    # 检测控制模式
    use_dynamic_control_alloc = any(
        name in ("actuator_motors", "actuator_servos")
        for name in topic_names
    )

    if use_dynamic_control_alloc:
        # 动态控制分配: vehicle_thrust_setpoint
        thrust = np.sqrt(xyz[0]² + xyz[1]² + xyz[2]²)
    else:
        # 传统控制: actuator_controls_0
        thrust = control[3]
```

---

## 3. 图表显示

### 3.1 Y 轴范围

- **Thrust**: 0 - 1 (归一化推力)
- **Magnetic Field**: 0.25 - 0.65 gauss (正常范围)

### 3.2 图例

- 绿色线: Thrust (推力)
- 橙色线: Norm of Magnetic Field (磁场范数)

### 3.3 异常检测

如果磁场范数出现以下情况，可能存在问题：

| 现象 | 可能原因 |
|------|----------|
| 磁场范数随推力增加而增加 | 电机电流对磁力计产生 EMI 干扰 |
| 磁场范数波动剧烈 | 磁力计校准问题或附近有铁磁性物质 |
| 磁场范数超出 0.25-0.65 范围 | 磁力计标定错误 |

---

## 4. 与 Flight Review 的一致性

本实现严格按照以下文档和源代码：

- **文档**: `/home/qinzihang/Programe/flight_review/docs/thrust_magnetic_field_plot.md`
- **源代码**: `PX4/flight_review/app/plot_app/configured_plots.py`
- **源代码**: `PX4/flight_review/app/plot_app/helper.py` (ActuatorControls 类)

### 4.1 一致性检查清单

- [x] 磁场 Topic 优先级: `vehicle_magnetometer` > `sensor_combined`
- [x] 磁场字段: 只使用 `magnetometer_ga[0/1/2]`
- [x] 磁场范数: 不做归一化
- [x] 推力动态控制分配检测
- [x] 推力动态控制分配: `vehicle_thrust_setpoint` + 三轴范数
- [x] 推力传统控制: `actuator_controls_0` + `control[3]`

---

## 5. 变更历史

| 日期 | 修改内容 |
|------|----------|
| 2024-02-14 | 移除磁场归一化，修复推力计算逻辑，严格对齐 Flight Review |
