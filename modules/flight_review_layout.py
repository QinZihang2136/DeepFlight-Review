"""
Flight Review 页面布局与信号映射定义。
"""

MODE_COLORS = {
    "MANUAL": "rgba(180,180,180,0.08)",
    "ALTCTL": "rgba(52,152,219,0.08)",
    "POSCTL": "rgba(46,204,113,0.08)",
    "AUTO_MISSION": "rgba(155,89,182,0.08)",
    "AUTO_LOITER": "rgba(241,196,15,0.08)",
    "AUTO_RTL": "rgba(230,126,34,0.08)",
    "AUTO_LAND": "rgba(231,76,60,0.08)",
    "OFFBOARD": "rgba(26,188,156,0.08)",
}

# 飞行模式中文描述
MODE_DESCRIPTIONS = {
    "MANUAL": "手动",
    "ALTCTL": "高度控制",
    "POSCTL": "位置控制",
    "AUTO_MISSION": "任务",
    "AUTO_LOITER": "悬停",
    "AUTO_RTL": "返航",
    "AUTO_LAND": "降落",
    "OFFBOARD": "离线控制",
}

# 用于图例显示的颜色（不透明版本）
MODE_COLORS_LEGEND = {
    "MANUAL": "#b4b4b4",
    "ALTCTL": "#3498db",
    "POSCTL": "#2ecc71",
    "AUTO_MISSION": "#9b59b6",
    "AUTO_LOITER": "#f1c40f",
    "AUTO_RTL": "#e67e22",
    "AUTO_LAND": "#e74c3c",
    "OFFBOARD": "#1abc9c",
}

GPS_NOISE_FIELDS = {
    "noise": ["noise_per_ms", "noise", "noise_per_ms[0]"],
    "jamming": ["jamming_indicator", "jamming", "jam_ind"],
}

ACCEL_AXIS_CANDIDATES = [
    {"topic": "sensor_combined", "fields": ["accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]"]},
    {"topic": "sensor_accel", "fields": ["x", "y", "z"]},
]

GYRO_AXIS_CANDIDATES = [
    {"topic": "sensor_combined", "fields": ["gyro_rad[0]", "gyro_rad[1]", "gyro_rad[2]"], "to_deg": True},
    {"topic": "vehicle_angular_velocity", "fields": ["xyz[0]", "xyz[1]", "xyz[2]"], "to_deg": True},
]

# Actuator Controls FFT 候选 (按 Flight Review 标准)
# 1. 传统模式: actuator_controls_0 + control[0/1/2]
# 2. 动态控制分配: vehicle_torque_setpoint + xyz[0/1/2] (需检测 actuator_motors/actuator_servos)
ACTUATOR_FFT_CANDIDATES_TRADITIONAL = [
    {"topic": "actuator_controls_0", "fields": ["control[0]", "control[1]", "control[2]"], "labels": ["Roll", "Pitch", "Yaw"]},
]

ACTUATOR_FFT_CANDIDATES_DYNAMIC = [
    {"topic": "vehicle_torque_setpoint", "fields": ["xyz[0]", "xyz[1]", "xyz[2]"], "labels": ["Roll", "Pitch", "Yaw"]},
]

# 兼容旧接口 (将在运行时根据动态控制分配检测选择)
ACTUATOR_FFT_CANDIDATES = ACTUATOR_FFT_CANDIDATES_TRADITIONAL

# Angular Velocity FFT 候选 (按 Flight Review 标准)
# 只使用 vehicle_angular_velocity，不需要备用 topic
ANGULAR_RATE_FFT_CANDIDATES = [
    {"topic": "vehicle_angular_velocity", "fields": ["xyz[0]", "xyz[1]", "xyz[2]"], "labels": ["Rollspeed", "Pitchspeed", "Yawspeed"], "to_deg": True},
]

THRUST_TOPIC_CANDIDATES = ["actuator_controls", "actuator_motors"]
MAGNETIC_TOPIC_CANDIDATES = ["sensor_mag", "vehicle_magnetometer", "sensor_combined"]

# 滤波器参数 (按 Flight Review 标准)
# Actuator Controls FFT 标记: MC_DTERM_CUTOFF, IMU_DGYRO_CUTOFF, IMU_GYRO_CUTOFF
# Angular Velocity FFT 标记: IMU_GYRO_CUTOFF, IMU_GYRO_NF_FREQ (仅当 > 0 时)
IMU_CUTOFF_PARAMS = ["MC_DTERM_CUTOFF", "IMU_DGYRO_CUTOFF", "IMU_GYRO_CUTOFF", "IMU_GYRO_NF_FREQ"]

# 用于检测动态控制分配的 topic 名称
DYNAMIC_CONTROL_ALLOC_TOPICS = ["actuator_motors", "actuator_servos"]

# 每组使用候选 topic，按先后回退。
FLIGHT_REVIEW_GROUPS = [
    {
        "key": "vehicle_status",
        "title": "Vehicle Status",
        "expanded": True,
        "topic_candidates": ["vehicle_status"],
        "signals": [
            ("arming_state", "arming_state"),
            ("nav_state", "nav_state"),
            ("failsafe", "failsafe"),
        ],
    },
    {
        "key": "attitude",
        "title": "Attitude (with Setpoint)",
        "expanded": True,
        "topic_candidates": ["vehicle_attitude"],
        "setpoint_topic": "vehicle_attitude_setpoint",
        "signals": [
            ("roll_deg", "Roll"),
            ("pitch_deg", "Pitch"),
            ("yaw_deg", "Yaw"),
        ],
        "setpoint_signals": [
            ("roll_body_deg", "Roll SP"),
            ("pitch_body_deg", "Pitch SP"),
            ("yaw_body_deg", "Yaw SP"),
        ],
    },
    {
        "key": "rates",
        "title": "Rates (with Setpoint)",
        "expanded": True,
        "topic_candidates": ["vehicle_angular_velocity"],
        "setpoint_topic": "vehicle_rates_setpoint",
        "signals": [
            ("xyz[0]_deg", "Roll Rate"),
            ("xyz[1]_deg", "Pitch Rate"),
            ("xyz[2]_deg", "Yaw Rate"),
        ],
        "setpoint_signals": [
            ("roll_deg", "Roll Rate SP"),
            ("pitch_deg", "Pitch Rate SP"),
            ("yaw_deg", "Yaw Rate SP"),
        ],
    },
    {
        "key": "position_velocity",
        "title": "Position / Velocity",
        "expanded": True,
        "topic_candidates": ["vehicle_local_position"],
        "signals": [
            ("x", "x"),
            ("y", "y"),
            ("altitude", "altitude"),
            ("vx", "vx"),
            ("vy", "vy"),
            ("vz", "vz"),
        ],
    },
    {
        "key": "actuators",
        "title": "Actuators",
        "expanded": False,
        "topic_candidates": ["actuator_outputs", "actuator_motors"],
        "signals": [],
    },
    {
        "key": "power",
        "title": "Power",
        "expanded": True,
        "topic_candidates": ["battery_status"],
        "signals": [
            ("voltage_v", "voltage_v"),
            ("current_a", "current_a"),
            ("remaining", "remaining"),
        ],
    },
    {
        "key": "gps",
        "title": "GPS",
        "expanded": False,
        "topic_candidates": ["vehicle_gps_position"],
        "signals": [
            ("fix_type", "fix_type"),
            ("eph", "eph"),
            ("epv", "epv"),
            ("vel_m_s", "speed_m_s"),
        ],
    },
    {
        "key": "ekf",
        "title": "Estimator / EKF",
        "expanded": False,
        "topic_candidates": ["estimator_status", "ekf2_estimator_status"],
        "signals": [
            ("pos_horiz_reset_counter", "pos_horiz_reset_counter"),
            ("pos_vert_reset_counter", "pos_vert_reset_counter"),
            ("vel_horiz_reset_counter", "vel_horiz_reset_counter"),
            ("vel_vert_reset_counter", "vel_vert_reset_counter"),
            ("yaw_reset_counter", "yaw_reset_counter"),
            ("filter_fault_flags", "filter_fault_flags"),
        ],
    },
    {
        "key": "vibration",
        "title": "Vibration",
        "expanded": False,
        "topic_candidates": ["sensor_combined"],
        "signals": [
            ("accelerometer_m_s2[0]", "acc_x"),
            ("accelerometer_m_s2[1]", "acc_y"),
            ("accelerometer_m_s2[2]", "acc_z"),
            ("gyro_rad[0]_deg", "gyro_x"),
            ("gyro_rad[1]_deg", "gyro_y"),
            ("gyro_rad[2]_deg", "gyro_z"),
        ],
    },
    {
        "key": "rc",
        "title": "RC / Manual Control",
        "expanded": False,
        "topic_candidates": ["manual_control_setpoint", "input_rc"],
        "signals": [],
    },
    {
        "key": "temperature",
        "title": "Temperature",
        "expanded": False,
        "topic_candidates": ["sensor_combined", "vehicle_air_data"],
        "signals": [
            ("baro_temp_celsius", "Baro Temperature [°C]"),
            ("accel_temp_celsius", "Accel Temperature [°C]"),
            ("gyro_temp_celsius", "Gyro Temperature [°C]"),
        ],
    },
    {
        "key": "cpu_ram",
        "title": "CPU & RAM",
        "expanded": False,
        "topic_candidates": ["cpuload"],
        "signals": [
            ("load", "CPU Load [%]"),
            ("ram_usage", "RAM Usage"),
        ],
    },
]
