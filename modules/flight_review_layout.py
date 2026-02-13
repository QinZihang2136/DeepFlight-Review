"""
Flight Review 页面布局与信号映射定义。
"""

MODE_COLORS = {
    "MANUAL": "rgba(180,180,180,0.12)",
    "ALTCTL": "rgba(52,152,219,0.12)",
    "POSCTL": "rgba(46,204,113,0.12)",
    "AUTO_MISSION": "rgba(155,89,182,0.12)",
    "AUTO_LOITER": "rgba(241,196,15,0.12)",
    "AUTO_RTL": "rgba(230,126,34,0.12)",
    "AUTO_LAND": "rgba(231,76,60,0.12)",
    "OFFBOARD": "rgba(26,188,156,0.12)",
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

ACTUATOR_FFT_CANDIDATES = [
    {"topic_prefix": "actuator_controls", "fields": ["control[0]", "control[1]", "control[2]"], "labels": ["Roll", "Pitch", "Yaw"]},
    {"topic": "vehicle_rates_setpoint", "fields": ["roll", "pitch", "yaw"], "labels": ["Roll", "Pitch", "Yaw"]},
]

ANGULAR_RATE_FFT_CANDIDATES = [
    {"topic": "vehicle_angular_velocity", "fields": ["xyz[0]", "xyz[1]", "xyz[2]"], "labels": ["Rollspeed", "Pitchspeed", "Yawspeed"], "to_deg": True},
    {"topic": "vehicle_rates_setpoint", "fields": ["roll", "pitch", "yaw"], "labels": ["Rollspeed", "Pitchspeed", "Yawspeed"], "to_deg": True},
]

THRUST_TOPIC_CANDIDATES = ["actuator_controls", "actuator_motors"]
MAGNETIC_TOPIC_CANDIDATES = ["sensor_mag", "vehicle_magnetometer", "sensor_combined"]
IMU_CUTOFF_PARAMS = ["IMU_GYRO_CUTOFF", "IMU_DGYRO_CUTOFF"]

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
        "title": "Attitude",
        "expanded": True,
        "topic_candidates": ["vehicle_attitude"],
        "signals": [
            ("roll_deg", "roll"),
            ("pitch_deg", "pitch"),
            ("yaw_deg", "yaw"),
        ],
    },
    {
        "key": "rates",
        "title": "Rates",
        "expanded": True,
        "topic_candidates": ["vehicle_angular_velocity"],
        "signals": [
            ("xyz[0]_deg", "roll_rate"),
            ("xyz[1]_deg", "pitch_rate"),
            ("xyz[2]_deg", "yaw_rate"),
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
]
