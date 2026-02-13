from pyulog import ULog
import pandas as pd
import numpy as np
import datetime
from typing import Dict, List, Optional


ARMING_STATE_MAP = {
    0: "INIT",
    1: "STANDBY",
    2: "ARMED",
    3: "STANDBY_ERROR",
    4: "REBOOT",
    5: "IN_AIR_RESTORED",
    6: "SHUTDOWN",
}

ARMING_STATE_DESC = {
    0: "初始化中",
    1: "待机（已上锁）",
    2: "已解锁",
    3: "待机错误",
    4: "重启中",
    5: "空中恢复",
    6: "关机",
}

NAV_STATE_MAP = {
    0: "MANUAL",
    1: "ALTCTL",
    2: "POSCTL",
    3: "AUTO_MISSION",
    4: "AUTO_LOITER",
    5: "AUTO_RTL",
    6: "AUTO_RCRS",
    7: "AUTO_FOLLOW_TARGET",
    8: "AUTO_LAND",
    9: "AUTO_READY",
    10: "AUTO_TAKEOFF",
    11: "AUTO_PRECLAND",
    12: "OFFBOARD",
    13: "STAB",
    14: "MANUAL_STABILIZED",
    15: "RATTITUDE",
    16: "ACRO",
    17: "AUTO_BOARD_OFFLOAD",
    18: "DESCEND",
    19: "TERMINATION",
    20: "OFFBOARD_TEST",
    21: "COUNT",
}

NAV_STATE_DESC = {
    0: "手动模式",
    1: "高度控制模式",
    2: "位置控制模式",
    3: "自动任务模式",
    4: "自动悬停模式",
    5: "自动返航模式",
    6: "自动返航航线模式",
    7: "自动跟随目标模式",
    8: "自动降落模式",
    9: "自动准备模式",
    10: "自动起飞模式",
    11: "自动精准降落模式",
    12: "离线控制模式",
    13: "自稳模式",
    14: "手动自稳模式",
    15: "半自稳模式",
    16: "特技模式",
    17: "自动板载卸载模式",
    18: "下降模式",
    19: "终止模式",
    20: "离线测试模式",
    21: "模式计数",
}


class LogAnalyzer:
    def __init__(self, file_path):
        self.ulog = ULog(file_path)
        self.data_cache = {}
        self.analysis_cache = {}
        self._parse_meta_data()
        self._calculate_kpis()

    def _parse_meta_data(self):
        try:
            self.start_time = datetime.datetime.fromtimestamp(self.ulog.start_timestamp / 1e6)
            self.duration = (self.ulog.last_timestamp - self.ulog.start_timestamp) / 1e6
            self.sys_name = self.ulog.msg_info_dict.get("sys_name", "Unknown")
            self.ver_sw = self.ulog.msg_info_dict.get("ver_sw", "Unknown")
            self.airframe = self.ulog.msg_info_dict.get("sys_autostart_id", "Unknown")
        except Exception:
            self.start_time = datetime.datetime.now()
            self.duration = 0

    def _calculate_kpis(self):
        self.kpis = {"max_alt": 0, "max_speed": 0, "dist_3d": 0}
        try:
            df = self.get_topic_data("vehicle_local_position", downsample=False)
            if df is not None and "z" in df:
                self.kpis["max_alt"] = abs(df["z"].min())

            if df is not None and "vx" in df and "vy" in df:
                speed = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
                self.kpis["max_speed"] = speed.max()
        except Exception:
            pass

    def _pick_first_topic(self, candidates):
        names = {(d.name, int(getattr(d, "multi_id", 0))) for d in self.ulog.data_list}
        for name in candidates:
            if (name, 0) in names:
                return name
        return None

    def _to_seconds(self, timestamps):
        return (timestamps.astype(np.float64) - float(self.ulog.start_timestamp)) / 1e6

    def _unique_change_times(self, t_s, values):
        if len(values) == 0:
            return []
        out = []
        last = values[0]
        out.append((float(t_s[0]), self._py(last)))
        for i in range(1, len(values)):
            if values[i] != last:
                last = values[i]
                out.append((float(t_s[i]), self._py(last)))
        return out

    def _py(self, value):
        if hasattr(value, "item"):
            return value.item()
        return value

    def _format_arming_info(self, value):
        v = int(value)
        return {
            "value": v,
            "name": ARMING_STATE_MAP.get(v, f"UNKNOWN({v})"),
            "description_cn": ARMING_STATE_DESC.get(v, f"未知状态({v})"),
        }

    def _format_nav_info(self, value):
        v = int(value)
        return {
            "value": v,
            "name": NAV_STATE_MAP.get(v, f"UNKNOWN({v})"),
            "description_cn": NAV_STATE_DESC.get(v, f"未知模式({v})"),
        }

    def get_gps_tracks(self):
        df_global = self.get_topic_data("vehicle_global_position", downsample=True)
        if df_global is not None and "lat" in df_global and "lon" in df_global:
            valid = df_global[(df_global["lat"] != 0) & (df_global["lon"] != 0)].copy()
            if not valid.empty:
                return valid[["lat", "lon", "alt", "timestamp"]].rename(
                    columns={"lat": "lat_deg", "lon": "lon_deg", "alt": "alt_rel"}
                )

        df_raw = self.get_topic_data("vehicle_gps_position", downsample=True)
        if df_raw is not None and "lat" in df_raw and "lon" in df_raw:
            valid = df_raw[(df_raw["fix_type"] >= 2) & (df_raw["lat"] != 0)].copy()
            if not valid.empty:
                if valid["lat"].abs().mean() > 180:
                    valid["lat_deg"] = valid["lat"] * 1e-7
                    valid["lon_deg"] = valid["lon"] * 1e-7
                else:
                    valid["lat_deg"] = valid["lat"]
                    valid["lon_deg"] = valid["lon"]

                valid["alt_rel"] = valid["alt"] * 1e-3 if valid["alt"].mean() > 1000 else valid["alt"]
                return valid[["lat_deg", "lon_deg", "alt_rel", "timestamp"]]

        return None

    def get_topic_data(self, topic_name, downsample=True):
        cache_key = f"{topic_name}_{downsample}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        try:
            dataset = None
            for d in self.ulog.data_list:
                if d.name == topic_name:
                    dataset = d
                    break
            if dataset is None:
                return None

            df = pd.DataFrame(dataset.data)
            df["timestamp"] = (df["timestamp"] - self.ulog.start_timestamp) / 1e6

            self._augment_data(topic_name, df)

            if downsample and len(df) > 5000:
                step = len(df) // 5000
                df = df.iloc[::step, :]

            self.data_cache[cache_key] = df
            return df
        except Exception:
            return None

    def get_topic(self, topic: str, downsample: bool = True) -> Optional[pd.DataFrame]:
        """Flight Review 风格别名接口。"""
        return self.get_topic_data(topic, downsample=downsample)

    def get_topic_multi(self, topic: str):
        """返回同名 topic 的所有 multi instance 数据。"""
        out = []
        for d in self.ulog.data_list:
            if d.name != topic:
                continue
            try:
                ds = self.ulog.get_dataset(topic, int(getattr(d, "multi_id", 0)))
                df = pd.DataFrame(ds.data)
                df["timestamp"] = (df["timestamp"] - self.ulog.start_timestamp) / 1e6
                self._augment_data(topic, df)
                df["multi_id"] = int(getattr(d, "multi_id", 0))
                out.append(df)
            except Exception:
                continue
        return out

    def _augment_data(self, topic, df):
        if "attitude" in topic and "q[0]" in df.columns:
            w, x, y, z = df["q[0]"], df["q[1]"], df["q[2]"], df["q[3]"]
            df["roll_deg"] = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
            df["pitch_deg"] = np.degrees(np.arcsin(2 * (w * y - z * x)))
            df["yaw_deg"] = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

        for col in df.columns:
            if any(k in col for k in ["roll", "pitch", "yaw", "xyz"]) and "deg" not in col:
                if df[col].abs().max() < 10:
                    df[col + "_deg"] = np.degrees(df[col])

        if "z" in df.columns and "position" in topic:
            df["altitude"] = -df["z"]

    def get_available_topics(self):
        return sorted(list(set([d.name for d in self.ulog.data_list])))

    def get_mode_segments(self):
        """返回 nav_state 模式区间，用于图背景着色。"""
        status_topic = self._pick_first_topic(["vehicle_status"])
        if not status_topic:
            return []
        try:
            st = self.ulog.get_dataset(status_topic, 0)
            if "timestamp" not in st.data or "nav_state" not in st.data:
                return []
            t_s = self._to_seconds(st.data["timestamp"].astype(np.int64))
            nav = st.data["nav_state"].astype(np.int64)
            changes = self._unique_change_times(t_s, nav)
            if not changes:
                return []
            segments = []
            for i, (t0, v) in enumerate(changes):
                t1 = changes[i + 1][0] if i + 1 < len(changes) else float(self.duration)
                v_int = int(v)
                segments.append(
                    {
                        "t0": float(t0),
                        "t1": float(t1),
                        "nav_state": v_int,
                        "name": NAV_STATE_MAP.get(v_int, f"UNKNOWN({v_int})"),
                    }
                )
            return segments
        except Exception:
            return []

    def get_messages(self, limit: int = 5000):
        """提取 ULog 日志消息，返回 DataFrame。"""
        rows = []
        try:
            for msg in getattr(self.ulog, "logged_messages", [])[:limit]:
                rows.append(
                    {
                        "timestamp": (int(msg.timestamp) - self.ulog.start_timestamp) / 1e6,
                        "level": int(getattr(msg, "log_level", 0)),
                        "level_str": str(getattr(msg, "log_level_str", "")),
                        "message": str(getattr(msg, "message", "")),
                    }
                )
            for msg in getattr(self.ulog, "logged_messages_tagged", [])[:limit]:
                rows.append(
                    {
                        "timestamp": (int(msg.timestamp) - self.ulog.start_timestamp) / 1e6,
                        "level": int(getattr(msg, "log_level", 0)),
                        "level_str": str(getattr(msg, "log_level_str", "")),
                        "message": str(getattr(msg, "message", "")),
                    }
                )
        except Exception:
            return pd.DataFrame(columns=["timestamp", "level", "level_str", "message"])
        if not rows:
            return pd.DataFrame(columns=["timestamp", "level", "level_str", "message"])
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return df.head(limit)

    def get_parameter_changes(self, limit: int = 2000):
        """返回参数变化 DataFrame。"""
        rows = []
        for ts_us, name, value in self.ulog.changed_parameters[:limit]:
            rows.append(
                {
                    "timestamp": round((ts_us - self.ulog.start_timestamp) / 1e6, 6),
                    "name": str(name),
                    "value": self._py(value),
                }
            )
        return pd.DataFrame(rows)

    def get_signal(self, topic: str, field: str, t0: float = None, t1: float = None):
        """返回单信号 time/value。"""
        df = self.get_topic_data(topic, downsample=False)
        if df is None or field not in df.columns or "timestamp" not in df.columns:
            return pd.DataFrame(columns=["timestamp", "value"])
        out = df[["timestamp", field]].rename(columns={field: "value"}).copy()
        if t0 is not None:
            out = out[out["timestamp"] >= float(t0)]
        if t1 is not None:
            out = out[out["timestamp"] <= float(t1)]
        out = out.replace([np.inf, -np.inf], np.nan).dropna()
        return out

    def get_multi_axis_signal(self, topic: str, fields: List[str], t0: float = None, t1: float = None):
        """返回 timestamp + 多轴有效字段。"""
        df = self.get_topic_data(topic, downsample=False)
        if df is None or "timestamp" not in df.columns:
            return pd.DataFrame(columns=["timestamp"])
        valid_fields = [f for f in fields if f in df.columns]
        if not valid_fields:
            return pd.DataFrame(columns=["timestamp"])
        out = df[["timestamp"] + valid_fields].copy()
        if t0 is not None:
            out = out[out["timestamp"] >= float(t0)]
        if t1 is not None:
            out = out[out["timestamp"] <= float(t1)]
        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    def _estimate_dt(self, timestamps):
        if len(timestamps) < 2:
            return None
        dt = np.median(np.diff(timestamps.astype(float)))
        if not np.isfinite(dt) or dt <= 0:
            return None
        return float(dt)

    def _resample_step(self, n: int, max_points: int):
        if n <= max_points:
            return 1
        return int(np.ceil(n / float(max_points)))

    def compute_fft(self, topic: str, field: str, t0: float = None, t1: float = None):
        sig = self.get_signal(topic, field, t0=t0, t1=t1)
        if sig.empty or len(sig) < 8:
            return pd.DataFrame(columns=["freq_hz", "amplitude"])
        t = sig["timestamp"].to_numpy(dtype=float)
        y = sig["value"].to_numpy(dtype=float)
        dt = np.median(np.diff(t))
        if not np.isfinite(dt) or dt <= 0:
            return pd.DataFrame(columns=["freq_hz", "amplitude"])
        y = y - np.mean(y)
        n = len(y)
        freq = np.fft.rfftfreq(n, d=dt)
        amp = np.abs(np.fft.rfft(y)) / max(1, n)
        return pd.DataFrame({"freq_hz": freq, "amplitude": amp})

    def compute_fft_multi(
        self,
        topic: str,
        fields: List[str],
        t0: float = None,
        t1: float = None,
        window: str = "hann",
    ) -> Dict[str, pd.DataFrame]:
        """对多个字段计算 FFT。"""
        key = ("fft_multi", topic, tuple(fields), t0, t1, window)
        if key in self.analysis_cache:
            return self.analysis_cache[key]

        result = {f: pd.DataFrame(columns=["freq_hz", "amplitude"]) for f in fields}
        multi = self.get_multi_axis_signal(topic, fields, t0=t0, t1=t1)
        if multi.empty or "timestamp" not in multi.columns:
            self.analysis_cache[key] = result
            return result

        t = multi["timestamp"].to_numpy(dtype=float)
        if len(t) < 8:
            self.analysis_cache[key] = result
            return result

        step = self._resample_step(len(t), max_points=32768)
        if step > 1:
            multi = multi.iloc[::step, :]
            t = multi["timestamp"].to_numpy(dtype=float)

        dt = self._estimate_dt(t)
        if dt is None:
            self.analysis_cache[key] = result
            return result

        for field in fields:
            if field not in multi.columns:
                continue
            y = multi[field].to_numpy(dtype=float)
            mask = np.isfinite(y)
            y = y[mask]
            if len(y) < 8:
                continue
            y = y - np.mean(y)
            n = len(y)
            if window == "hann":
                win = np.hanning(n)
                yw = y * win
                scale = np.sum(win) / max(1, n)
                scale = scale if scale > 0 else 1.0
            else:
                yw = y
                scale = 1.0
            freq = np.fft.rfftfreq(n, d=dt)
            amp = np.abs(np.fft.rfft(yw)) / max(1, n) / scale
            result[field] = pd.DataFrame({"freq_hz": freq, "amplitude": amp})

        self.analysis_cache[key] = result
        return result

    def compute_psd(self, topic: str, field: str, t0: float = None, t1: float = None):
        sig = self.get_signal(topic, field, t0=t0, t1=t1)
        if sig.empty or len(sig) < 16:
            return pd.DataFrame(columns=["freq_hz", "psd"])
        t = sig["timestamp"].to_numpy(dtype=float)
        y = sig["value"].to_numpy(dtype=float)
        dt = np.median(np.diff(t))
        if not np.isfinite(dt) or dt <= 0:
            return pd.DataFrame(columns=["freq_hz", "psd"])
        fs = 1.0 / dt
        y = y - np.mean(y)
        try:
            from scipy.signal import welch

            nperseg = min(1024, max(64, len(y) // 2))
            freq, psd = welch(y, fs=fs, nperseg=nperseg)
        except Exception:
            # scipy 不可用时退化到简单周期图
            n = len(y)
            freq = np.fft.rfftfreq(n, d=dt)
            yf = np.fft.rfft(y)
            psd = (np.abs(yf) ** 2) / (fs * n)
        return pd.DataFrame({"freq_hz": freq, "psd": psd})

    def compute_spectrogram(
        self,
        topic: str,
        field: str,
        t0: float = None,
        t1: float = None,
        nperseg: int = 512,
        noverlap: int = 256,
    ):
        """返回时频谱长表: time_s, freq_hz, power_db。"""
        key = ("spectrogram", topic, field, t0, t1, nperseg, noverlap)
        if key in self.analysis_cache:
            return self.analysis_cache[key]

        sig = self.get_signal(topic, field, t0=t0, t1=t1)
        if sig.empty or len(sig) < 64:
            out = pd.DataFrame(columns=["time_s", "freq_hz", "power_db"])
            self.analysis_cache[key] = out
            return out

        step = self._resample_step(len(sig), max_points=120000)
        if step > 1:
            sig = sig.iloc[::step, :]
        t = sig["timestamp"].to_numpy(dtype=float)
        y = sig["value"].to_numpy(dtype=float)
        dt = self._estimate_dt(t)
        if dt is None:
            out = pd.DataFrame(columns=["time_s", "freq_hz", "power_db"])
            self.analysis_cache[key] = out
            return out

        y = y - np.mean(y)
        fs = 1.0 / dt
        nseg = int(min(max(64, nperseg), len(y)))
        nov = int(min(max(0, noverlap), nseg - 1))

        try:
            from scipy.signal import spectrogram

            f_hz, t_rel, pxx = spectrogram(
                y,
                fs=fs,
                window="hann",
                nperseg=nseg,
                noverlap=nov,
                detrend=False,
                scaling="density",
                mode="psd",
            )
        except Exception:
            out = pd.DataFrame(columns=["time_s", "freq_hz", "power_db"])
            self.analysis_cache[key] = out
            return out

        power_db = 10.0 * np.log10(np.maximum(pxx, 1e-15))
        t_abs = t_rel + float(sig["timestamp"].iloc[0])
        tt, ff = np.meshgrid(t_abs, f_hz)
        out = pd.DataFrame(
            {
                "time_s": tt.ravel(),
                "freq_hz": ff.ravel(),
                "power_db": power_db.ravel(),
            }
        )
        self.analysis_cache[key] = out
        return out

    def _pick_topic_by_prefix(self, prefixes: List[str]):
        names = sorted({d.name for d in self.ulog.data_list})
        for p in prefixes:
            for name in names:
                if name == p or name.startswith(p):
                    return name
        return None

    def _pick_field(self, df: pd.DataFrame, candidates: List[str]):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def get_gps_noise_jamming(self, t0: float = None, t1: float = None):
        """返回 GPS 噪声与干扰指标。"""
        topic = self._pick_topic_by_prefix(["vehicle_gps_position"])
        if not topic:
            return pd.DataFrame(columns=["timestamp", "noise_per_ms", "jamming_indicator"])
        df = self.get_topic_data(topic, downsample=False)
        if df is None or "timestamp" not in df.columns:
            return pd.DataFrame(columns=["timestamp", "noise_per_ms", "jamming_indicator"])

        noise_field = self._pick_field(df, ["noise_per_ms", "noise", "noise_per_ms[0]"])
        jam_field = self._pick_field(df, ["jamming_indicator", "jamming", "jam_ind"])
        if not noise_field and not jam_field:
            return pd.DataFrame(columns=["timestamp", "noise_per_ms", "jamming_indicator"])

        out = pd.DataFrame({"timestamp": df["timestamp"]})
        out["noise_per_ms"] = df[noise_field] if noise_field else np.nan
        out["jamming_indicator"] = df[jam_field] if jam_field else np.nan
        if t0 is not None:
            out = out[out["timestamp"] >= float(t0)]
        if t1 is not None:
            out = out[out["timestamp"] <= float(t1)]
        return out.replace([np.inf, -np.inf], np.nan).dropna(how="all", subset=["noise_per_ms", "jamming_indicator"])

    def _get_thrust_series(self):
        topic = self._pick_topic_by_prefix(["actuator_controls", "actuator_motors"])
        if not topic:
            return None
        df = self.get_topic_data(topic, downsample=False)
        if df is None or "timestamp" not in df.columns:
            return None

        thrust_field = self._pick_field(df, ["thrust_body[2]", "control[3]", "thrust", "thrust_z"])
        if thrust_field:
            out = pd.DataFrame({"timestamp": df["timestamp"], "thrust": df[thrust_field].astype(float)})
            return out

        motor_fields = [c for c in df.columns if c != "timestamp" and ("output[" in c or "control[" in c)]
        if not motor_fields:
            return None
        out = pd.DataFrame({"timestamp": df["timestamp"], "thrust": df[motor_fields].astype(float).mean(axis=1)})
        return out

    def _get_magnetic_series(self):
        topic = self._pick_topic_by_prefix(["sensor_mag", "vehicle_magnetometer", "sensor_combined"])
        if not topic:
            return None
        df = self.get_topic_data(topic, downsample=False)
        if df is None or "timestamp" not in df.columns:
            return None

        xyz_sets = [
            ["magnetometer_ga[0]", "magnetometer_ga[1]", "magnetometer_ga[2]"],
            ["magnetic_field_ga[0]", "magnetic_field_ga[1]", "magnetic_field_ga[2]"],
            ["x", "y", "z"],
        ]
        chosen = None
        for s in xyz_sets:
            if all(c in df.columns for c in s):
                chosen = s
                break
        if not chosen:
            return None
        mag = np.sqrt((df[chosen[0]].astype(float) ** 2) + (df[chosen[1]].astype(float) ** 2) + (df[chosen[2]].astype(float) ** 2))
        out = pd.DataFrame({"timestamp": df["timestamp"], "mag_norm": mag})
        return out

    def get_thrust_and_magnetic(self, t0: float = None, t1: float = None):
        """返回推力与磁场范数。"""
        thrust_df = self._get_thrust_series()
        mag_df = self._get_magnetic_series()
        if thrust_df is None and mag_df is None:
            return pd.DataFrame(columns=["timestamp", "thrust", "mag_norm"])
        if thrust_df is None:
            out = mag_df.copy()
            out["thrust"] = np.nan
        elif mag_df is None:
            out = thrust_df.copy()
            out["mag_norm"] = np.nan
        else:
            out = pd.merge(thrust_df, mag_df, on="timestamp", how="outer").sort_values("timestamp")
        if t0 is not None:
            out = out[out["timestamp"] >= float(t0)]
        if t1 is not None:
            out = out[out["timestamp"] <= float(t1)]
        return out.replace([np.inf, -np.inf], np.nan).dropna(how="all", subset=["thrust", "mag_norm"])

    def get_flight_summary(self):
        summary = {
            "duration_s": float(self.duration),
            "arming_state_changes": [],
            "nav_state_changes": [],
            "failsafe_changes": [],
            "max_altitude_m": None,
            "max_speed_mps": None,
            "battery": None,
            "gps": None,
            "ekf": None,
        }

        lpos_topic = self._pick_first_topic(["vehicle_local_position"])
        if lpos_topic:
            lpos = self.ulog.get_dataset(lpos_topic, 0)
            if "z" in lpos.data:
                alt = -lpos.data["z"].astype(np.float64)
                summary["max_altitude_m"] = float(np.nanmax(alt))
            if all(k in lpos.data for k in ["vx", "vy", "vz"]):
                vx = lpos.data["vx"].astype(np.float64)
                vy = lpos.data["vy"].astype(np.float64)
                vz = lpos.data["vz"].astype(np.float64)
                summary["max_speed_mps"] = float(np.nanmax(np.sqrt(vx * vx + vy * vy + vz * vz)))

        status_topic = self._pick_first_topic(["vehicle_status"])
        if status_topic:
            status = self.ulog.get_dataset(status_topic, 0)
            if "timestamp" in status.data:
                t_s = self._to_seconds(status.data["timestamp"].astype(np.int64))
                if "arming_state" in status.data:
                    changes = self._unique_change_times(t_s, status.data["arming_state"])[:50]
                    summary["arming_state_changes"] = [
                        {"time_s": float(t), "arming_state": self._format_arming_info(v)}
                        for t, v in changes
                    ]
                if "nav_state" in status.data:
                    changes = self._unique_change_times(t_s, status.data["nav_state"])[:50]
                    summary["nav_state_changes"] = [
                        {"time_s": float(t), "nav_state": self._format_nav_info(v)}
                        for t, v in changes
                    ]
                if "failsafe" in status.data:
                    changes = self._unique_change_times(t_s, status.data["failsafe"])[:50]
                    summary["failsafe_changes"] = [
                        {"time_s": float(t), "failsafe": bool(v)} for t, v in changes
                    ]

        batt_topic = self._pick_first_topic(["battery_status"])
        if batt_topic:
            batt = self.ulog.get_dataset(batt_topic, 0)
            battery = {}
            if "voltage_v" in batt.data:
                battery["min_voltage_v"] = float(np.nanmin(batt.data["voltage_v"].astype(np.float64)))
            if "current_a" in batt.data:
                battery["max_current_a"] = float(np.nanmax(batt.data["current_a"].astype(np.float64)))
            if "remaining" in batt.data:
                battery["min_remaining"] = float(np.nanmin(batt.data["remaining"].astype(np.float64)))
            if battery:
                summary["battery"] = battery

        gps_topic = self._pick_first_topic(["vehicle_gps_position"])
        if gps_topic:
            gps = self.ulog.get_dataset(gps_topic, 0)
            gps_sum = {}
            if "fix_type" in gps.data:
                gps_sum["best_fix_type"] = int(np.nanmax(gps.data["fix_type"].astype(np.int64)))
            if "eph" in gps.data:
                gps_sum["best_eph_m"] = float(np.nanmin(gps.data["eph"].astype(np.float64)))
            if "epv" in gps.data:
                gps_sum["best_epv_m"] = float(np.nanmin(gps.data["epv"].astype(np.float64)))
            if gps_sum:
                summary["gps"] = gps_sum

        ekf_topic = self._pick_first_topic(["estimator_status", "ekf2_estimator_status"])
        if ekf_topic:
            ekf = self.ulog.get_dataset(ekf_topic, 0)
            ekf_sum = {"topic": ekf_topic}
            for k in [
                "pos_horiz_reset_counter",
                "pos_vert_reset_counter",
                "vel_horiz_reset_counter",
                "vel_vert_reset_counter",
                "yaw_reset_counter",
            ]:
                if k in ekf.data:
                    ekf_sum[k] = int(np.nanmax(ekf.data[k].astype(np.int64)))
            for k in ["filter_fault_flags", "health_flags"]:
                if k in ekf.data:
                    ekf_sum[f"any_{k}_nonzero"] = bool(np.nanmax(ekf.data[k].astype(np.int64)) != 0)
            summary["ekf"] = ekf_sum

        return summary

    def get_event_timeline(self, max_events=200):
        events = []

        def add_event(t, kind, detail):
            events.append({"t_s": float(t), "kind": kind, "detail": detail})

        status_topic = self._pick_first_topic(["vehicle_status"])
        if status_topic:
            st = self.ulog.get_dataset(status_topic, 0)
            if "timestamp" in st.data:
                t_s = self._to_seconds(st.data["timestamp"].astype(np.int64))
                if "arming_state" in st.data:
                    for t, v in self._unique_change_times(t_s, st.data["arming_state"]):
                        v_int = int(v)
                        add_event(
                            t,
                            "arming_state",
                            {
                                "value": v_int,
                                "state_cn": ARMING_STATE_DESC.get(v_int, f"未知状态({v_int})"),
                                "state_en": ARMING_STATE_MAP.get(v_int, f"UNKNOWN({v_int})"),
                            },
                        )
                if "nav_state" in st.data:
                    for t, v in self._unique_change_times(t_s, st.data["nav_state"]):
                        v_int = int(v)
                        add_event(
                            t,
                            "mode_change",
                            {
                                "value": v_int,
                                "mode_cn": NAV_STATE_DESC.get(v_int, f"未知模式({v_int})"),
                                "mode_en": NAV_STATE_MAP.get(v_int, f"UNKNOWN({v_int})"),
                            },
                        )
                if "failsafe" in st.data:
                    for t, v in self._unique_change_times(t_s, st.data["failsafe"]):
                        add_event(t, "failsafe", {"failsafe": bool(v)})

        ekf_topic = self._pick_first_topic(["estimator_status", "ekf2_estimator_status"])
        if ekf_topic:
            ekf = self.ulog.get_dataset(ekf_topic, 0)
            if "timestamp" in ekf.data:
                t_s = self._to_seconds(ekf.data["timestamp"].astype(np.int64))
                for k in [
                    "pos_horiz_reset_counter",
                    "pos_vert_reset_counter",
                    "vel_horiz_reset_counter",
                    "vel_vert_reset_counter",
                    "yaw_reset_counter",
                ]:
                    if k in ekf.data:
                        arr = ekf.data[k].astype(np.int64)
                        for i in range(1, len(arr)):
                            if arr[i] != arr[i - 1]:
                                add_event(float(t_s[i]), "ekf_reset", {"counter": k, "value": int(arr[i])})

        sp_topic = self._pick_first_topic(["trajectory_setpoint", "vehicle_local_position_setpoint"])
        if sp_topic:
            sp = self.ulog.get_dataset(sp_topic, 0)
            if "timestamp" in sp.data:
                t_s = self._to_seconds(sp.data["timestamp"].astype(np.int64))
                cand_fields = [f for f in ["x", "y", "z"] if f in sp.data]
                if cand_fields:
                    values = np.vstack([sp.data[f].astype(np.float64) for f in cand_fields]).T
                    for i in range(1, len(values)):
                        if np.linalg.norm(values[i] - values[i - 1]) > 0.5:
                            add_event(float(t_s[i]), "setpoint_step", {"topic": sp_topic, "fields": cand_fields})

        events_sorted = sorted(events, key=lambda e: e["t_s"])
        events_sorted = events_sorted[:max_events]
        return {"events": events_sorted, "count": len(events_sorted)}

    def list_parameters(self, prefix=None, keyword=None, max_results=500):
        params = dict(self.ulog.initial_parameters)
        out = []
        for name, value in params.items():
            if prefix and not name.startswith(prefix):
                continue
            if keyword and keyword.lower() not in name.lower():
                continue
            out.append({"name": name, "value": self._py(value)})
            if len(out) >= max_results:
                break
        return out

    def list_parameter_changes(self, limit=200):
        changes = []
        for ts_us, name, value in self.ulog.changed_parameters[:limit]:
            changes.append(
                {
                    "t_s": round((ts_us - self.ulog.start_timestamp) / 1e6, 6),
                    "name": name,
                    "value": self._py(value),
                }
            )
        return changes

    def get_topic_numeric_fields(self, topic):
        try:
            ds = self.ulog.get_dataset(topic, 0)
        except Exception:
            return []
        fields = []
        for key, arr in ds.data.items():
            if key == "timestamp":
                continue
            if np.issubdtype(arr.dtype, np.number):
                fields.append(key)
        return sorted(fields)

    def compute_field_statistics(self, topic, field, t_start_s=None, t_end_s=None):
        try:
            ds = self.ulog.get_dataset(topic, 0)
            data = ds.data.get(field)
            if data is None:
                return {"field": f"{topic}.{field}", "error": "字段不存在"}

            data = data.astype(np.float64)
            timestamps = ds.data.get("timestamp")

            if timestamps is not None and (t_start_s is not None or t_end_s is not None):
                t_s = self._to_seconds(timestamps)
                mask = np.ones(len(t_s), dtype=bool)
                if t_start_s is not None:
                    mask &= t_s >= t_start_s
                if t_end_s is not None:
                    mask &= t_s <= t_end_s
                data = data[mask]

            data = data[~np.isnan(data)]
            if len(data) == 0:
                return {"field": f"{topic}.{field}", "error": "没有有效数据"}

            q25 = np.percentile(data, 25)
            q75 = np.percentile(data, 75)
            iqr = q75 - q25
            lower = q25 - 1.5 * iqr
            upper = q75 + 1.5 * iqr
            outlier_mask = (data < lower) | (data > upper)

            return {
                "field": f"{topic}.{field}",
                "count": int(len(data)),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "median": float(np.median(data)),
                "range": float(np.max(data) - np.min(data)),
                "percentiles": {
                    "p25": float(np.percentile(data, 25)),
                    "p50": float(np.percentile(data, 50)),
                    "p75": float(np.percentile(data, 75)),
                    "p90": float(np.percentile(data, 90)),
                    "p95": float(np.percentile(data, 95)),
                    "p99": float(np.percentile(data, 99)),
                },
                "outliers": {
                    "count": int(np.sum(outlier_mask)),
                    "percentage": float(np.sum(outlier_mask) / len(data) * 100),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                },
            }
        except Exception as e:
            return {"field": f"{topic}.{field}", "error": str(e)}

    def detect_anomalies(self, topic, field, threshold_std=3.0):
        try:
            ds = self.ulog.get_dataset(topic, 0)
            data = ds.data.get(field)
            if data is None:
                return {"field": f"{topic}.{field}", "error": "字段不存在"}

            data = data.astype(np.float64)
            data = data[~np.isnan(data)]
            if len(data) == 0:
                return {"field": f"{topic}.{field}", "error": "没有有效数据"}

            mean = np.mean(data)
            std = np.std(data)
            lower = mean - threshold_std * std
            upper = mean + threshold_std * std
            mask = (data < lower) | (data > upper)
            idx = np.where(mask)[0]

            anomalies = []
            for i in idx[:20]:
                value = data[i]
                deviation = (value - mean) / std if std > 0 else 0.0
                anomalies.append(
                    {
                        "index": int(i),
                        "value": float(value),
                        "deviation_sigma": float(deviation),
                        "type": "high" if value > mean else "low",
                    }
                )

            return {
                "field": f"{topic}.{field}",
                "threshold_std": float(threshold_std),
                "mean": float(mean),
                "std": float(std),
                "bounds": {"lower": float(lower), "upper": float(upper)},
                "anomaly_count": int(np.sum(mask)),
                "anomaly_percentage": float(np.sum(mask) / len(data) * 100),
                "anomalies": anomalies,
            }
        except Exception as e:
            return {"field": f"{topic}.{field}", "error": str(e)}
