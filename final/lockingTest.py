

import telemetry
from pymavlink import mavutil
from pathlib import Path
from ultralytics import YOLO
import controllers
import os
import mathHelpers
import time
import math
import cv2
import numpy as np
import requests
import threading
import socket
import sys
import queue
from urllib.parse import urlparse

VALID_MISSION_MODES = {"enemy", "qr", "normal"}
CURRENT_MISSION_MODE = os.getenv("IHA_MISSION_MODE", "normal").strip().lower()
if CURRENT_MISSION_MODE not in VALID_MISSION_MODES:
    CURRENT_MISSION_MODE = "normal"
BASE_URL = os.getenv("IHA_BASE_URL", "http://192.168.10.2:10001")
USERNAME = os.getenv("IHA_USERNAME", "4")
PASSWORD = os.getenv("IHA_PASSWORD", "4")
TEAM_NO = int(os.getenv("IHA_TEAM_NO", "4"))
MAVLINK_URL = os.getenv("IHA_MAVLINK_URL", "udpin:192.168.10.1:14580")
QR_DEBUG = os.getenv("IHA_QR_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")
ENABLE_STDIN_MISSION_INPUT = os.getenv("IHA_STDIN_MISSION_INPUT", "1").strip().lower() not in ("0", "false", "no", "off")

## GLOBAL VARIABLES ##
FLIGHT_BOUNDARIES = [
    (38.70463973, 27.45086697),
    (38.70462329, 27.45996806),
    (38.69837561, 27.45758745),
    (38.69842494, 27.44964507),
]

AXIS_BOUNDS = {
    'pitch': (-35.0, 35.0),
    'roll': (-45.0, 45.0),
    'yaw': (-180.0, 180.0),
    'alt': (50.0, 300.0),
    'speed': (13, 30)
}

MODEL_PATH = Path.home() / "Desktop" / "best.pt"

TAKEOFF_ALT_TARGET = 50.0
TAKEOFF_ALT_THRESH = 5.0
DX_CONST = 0.25
DT_MIN = 0.01
PREV_MEAS_RATE_CONST = 0.75
CONF = 0.25
IMG_SIZE = 640
FOV_X_DEG = 80.0
FOV_Y_DEG = 60.0
QR_APPROACH_ALT_M = 110.0
QR_DIVE_START_DIST_M = 170.0
QR_DIVE_MIN_ALT_M = 100.0
QR_DIVE_ABORT_ALT_M = 45.0
QR_DIVE_ALIGN_YAW_ERR_DEG = 12.0
QR_DIVE_PITCH_DEG = -35.0
QR_VISION_YAW_GAIN = 0.85
QR_STATUS_INTERVAL = 0.5
PREARM_CONST = mavutil.mavlink.MAV_SYS_STATUS_PREARM_CHECK
MAX_DISTANCE_BETWEEN_ENEMY = 500.0
AUTONOMOUS_FLIGHT_STATUS = 1
FOLLOW_DISTANCE_M = 25.0
FOLLOW_BASE_SPEED = 18.0
FOLLOW_SPEED_GAIN = 0.45
TARGET_AREA_RATIO = 0.045
TRACKER_MAX_AGE = 3.0
VISION_HOLD_SECONDS = 1.0
STATUS_INTERVAL = 1.0
VISION_YAW_GAIN = 0.85
VISION_X_DEADBAND = 0.01
VISION_DX_DAMPING_TIME = 0.04
VISION_MIN_DISTANCE_M = 8.0
VISION_TRAIL_RECOVERY_DISTANCE_M = 10.0
VISION_CLOSE_SPEED_MARGIN = 1.5
LOCK_AREA_LEFT_RATIO = 0.25
LOCK_AREA_RIGHT_RATIO = 0.75
LOCK_AREA_TOP_RATIO = 0.10
LOCK_AREA_BOTTOM_RATIO = 0.90
LOCK_MIN_WIDTH_RATIO = 0.05
LOCK_MIN_HEIGHT_RATIO = 0.05
LOCK_TARGET_SIZE_RATIO = 0.06
LOCK_INSIDE_CENTERING_GAIN = 0.50
LOCK_REQUIRED_SECONDS = 4.0
LOCK_TOLERANCE_SECONDS = 1.0
VISION_DYNAMIC_FOLLOW_MIN_M = 18.0
VISION_DYNAMIC_FOLLOW_MAX_M = 35.0
VISION_PITCH_GAIN = 0.55
VISION_PITCH_OFFSET_LIMIT_DEG = 10.0
VISION_DIVE_GUARD_DISTANCE_M = 35.0
VISION_CLOSE_DIVE_LIMIT_DEG = 6.0
VISION_PITCH_HOLD_SECONDS = 0.35
VISION_PITCH_CMD_RATE_DPS = 18.0
GPS_REACQUIRE_PITCH_LIMIT_DEG = 30.0
GPS_PITCH_CMD_RATE_DPS = 15.0
ALTITUDE_DEADBAND_M = 3.0
GUIDED_TRIM_THROTTLE = 0.60
GUIDED_MIN_THROTTLE = 0.25
GUIDED_MAX_THROTTLE = 1.00
GUIDED_THROTTLE_RATE = 0.80
session = requests.Session()
hss_list = []


## HELPERS ##

connection = mavutil.mavlink_connection(MAVLINK_URL)
connection.wait_heartbeat()
print(f"Connected to Fixed-Wing Vehicle via {MAVLINK_URL}...")

connection.mav.request_data_stream_send(
    connection.target_system,
    connection.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    20,
    1
)


class TargetTelemetry:
    SERVER_FIELDS = {
        'iha_enlem': 'lat',
        'iha_boylam': 'lon',
        'iha_irtifa': 'alt',
        'iha_dikilme': 'pitch',
        'iha_yonelme': 'yaw',
        'iha_yatis': 'roll',
        'iha_hizi': 'speed',
        'gps_saati': 'gps_time',
    }

    def __init__(self):
        for attr in self.SERVER_FIELDS.values():
            setattr(self, attr, None)
        self.team_no = None

    def update_from_server(self, enemy):
        if not enemy:
            return False
        self.team_no = enemy.get('takim_numarasi')
        for field, attr in self.SERVER_FIELDS.items():
            value = enemy.get(field)
            if field == 'gps_saati':
                setattr(self, attr, value)
                continue
            try:
                value = None if value is None else float(value)
            except (TypeError, ValueError):
                value = None
            setattr(self, attr, value)
        return self.has_position()

    def has_position(self):
        return self.lat is not None and self.lon is not None

    def has_reacquire_data(self):
        return self.has_position() and self.alt is not None and self.yaw is not None


def _time_boot_ms():
    return int((time.monotonic() * 1000.0) % 4294967295)


def _param_name(param_id):
    if isinstance(param_id, bytes):
        return param_id.split(b'\x00', 1)[0].decode('ascii', errors='ignore')
    return str(param_id).split('\x00', 1)[0]


def set_float_param(name, value, retries=3):
    encoded_name = name.encode('ascii')
    for _ in range(retries):
        connection.mav.param_set_send(
            connection.target_system,
            connection.target_component,
            encoded_name,
            float(value),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )

        deadline = time.time() + 1.0
        while time.time() < deadline:
            msg = connection.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.2)
            if msg is None or _param_name(msg.param_id) != name:
                continue
            print(f"[Param] {name} = {msg.param_value:.1f}")
            return True

    print(f"[Param] WARNING: {name} could not be set to {value}")
    return False


def configure_speed_limits():
    set_float_param('AIRSPEED_MAX', AXIS_BOUNDS['speed'][1])


def _euler_to_quaternion(roll_deg, pitch_deg, yaw_deg):
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _attitude_type_mask(include_pitch, include_throttle=False):
    mask = (
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE
        | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE
        | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
    )
    if not include_pitch:
        mask |= mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
    if not include_throttle:
        mask |= mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_THROTTLE_IGNORE
    return mask


def send_guided_heading(heading_deg):
    roll_limit = AXIS_BOUNDS['roll'][1]
    heading_accel = math.tan(math.radians(roll_limit)) * 9.80665
    connection.mav.command_int_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL,
        mavutil.mavlink.MAV_CMD_GUIDED_CHANGE_HEADING,
        0,
        0,
        1,
        heading_deg % 360.0,
        heading_accel,
        0,
        0,
        0,
        0,
    )


def send_guided_speed(speed):
    if speed is None:
        return
    speed = mathHelpers.clamp(speed, AXIS_BOUNDS['speed'][0], AXIS_BOUNDS['speed'][1])
    connection.mav.command_int_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL,
        mavutil.mavlink.MAV_CMD_GUIDED_CHANGE_SPEED,
        0,
        0,
        0,
        speed,
        0,
        0,
        0,
        0,
        0,
    )


def send_guided_altitude(alt_m):
    if alt_m is None:
        return
    alt_m = mathHelpers.clamp(alt_m, AXIS_BOUNDS['alt'][0], AXIS_BOUNDS['alt'][1])
    connection.mav.command_int_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        mavutil.mavlink.MAV_CMD_GUIDED_CHANGE_ALTITUDE,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        float(alt_m),
    )


def send_guided_attitude(current_roll, target_pitch, current_yaw, throttle=None):
    include_pitch = target_pitch is not None
    include_throttle = throttle is not None
    if not include_pitch and not include_throttle:
        return

    pitch = 0.0 if target_pitch is None else target_pitch
    thrust = 0.0 if throttle is None else mathHelpers.clamp(
        throttle,
        GUIDED_MIN_THROTTLE,
        GUIDED_MAX_THROTTLE,
    )
    q = _euler_to_quaternion(current_roll, pitch, current_yaw)
    connection.mav.set_attitude_target_send(
        _time_boot_ms(),
        connection.target_system,
        connection.target_component,
        _attitude_type_mask(include_pitch, include_throttle),
        q,
        0.0,
        0.0,
        0.0,
        thrust,
    )


def input_thread_func():
    global CURRENT_MISSION_MODE
    if not ENABLE_STDIN_MISSION_INPUT:
        print(f"[MISSION] Stdin input disabled. Initial mode: {CURRENT_MISSION_MODE}")
        return
    print(f"[MISSION] Initial mode: {CURRENT_MISSION_MODE}. Type enemy, qr or normal then Enter.")
    while True:
        try:
            val = input("> ").strip().lower()
            if val in VALID_MISSION_MODES:
                CURRENT_MISSION_MODE = val
                print(f"[MISSION] Mode switched to: {CURRENT_MISSION_MODE}")
            else:
                print(f"[MISSION] Invalid mode '{val}'. Use 'enemy', 'qr' or 'normal'.")
        except:
            break


# =========================
# GUI: 3 Mission Button + Mission/Telemetry Log Panes
# =========================
USE_GUI = os.getenv("IHA_USE_GUI", "1").strip().lower() not in ("0", "false", "no", "off")

TELEMETRY_LOG_KEYWORDS = ("TELEMETRY STATUS", "LOCK STATUS", "KAMIKAZE STATUS", "LOGIN STATUS", "LOGIN RESP", "SEND QR")


class LogRouter:
    """Replace sys.stdout. Routes lines to either the mission queue or the
    telemetry queue based on keyword. Always echoes to original stdout too."""

    def __init__(self, original):
        self.original = original
        self.mission_q = queue.Queue()
        self.telemetry_q = queue.Queue()
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, text):
        try:
            self.original.write(text)
            self.original.flush()
        except Exception:
            pass
        with self._lock:
            self._buffer += text
            while '\n' in self._buffer:
                line, self._buffer = self._buffer.split('\n', 1)
                self._route(line)

    def _route(self, line):
        if not line.strip():
            return
        if any(k in line for k in TELEMETRY_LOG_KEYWORDS):
            self.telemetry_q.put(line)
        else:
            self.mission_q.put(line)

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass


class MissionUI:
    """Tkinter window: 3 buttons + 2 log panes. Runs on the main thread."""

    def __init__(self, router):
        import tkinter as tk
        from tkinter import scrolledtext

        self._tk = tk
        self.router = router
        self.root = tk.Tk()
        self.root.title("IHA Mission Control")
        self.root.geometry("1100x650")
        self.root.configure(bg="#1e1e1e")

        # --- Top bar: buttons + status ---
        top = tk.Frame(self.root, bg="#1e1e1e")
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        btn_specs = [
            ("QR", "qr", "#2e7d32"),
            ("ENEMY", "enemy", "#c62828"),
            ("NORMAL", "normal", "#1565c0"),
        ]
        for label, mode, color in btn_specs:
            b = tk.Button(
                top, text=label, command=lambda m=mode: self.set_mode(m),
                bg=color, fg="white", activebackground=color,
                font=("Arial", 14, "bold"), width=10, height=2,
                relief=tk.FLAT, bd=0, padx=8, pady=4,
            )
            b.pack(side=tk.LEFT, padx=6)

        self.mode_label = tk.Label(
            top, text=f"Mode: {CURRENT_MISSION_MODE.upper()}",
            font=("Arial", 14, "bold"), fg="#ffd54f", bg="#1e1e1e",
        )
        self.mode_label.pack(side=tk.LEFT, padx=24)

        # --- Two text panes side by side ---
        body = tk.Frame(self.root, bg="#1e1e1e")
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        left = tk.LabelFrame(body, text="Mission / Attitude", fg="#fff", bg="#1e1e1e",
                              font=("Arial", 10, "bold"))
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self.mission_text = scrolledtext.ScrolledText(
            left, bg="#0d1117", fg="#c9f5c0", insertbackground="#fff",
            font=("Courier", 9), wrap=tk.NONE, state=tk.DISABLED,
        )
        self.mission_text.pack(fill=tk.BOTH, expand=True)

        right = tk.LabelFrame(body, text="Server Telemetry", fg="#fff", bg="#1e1e1e",
                               font=("Arial", 10, "bold"))
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4)
        self.telemetry_text = scrolledtext.ScrolledText(
            right, bg="#0d1117", fg="#ffcb8b", insertbackground="#fff",
            font=("Courier", 9), wrap=tk.NONE, state=tk.DISABLED,
        )
        self.telemetry_text.pack(fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_queues)

    def set_mode(self, mode):
        global CURRENT_MISSION_MODE
        if mode not in VALID_MISSION_MODES:
            return
        CURRENT_MISSION_MODE = mode
        self.mode_label.config(text=f"Mode: {mode.upper()}")
        print(f"[MISSION] Mode switched to: {mode} (UI button)")

    def _append(self, widget, line):
        widget.configure(state=self._tk.NORMAL)
        widget.insert(self._tk.END, line + "\n")
        widget.see(self._tk.END)
        line_count = int(widget.index("end-1c").split(".")[0])
        if line_count > 1500:
            widget.delete("1.0", f"{line_count - 1500}.0")
        widget.configure(state=self._tk.DISABLED)

    def _drain(self, q, widget):
        drained = 0
        while drained < 200:
            try:
                line = q.get_nowait()
            except queue.Empty:
                break
            self._append(widget, line)
            drained += 1

    def _poll_queues(self):
        self._drain(self.router.mission_q, self.mission_text)
        self._drain(self.router.telemetry_q, self.telemetry_text)
        try:
            if CURRENT_MISSION_MODE.upper() not in self.mode_label.cget("text"):
                self.mode_label.config(text=f"Mode: {CURRENT_MISSION_MODE.upper()}")
        except Exception:
            pass
        self.root.after(100, self._poll_queues)

    def _on_close(self):
        try:
            self.root.destroy()
        except Exception:
            pass
        os._exit(0)

    def run(self):
        self.root.mainloop()


def wait_for_prearm():
    print("Waiting for pre-arm...")
    while True:
        msg = connection.recv_match(type='SYS_STATUS', blocking=True)

        if msg.onboard_control_sensors_health & PREARM_CONST == PREARM_CONST:
            print("Pre-arm good...")
            break


def auto_and_arm():
    print("Setting Mode to TAKEOFF...")
    connection.set_mode('TAKEOFF')

    print("Waiting for EKF alignment and arming...")
    while True:
        connection.mav.command_long_send(
            connection.target_system, connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
        )

        msg = connection.recv_match(type='HEARTBEAT', blocking=True, timeout=1.0)

        if msg and (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            print("Plane is successfully Armed!")
            break

        time.sleep(1.0)


def wait_for_takeoff():
    print(f"[Takeoff] Waiting for plane to climb ? {TAKEOFF_ALT_TARGET - TAKEOFF_ALT_THRESH:.0f} m ...")
    last_print = 0.0
    while True:
        msg = connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1.0)
        if msg is None: continue

        alt_m = msg.relative_alt / 1000.0
        telemetry_data['alt'] = alt_m
        telemetry_data['lat'] = msg.lat / 1e7
        telemetry_data['lon'] = msg.lon / 1e7
        now = time.time()
        if now - last_print >= 2.0:
            print(f"[Takeoff]   alt = {alt_m:.1f} m  (target {TAKEOFF_ALT_TARGET:.0f} m)")
            last_print = now

        if alt_m >= TAKEOFF_ALT_TARGET - TAKEOFF_ALT_THRESH:
            print(f"[Takeoff] Target altitude reached. Switching to GUIDED setpoints ...")
            break


# =========================
# RTP/H264 UDP GİRİŞ
# =========================
UDP_IN_PORT = int(os.getenv("IHA_CAMERA_IN_PORT", "5600"))

# =========================
# ÇIKIŞ JPEG UDP
# =========================
STREAM_PORTS = {
    20: 5420,
    1: 5425,
    2: 5426,
    3: 5427,
    4: 5428,
    5: 5429,
}


def _base_host(base_url):
    return urlparse(base_url).hostname or "127.0.0.1"


UDP_OUT_IP = os.getenv("IHA_VIDEO_HOST", _base_host(BASE_URL))
UDP_OUT_PORT = int(os.getenv("IHA_VIDEO_PORT", str(STREAM_PORTS.get(TEAM_NO, 5400 + TEAM_NO))))

# =========================
# Görüntü boyutu
# =========================
WIDTH = 1920
HEIGHT = 1080
JPEG_QUALITY = 50

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

UDP_IN_ADDRESS = os.getenv("IHA_CAMERA_IN_ADDRESS", "127.0.0.1")
GZ_WORLD_NAME = os.getenv("IHA_GZ_WORLD", "runway")
GZ_MODEL_NAME = os.getenv("IHA_GZ_MODEL", "observer")
ENABLE_GZ_TOPIC = os.getenv("IHA_ENABLE_GAZEBO_TOPIC", "0").strip().lower() in ("1", "true", "yes", "on")


def enable_gazebo_camera():
    if not ENABLE_GZ_TOPIC:
        return
    topic = f"/world/{GZ_WORLD_NAME}/model/{GZ_MODEL_NAME}/link/base_link/sensor/nose_camera/image/enable_streaming"
    print(f"[Gazebo] Sending 'enable' signal to {topic}")
    os.system(f'gz topic -t {topic} -m gz.msgs.Boolean -p "data: true"')


def build_gst_pipeline():
    return (
        f"udpsrc port={UDP_IN_PORT} address={UDP_IN_ADDRESS} ! "
        "application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96 ! "
        "rtph264depay ! avdec_h264 ! videoconvert ! videoscale ! "
        f"video/x-raw, format=BGR, width={WIDTH}, height={HEIGHT} ! "
        "appsink drop=true sync=false max-buffers=1"
    )


def start_camera_capture():
    enable_gazebo_camera()
    pipeline = build_gst_pipeline()
    print("[GStreamer] Pipeline başlatılıyor:")
    print(pipeline)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cap


def make_tracker():
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
        return cv2.legacy.TrackerKCF_create()
    return None


def xyxy_to_tracker_box(x1, y1, x2, y2, frame_w, frame_h):
    x1 = mathHelpers.clamp(float(x1), 0.0, frame_w - 1.0)
    y1 = mathHelpers.clamp(float(y1), 0.0, frame_h - 1.0)
    x2 = mathHelpers.clamp(float(x2), x1 + 1.0, frame_w)
    y2 = mathHelpers.clamp(float(y2), y1 + 1.0, frame_h)
    return (x1, y1, x2 - x1, y2 - y1)


def tracker_box_to_xyxy(box):
    x, y, bw, bh = box
    return x, y, x + bw, y + bh


def lock_area_bounds(frame_w, frame_h):
    return (
        int(frame_w * LOCK_AREA_LEFT_RATIO),
        int(frame_h * LOCK_AREA_TOP_RATIO),
        int(frame_w * LOCK_AREA_RIGHT_RATIO),
        int(frame_h * LOCK_AREA_BOTTOM_RATIO),
    )


def lock_status(obj_cx, obj_cy, box_w, box_h, frame_w, frame_h):
    left, top, right, bottom = lock_area_bounds(frame_w, frame_h)
    in_x = left <= obj_cx <= right
    in_y = top <= obj_cy <= bottom
    center_inside = in_x and in_y
    size_ok = (
        box_w >= frame_w * LOCK_MIN_WIDTH_RATIO and
        box_h >= frame_h * LOCK_MIN_HEIGHT_RATIO
    )
    return center_inside, size_ok, in_x, in_y


def lock_guidance_error(dx_norm, dy_norm, in_x, in_y):
    guide_dx = dx_norm * LOCK_INSIDE_CENTERING_GAIN if in_x else dx_norm
    guide_dy = dy_norm * LOCK_INSIDE_CENTERING_GAIN if in_y else dy_norm
    return guide_dx, guide_dy


def speed_from_distance(current_lat, current_lon, target_lat, target_lon):
    if None in (current_lat, current_lon, target_lat, target_lon):
        return None

    dist = mathHelpers.get_distance(current_lat, current_lon, target_lat, target_lon)
    dist_error = dist - FOLLOW_DISTANCE_M
    desired_speed = mathHelpers.clamp(
        FOLLOW_BASE_SPEED + (dist_error * FOLLOW_SPEED_GAIN),
        AXIS_BOUNDS['speed'][0],
        AXIS_BOUNDS['speed'][1],
    )
    return desired_speed, dist


def dynamic_follow_distance_from_box(box_w_ratio, box_h_ratio, current_dist):
    visible_size = min(box_w_ratio, box_h_ratio)
    if current_dist is None or visible_size <= 0.0:
        return FOLLOW_DISTANCE_M

    desired_distance = current_dist * (visible_size / LOCK_TARGET_SIZE_RATIO)
    return mathHelpers.clamp(
        desired_distance,
        VISION_DYNAMIC_FOLLOW_MIN_M,
        VISION_DYNAMIC_FOLLOW_MAX_M,
    )


def vision_speed_from_distance(current_lat, current_lon, target_lat, target_lon, target_speed, box_w_ratio, box_h_ratio):
    if None in (current_lat, current_lon, target_lat, target_lon):
        return None

    dist = mathHelpers.get_distance(current_lat, current_lon, target_lat, target_lon)
    follow_distance = dynamic_follow_distance_from_box(box_w_ratio, box_h_ratio, dist)
    dist_error = dist - follow_distance
    low, high = AXIS_BOUNDS['speed']
    target_speed_cmd = None
    if target_speed is not None:
        target_speed_cmd = mathHelpers.clamp(target_speed, low, high)

    base_speed = target_speed_cmd if target_speed_cmd is not None else FOLLOW_BASE_SPEED
    desired_speed = base_speed + (dist_error * FOLLOW_SPEED_GAIN)

    if dist <= VISION_MIN_DISTANCE_M:
        if target_speed_cmd is not None:
            close_speed = mathHelpers.clamp(target_speed_cmd - VISION_CLOSE_SPEED_MARGIN, low, high)
            return close_speed, dist, follow_distance
        return low, dist, follow_distance

    if target_speed_cmd is not None and dist <= follow_distance:
        close_speed = mathHelpers.clamp(target_speed_cmd - VISION_CLOSE_SPEED_MARGIN, low, high)
        blend = (dist - VISION_MIN_DISTANCE_M) / (follow_distance - VISION_MIN_DISTANCE_M)
        blend = mathHelpers.clamp(blend, 0.0, 1.0)
        desired_speed = close_speed + (blend * (target_speed_cmd - close_speed))
        return mathHelpers.clamp(desired_speed, low, high), dist, follow_distance

    if target_speed_cmd is not None:
        desired_speed = max(desired_speed, target_speed_cmd)

    return mathHelpers.clamp(desired_speed, low, high), dist, follow_distance


def target_trail_yaw(current_lat, current_lon, target_lat, target_lon, target_yaw, trail_distance=FOLLOW_DISTANCE_M):
    if None in (current_lat, current_lon, target_lat, target_lon, target_yaw):
        return None

    behind_bearing = (target_yaw + 180.0) % 360.0
    behind_lat, behind_lon = mathHelpers.destination_point(
        target_lat,
        target_lon,
        behind_bearing,
        trail_distance,
    )
    return mathHelpers.get_bearing(current_lat, current_lon, behind_lat, behind_lon)


def vision_pitch_from_target(smoothed_dy, target_pitch, dist):
    low, high = AXIS_BOUNDS['pitch']
    pitch_reference = target_pitch if target_pitch is not None else 0.0
    pitch_reference = mathHelpers.clamp(pitch_reference, low, high)
    pitch_offset = -smoothed_dy * (FOV_Y_DEG / 2.0) * VISION_PITCH_GAIN
    pitch_offset = mathHelpers.clamp(
        pitch_offset,
        -VISION_PITCH_OFFSET_LIMIT_DEG,
        VISION_PITCH_OFFSET_LIMIT_DEG,
    )
    desired_pitch = pitch_reference + pitch_offset

    if dist is not None and dist <= VISION_DIVE_GUARD_DISTANCE_M:
        min_pitch = mathHelpers.clamp(
            pitch_reference - VISION_CLOSE_DIVE_LIMIT_DEG,
            low,
            high,
        )
        desired_pitch = max(desired_pitch, min_pitch)

    return mathHelpers.clamp(desired_pitch, low, high)


def rate_limit_value(current_value, desired_value, max_rate, dt):
    step_limit = abs(max_rate) * max(dt, DT_MIN)
    return current_value + mathHelpers.clamp(
        desired_value - current_value,
        -step_limit,
        step_limit,
    )


def draw_minimap(current_lat, current_lon, current_yaw, hss_list, flight_boundaries, target_lat=None, target_lon=None, qr_lat=None, qr_lon=None):
    map_w, map_h = 600, 600
    map_img = np.ones((map_h, map_w, 3), dtype=np.uint8) * 30
    
    cx, cy = map_w // 2, map_h // 2
    
    if current_lat is None or current_lon is None:
        cv2.putText(map_img, "Waiting for GPS...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return map_img
        
    meter_per_deg_lat = 111320.0
    meter_per_deg_lon = 111320.0 * math.cos(math.radians(current_lat))
    pixels_per_meter = 1.5
    
    if flight_boundaries and len(flight_boundaries) >= 3:
        for i in range(len(flight_boundaries)):
            lat1, lon1 = flight_boundaries[i]
            lat2, lon2 = flight_boundaries[(i + 1) % len(flight_boundaries)]
            
            dy_m1 = (lat1 - current_lat) * meter_per_deg_lat
            dx_m1 = (lon1 - current_lon) * meter_per_deg_lon
            
            dy_m2 = (lat2 - current_lat) * meter_per_deg_lat
            dx_m2 = (lon2 - current_lon) * meter_per_deg_lon
            
            px1 = int(cx + dx_m1 * pixels_per_meter)
            py1 = int(cy - dy_m1 * pixels_per_meter)
            
            px2 = int(cx + dx_m2 * pixels_per_meter)
            py2 = int(cy - dy_m2 * pixels_per_meter)
            
            cv2.line(map_img, (px1, py1), (px2, py2), (0, 165, 255), 2)
            
    for hss in hss_list:
        h_lat = hss.get("hssEnlem", 0.0)
        h_lon = hss.get("hssBoylam", 0.0)
        h_rad = hss.get("hssYaricap", 0.0)
        
        dy_m = (h_lat - current_lat) * meter_per_deg_lat
        dx_m = (h_lon - current_lon) * meter_per_deg_lon
        
        if abs(dx_m) > 10000 or abs(dy_m) > 10000:
            continue
            
        px = int(cx + dx_m * pixels_per_meter)
        py = int(cy - dy_m * pixels_per_meter)
        r_px = int(h_rad * pixels_per_meter)
        
        cv2.circle(map_img, (px, py), r_px, (0, 0, 255), 2)
        cv2.drawMarker(map_img, (px, py), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
        cv2.putText(map_img, f"HSS R:{int(h_rad)}m", (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    if target_lat is not None and target_lon is not None:
        dy_m = (target_lat - current_lat) * meter_per_deg_lat
        dx_m = (target_lon - current_lon) * meter_per_deg_lon
        
        if abs(dx_m) < 10000 and abs(dy_m) < 10000:
            px = int(cx + dx_m * pixels_per_meter)
            py = int(cy - dy_m * pixels_per_meter)
            cv2.circle(map_img, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(map_img, "Target", (px+10, py+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if qr_lat is not None and qr_lon is not None:
        dy_m = (qr_lat - current_lat) * meter_per_deg_lat
        dx_m = (qr_lon - current_lon) * meter_per_deg_lon
        
        if abs(dx_m) < 10000 and abs(dy_m) < 10000:
            px = int(cx + dx_m * pixels_per_meter)
            py = int(cy - dy_m * pixels_per_meter)
            cv2.circle(map_img, (px, py), 6, (255, 0, 255), -1)
            cv2.putText(map_img, "QR", (px+10, py+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    yaw_rad = math.radians(current_yaw if current_yaw else 0.0)
    dx_drone = math.sin(yaw_rad) * 20
    dy_drone = -math.cos(yaw_rad) * 20
    cv2.circle(map_img, (cx, cy), 6, (255, 255, 0), -1)
    cv2.line(map_img, (cx, cy), (int(cx+dx_drone), int(cy+dy_drone)), (255, 255, 255), 2)
    cv2.putText(map_img, "US", (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.line(map_img, (10, map_h-20), (10 + int(100 * pixels_per_meter), map_h-20), (255, 255, 255), 2)
    cv2.putText(map_img, "100m", (10, map_h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return map_img


## CONTROLLER SETUP & MAIN LOOP

def make_controllers() -> dict:
    return {
        'altitude': controllers.HybridController(
            pid_ctrl=controllers.PIDController(kp=0.65, ki=0.08, kd=0.04, integral_limit=60.0, output_limit=18.0,
                                               integral_zone=35.0, rate_filter_tau=0.18),
            fuzzy_ctrl=controllers.FuzzyGainScheduler(error_range=40.0, rate_range=8.0),
        ),
        'speed': controllers.HybridController(
            pid_ctrl=controllers.PIDController(kp=0.07, ki=0.03, kd=0.01, integral_limit=10.0, output_limit=0.35,
                                               integral_zone=12.0, rate_filter_tau=0.20),
            fuzzy_ctrl=controllers.FuzzyGainScheduler(error_range=12.0, rate_range=6.0),
        ),
    }

telemetry_data = {
    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
    'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0,
    'alt': 0.0, 'lat': None, 'lon': None,
    'spd': 0.0, 'battery': 100.0
}

def mavlink_reader_thread():
    global telemetry_data
    while True:
        msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD', 'SYS_STATUS'], blocking=True)
        if msg is not None:
            msg_type = msg.get_type()
            if msg_type == 'ATTITUDE':
                telemetry_data['roll'] = math.degrees(msg.roll)
                telemetry_data['pitch'] = math.degrees(msg.pitch)
                telemetry_data['yaw'] = math.degrees(msg.yaw)
                telemetry_data['roll_rate'] = math.degrees(getattr(msg, 'rollspeed', 0.0))
                telemetry_data['pitch_rate'] = math.degrees(getattr(msg, 'pitchspeed', 0.0))
                telemetry_data['yaw_rate'] = math.degrees(getattr(msg, 'yawspeed', 0.0))
            elif msg_type == 'GLOBAL_POSITION_INT':
                telemetry_data['alt'] = msg.relative_alt / 1000.0
                telemetry_data['lat'] = msg.lat / 1e7
                telemetry_data['lon'] = msg.lon / 1e7
            elif msg_type == 'VFR_HUD':
                telemetry_data['spd'] = msg.airspeed
            elif msg_type == 'SYS_STATUS':
                telemetry_data['battery'] = msg.battery_remaining

closest_enemy = None

def fetch_hss_thread():
    global hss_list
    while True:
        try:
            hss_data = telemetry.get_hss(session, BASE_URL, token)
            if hss_data:
                hss_list = hss_data
        except Exception as e:
            pass
        time.sleep(60.0)

class CameraThread:
    def __init__(self, factory):
        self.factory = factory
        self.cap = None
        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def _open(self):
        try:
            cap = self.factory()
        except Exception as e:
            print(f"[Camera] Pipeline factory error: {e}")
            return None
        if cap is None or not cap.isOpened():
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            return None
        return cap

    def update(self):
        retry_count = 0
        no_frame_count = 0
        while self.running:
            if self.cap is None:
                cap = self._open()
                if cap is None:
                    retry_count += 1
                    print(f"[Camera] GStreamer açılamadı, tekrar deneme #{retry_count} (2sn)...")
                    time.sleep(2.0)
                    continue
                self.cap = cap
                retry_count = 0
                no_frame_count = 0
                print("[Camera] GStreamer pipeline açıldı.")

            ok, frame = self.cap.read()
            if not ok or frame is None:
                no_frame_count += 1
                if no_frame_count >= 50:
                    print("[Camera] Frame alınamıyor, pipeline yeniden açılacak...")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    no_frame_count = 0
                    continue
                time.sleep(0.02)
                continue

            no_frame_count = 0
            with self.lock:
                self.ret = True
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def release(self):
        self.running = False
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


_dispatch_state = {'kilit': 0, 'hx': 0, 'hy': 0, 'hw': 0, 'hh': 0}
_dispatch_state_lock = threading.Lock()


def update_dispatch_state(kilit=0, hx=0, hy=0, hw=0, hh=0):
    with _dispatch_state_lock:
        _dispatch_state['kilit'] = kilit
        _dispatch_state['hx'] = hx
        _dispatch_state['hy'] = hy
        _dispatch_state['hw'] = hw
        _dispatch_state['hh'] = hh


def telemetry_sender_thread():
    last_send = 0.0
    while True:
        try:
            now = time.time()
            if now - last_send >= 0.5:
                with _dispatch_state_lock:
                    state = dict(_dispatch_state)
                dispatch_current_telemetry(**state)
                last_send = now
        except Exception as e:
            print("[Telemetry Thread] error:", e)
        time.sleep(0.05)


def dispatch_current_telemetry(kilit=0, hx=0, hy=0, hw=0, hh=0):
    if telemetry_data['lat'] is None or telemetry_data['lon'] is None:
        return
    try:
        token_value = token
    except NameError:
        return

    telemetry_args = {
        'session': session,
        'base_url': BASE_URL,
        'token': token_value,
        'team_no': TEAM_NO,
        'lat': max(-90.0, min(90.0, float(telemetry_data['lat']))),
        'lon': max(-180.0, min(180.0, float(telemetry_data['lon']))),
        'alt': max(0.0, min(10000.0, float(telemetry_data['alt']))),
        'pitch': max(-90.0, min(90.0, float(telemetry_data['pitch']))),
        'yaw': float(telemetry_data['yaw']) % 360,
        'roll': max(-90.0, min(90.0, float(telemetry_data['roll']))),
        'spd': max(0.0, min(200.0, float(telemetry_data['spd']))),
        'battery': max(0, min(100, int(telemetry_data['battery']))),
        'otonom': AUTONOMOUS_FLIGHT_STATUS,
        'gps_time': telemetry.now_clock(),
        'kilit': kilit,
        'hx': hx,
        'hy': hy,
        'hw': hw,
        'hh': hh,
    }
    threading.Thread(target=async_send_telemetry, args=(telemetry_args,), daemon=True).start()


def async_send_telemetry(kwargs):
    global closest_enemy
    try:
        resp = telemetry.send_telemetry(
            session=kwargs['session'], base_url=kwargs['base_url'], token=kwargs['token'],
            team_no=kwargs['team_no'], iha_enlem=kwargs['lat'], iha_boylam=kwargs['lon'],
            iha_irtifa=kwargs['alt'], iha_dikilme=kwargs['pitch'], iha_yonelme=kwargs['yaw'],
            iha_yatis=kwargs['roll'], iha_hiz=kwargs['spd'], iha_batarya=kwargs['battery'],
            iha_otonom=kwargs['otonom'], gps_saati=kwargs['gps_time'], iha_kilitlenme=kwargs['kilit'],
            hedef_merkez_X=kwargs['hx'], hedef_merkez_Y=kwargs['hy'], 
            hedef_genislik=kwargs['hw'], hedef_yukseklik=kwargs['hh']
        )
        if resp.status_code == 200:
            enemies = resp.json().get("konumBilgileri", [])
            closest_enemy = mathHelpers.find_closest_target(
                kwargs['lat'], kwargs['lon'], enemies, MAX_DISTANCE_BETWEEN_ENEMY, kwargs['team_no']
            )
    except Exception as e:
        print("Telemetry send failed:", e)


def async_send_lock(lock_end_time):
    try:
        telemetry.send_lock(session, BASE_URL, token, AUTONOMOUS_FLIGHT_STATUS, lock_end_time)
    except Exception as e:
        print("[LOCK] API Hatası:", e)


def main_loop():
    global CURRENT_MISSION_MODE
    global closest_enemy
    if not USE_GUI:
        input_thread = threading.Thread(target=input_thread_func, daemon=True)
        input_thread.start()

    threading.Thread(target=telemetry_sender_thread, daemon=True).start()

    if not MODEL_PATH.exists():
        print("Yolo Model does not exist")
        return

    model = YOLO(str(MODEL_PATH))

    ctrls = make_controllers()

    wait_for_takeoff()

    cam_thread = CameraThread(start_camera_capture)
    threading.Thread(target=fetch_hss_thread, daemon=True).start()

    reader_thread = threading.Thread(target=mavlink_reader_thread, daemon=True)
    reader_thread.start()
    time.sleep(1.0)

    cruise_roll_deg = telemetry_data['roll']
    cruise_pitch_deg = telemetry_data['pitch']
    cruise_yaw_deg = telemetry_data['yaw']

    connection.set_mode('GUIDED')
    time.sleep(0.5)

    cmd_pitch = None
    cmd_yaw = cruise_yaw_deg
    cmd_alt = None
    cmd_speed = FOLLOW_BASE_SPEED

    prev_meas = {'alt': None, 'speed': None, 'alt_rate_smoothed': 0.0, 'spd_rate_smoothed': 0.0}
    prev_time = time.time()

    current_roll = cruise_roll_deg
    current_pitch = cruise_pitch_deg
    current_yaw = cruise_yaw_deg
    current_alt = TAKEOFF_ALT_TARGET
    current_spd = 15.0
    current_lat = None
    current_lon = None

    smoothed_dx = 0.0
    smoothed_dy = 0.0
    filter_alpha = 0.3

    target = TargetTelemetry()
    target_pitch_cmd = cruise_pitch_deg
    target_speed_cmd = current_spd
    target_throttle_cmd = GUIDED_TRIM_THROTTLE

    last_vision_yaw = None
    last_vision_pitch = None
    last_visual_speed = FOLLOW_BASE_SPEED
    last_visual_time = 0.0
    last_yolo_time = 0.0
    last_status_time = 0.0
    last_telemetry_time = 0.0
    lock_start_time = None
    lock_last_valid_time = 0.0
    lock_elapsed = 0.0
    lock_ready = False
    lock_sent = False

    yolo_hits = 0
    tracker_hits = 0
    frame_count = 0
    tracker = None
    tracker_active = False

    qr_resp = telemetry.get_qr(session, BASE_URL, token)
    qr_enlem = qr_resp.get("qrEnlem") if qr_resp else None
    qr_boylam = qr_resp.get("qrBoylam") if qr_resp else None
    qr_detector = cv2.QRCodeDetector()
    qr_mission_state = "APPROACH"
    qr_data = None
    kamikaze_start_time = {}
    sended_qr = False
    last_qr_status_time = 0.0

    global hss_list
    while True:
        now = time.time()
        dt = max(now - prev_time, DT_MIN)
        prev_time = now

        current_roll = telemetry_data['roll']
        current_pitch = telemetry_data['pitch']
        current_yaw = telemetry_data['yaw']
        current_alt = telemetry_data['alt']
        current_lat = telemetry_data['lat']
        current_lon = telemetry_data['lon']
        current_spd = telemetry_data['spd']
        battery = telemetry_data['battery']
        gps_time = telemetry.now_clock()

        if CURRENT_MISSION_MODE == "enemy":
            target.update_from_server(closest_enemy)

        hedef_x = hedef_y = hedef_w = hedef_h = 0
        visual_source = 'none'
        yaw_source = 'GPS'
        pitch_source = 'LEVEL'
        visual_score = 0.0
        current_lock_candidate = False
        lock_center_inside = False
        lock_size_ok = False
        box_w_ratio = 0.0
        box_h_ratio = 0.0
        vision_follow_distance = FOLLOW_DISTANCE_M
        ret, frame = cam_thread.read()

        if CURRENT_MISSION_MODE != "enemy":
            tracker = None
            tracker_active = False

        if ret:
            frame_count += 1
            h, w = frame.shape[:2]
            cv2.circle(frame, (w // 2, h // 2), 5, (0, 255, 255), -1)
            lock_left, lock_top, lock_right, lock_bottom = lock_area_bounds(w, h)
            cv2.rectangle(frame, (lock_left, lock_top), (lock_right, lock_bottom), (0, 255, 255), 2)

            visual_box = None
            if CURRENT_MISSION_MODE == "enemy":
                results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    best_idx = confs.argmax()
                    visual_box = tuple(float(x) for x in boxes_xyxy[best_idx])
                    visual_score = float(confs[best_idx])
                    visual_source = 'YOLO'
                    yolo_hits += 1
                    last_yolo_time = now

                    new_tracker = make_tracker()
                    if new_tracker is not None:
                        try:
                            new_tracker.init(frame, xyxy_to_tracker_box(*visual_box, w, h))
                            tracker = new_tracker
                            tracker_active = True
                        except cv2.error:
                            tracker = None
                            tracker_active = False

                elif tracker_active and tracker is not None and (now - last_yolo_time) <= TRACKER_MAX_AGE:
                    try:
                        ok, track_box = tracker.update(frame)
                    except cv2.error:
                        ok, track_box = False, None

                    if ok:
                        visual_box = tracker_box_to_xyxy(track_box)
                        visual_source = 'TRACK'
                        tracker_hits += 1
                    else:
                        tracker = None
                        tracker_active = False

            if CURRENT_MISSION_MODE == "enemy" and visual_box is not None:
                x1, y1, bw, bh = xyxy_to_tracker_box(*visual_box, w, h)
                x2 = x1 + bw
                y2 = y1 + bh
                obj_cx, obj_cy, dx_norm, dy_norm = mathHelpers.compute_center_deviation(x1, y1, x2, y2, w, h)
                hedef_x = int(obj_cx)
                hedef_y = int(obj_cy)
                hedef_w = int(bw)
                hedef_h = int(bh)
                box_w_ratio = bw / w
                box_h_ratio = bh / h

                lock_center_inside, lock_size_ok, lock_in_x, lock_in_y = lock_status(obj_cx, obj_cy, bw, bh, w, h)
                current_lock_candidate = lock_center_inside and lock_size_ok
                guide_dx_norm, guide_dy_norm = lock_guidance_error(dx_norm, dy_norm, lock_in_x, lock_in_y)

                source_alpha = filter_alpha if visual_source == 'YOLO' else 0.18
                prev_smoothed_dx = smoothed_dx
                smoothed_dx = (source_alpha * guide_dx_norm) + ((1.0 - source_alpha) * smoothed_dx)
                smoothed_dy = (source_alpha * guide_dy_norm) + ((1.0 - source_alpha) * smoothed_dy)
                dx_rate = (smoothed_dx - prev_smoothed_dx) / dt

                if current_lock_candidate:
                    box_color = (0, 255, 0)
                elif lock_center_inside:
                    box_color = (0, 165, 255)
                else:
                    box_color = (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                cv2.circle(frame, (int(obj_cx), int(obj_cy)), 5, (0, 0, 255), -1)
                cv2.line(frame, (w // 2, h // 2), (int(obj_cx), int(obj_cy)), (255, 0, 0), 2)
                cv2.putText(frame, f"{visual_source} conf={visual_score:.2f}", (int(x1), max(20, int(y1) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                yaw_control_dx = smoothed_dx
                if abs(yaw_control_dx) <= VISION_X_DEADBAND:
                    yaw_control_dx = 0.0
                elif yaw_control_dx * dx_rate < 0.0:
                    damped_dx = yaw_control_dx + (dx_rate * VISION_DX_DAMPING_TIME)
                    if yaw_control_dx > 0.0:
                        yaw_control_dx = mathHelpers.clamp(damped_dx, 0.0, yaw_control_dx)
                    else:
                        yaw_control_dx = mathHelpers.clamp(damped_dx, yaw_control_dx, 0.0)

                angle_offset = yaw_control_dx * (FOV_X_DEG / 2.0) * VISION_YAW_GAIN
                desired_yaw_from_vision = (current_yaw + angle_offset) % 360.0
                yaw_source = 'IMG'

                speed_info = vision_speed_from_distance(
                    current_lat,
                    current_lon,
                    target.lat,
                    target.lon,
                    target.speed,
                    box_w_ratio,
                    box_h_ratio,
                )
                vision_dist = speed_info[1] if speed_info is not None else None
                vision_follow_distance = speed_info[2] if speed_info is not None else FOLLOW_DISTANCE_M

                if (
                    vision_dist is not None
                    and vision_dist < VISION_TRAIL_RECOVERY_DISTANCE_M
                    and not lock_center_inside
                ):
                    trail_yaw = target_trail_yaw(
                        current_lat,
                        current_lon,
                        target.lat,
                        target.lon,
                        target.yaw,
                        vision_follow_distance,
                    )
                    if trail_yaw is not None:
                        desired_yaw_from_vision = trail_yaw
                        yaw_source = 'TRAIL'

                desired_pitch_from_vision = vision_pitch_from_target(smoothed_dy, target.pitch, vision_dist)
                last_vision_yaw = desired_yaw_from_vision
                last_vision_pitch = desired_pitch_from_vision
                last_visual_time = now
                cmd_yaw = desired_yaw_from_vision
                cmd_pitch = desired_pitch_from_vision
                cmd_alt = None

                if speed_info is not None:
                    cmd_speed = speed_info[0]
                    last_visual_speed = cmd_speed
                else:
                    box_area = bw * bh
                    area_ratio = box_area / (w * h)
                    area_error = TARGET_AREA_RATIO - area_ratio
                    cmd_speed = mathHelpers.clamp(
                        FOLLOW_BASE_SPEED + (area_error * 80.0),
                        AXIS_BOUNDS['speed'][0],
                        AXIS_BOUNDS['speed'][1],
                    )
                    last_visual_speed = cmd_speed

            elif CURRENT_MISSION_MODE == "enemy":
                visual_recent = last_visual_time > 0.0 and (now - last_visual_time) <= VISION_HOLD_SECONDS
                if visual_recent and last_vision_yaw is not None and last_vision_pitch is not None:
                    visual_source = 'HOLD'
                    yaw_source = 'HOLD'
                    visual_age = now - last_visual_time
                    cmd_yaw = last_vision_yaw
                    cmd_pitch = last_vision_pitch if visual_age <= VISION_PITCH_HOLD_SECONDS else None
                    cmd_alt = None
                    cmd_speed = last_visual_speed
                else:
                    cmd_pitch = None
                    speed_info = speed_from_distance(current_lat, current_lon, target.lat, target.lon)
                    cmd_speed = speed_info[0] if speed_info is not None else FOLLOW_BASE_SPEED

                    if current_lat is not None and current_lon is not None and target.has_reacquire_data():
                        behind_bearing = (target.yaw + 180.0) % 360.0
                        behind_lat, behind_lon = mathHelpers.destination_point(
                            target.lat,
                            target.lon,
                            behind_bearing,
                            FOLLOW_DISTANCE_M,
                        )
                        desired_yaw = mathHelpers.get_bearing(current_lat, current_lon, behind_lat, behind_lon)
                        speed_info = speed_from_distance(current_lat, current_lon, behind_lat, behind_lon)
                        cmd_speed = speed_info[0] if speed_info is not None else FOLLOW_BASE_SPEED
                        cmd_yaw = desired_yaw
                        cmd_alt = target.alt
                        yaw_source = 'TRAIL'
                    else:
                        if cmd_yaw is None:
                            cmd_yaw = current_yaw
                        cmd_alt = TAKEOFF_ALT_TARGET
                        yaw_source = 'LEVEL'

            if CURRENT_MISSION_MODE == "qr":
                last_vision_yaw = None
                last_vision_pitch = None
                cmd_speed = FOLLOW_BASE_SPEED
                yaw_source = 'QR'

                if qr_enlem is not None and qr_boylam is not None and current_lat is not None and current_lon is not None:
                    dist_to_qr = mathHelpers.get_distance(current_lat, current_lon, qr_enlem, qr_boylam)
                    bearing_to_qr = mathHelpers.get_bearing(current_lat, current_lon, qr_enlem, qr_boylam)
                    yaw_error_to_qr = mathHelpers.wrap_angle_deg(bearing_to_qr - current_yaw)
                    cmd_yaw = bearing_to_qr

                    if qr_mission_state == "APPROACH":
                        cmd_alt = QR_APPROACH_ALT_M
                        cmd_pitch = None
                        if QR_DEBUG and now - last_qr_status_time >= QR_STATUS_INTERVAL:
                            print(
                                f"[QR MISSION] APPROACH dist={dist_to_qr:.1f}m "
                                f"bearing={bearing_to_qr:.1f} yaw_err={yaw_error_to_qr:.1f}deg "
                                f"alt={current_alt:.1f}m"
                            )
                            last_qr_status_time = now
                        if (
                            dist_to_qr < QR_DIVE_START_DIST_M
                            and current_alt > QR_DIVE_MIN_ALT_M
                            and abs(yaw_error_to_qr) <= QR_DIVE_ALIGN_YAW_ERR_DEG
                        ):
                            kamikaze_start_time = telemetry.now_clock()
                            qr_mission_state = "DIVE"
                            print("[QR MISSION] Close to QR! Initiating DIVE!")

                    elif qr_mission_state == "DIVE":
                        cmd_pitch = QR_DIVE_PITCH_DEG
                        cmd_alt = None
                        cmd_speed = AXIS_BOUNDS['speed'][0]
                        qr_vision_yaw = None
                        qr_data, bbox = None, None
                        qr_dx_norm = None
                        qr_dy_norm = None

                        try:
                            qr_data, bbox, _ = qr_detector.detectAndDecode(frame)
                        except cv2.error:
                            pass

                        if bbox is not None:
                            pts = bbox[0]
                            qr_cx = sum(p[0] for p in pts) / 4.0
                            qr_cy = sum(p[1] for p in pts) / 4.0
                            dx_norm = (qr_cx - w / 2.0) / (w / 2.0)
                            dy_norm = (qr_cy - h / 2.0) / (h / 2.0)
                            qr_dx_norm = dx_norm
                            qr_dy_norm = dy_norm
                            qr_vision_yaw = (current_yaw + dx_norm * (FOV_X_DEG / 2.0) * QR_VISION_YAW_GAIN) % 360.0
                            vision_pitch = current_pitch - dy_norm * (FOV_Y_DEG / 2.0)
                            cmd_pitch = mathHelpers.clamp(vision_pitch, -45.0, -15.0)
                            pts = np.int32(pts).reshape(-1, 1, 2)
                            cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
                            cv2.circle(frame, (int(qr_cx), int(qr_cy)), 5, (0, 255, 255), -1)

                        if QR_DEBUG and now - last_qr_status_time >= QR_STATUS_INTERVAL:
                            qr_dx_text = f"{qr_dx_norm:.2f}" if qr_dx_norm is not None else "n/a"
                            qr_dy_text = f"{qr_dy_norm:.2f}" if qr_dy_norm is not None else "n/a"
                            print(
                                f"[QR MISSION] DIVE dist={dist_to_qr:.1f}m "
                                f"bearing={bearing_to_qr:.1f} yaw_err={yaw_error_to_qr:.1f}deg "
                                f"alt={current_alt:.1f}m visible={bbox is not None} "
                                f"decoded={bool(qr_data)} dx={qr_dx_text} dy={qr_dy_text}"
                            )
                            last_qr_status_time = now

                        if qr_data:
                            print(f"[QR MISSION] QR detected: {qr_data}")
                            if qr_resp and not sended_qr:
                                kamikaze_zaman = telemetry.now_clock()
                                telemetry.send_kamikaze(session, BASE_URL, token, TEAM_NO, qr_data, kamikaze_start_time, kamikaze_zaman)
                                sended_qr = True
                                qr_mission_state = "PULLOUT"
                                print("[QR MISSION] QR read successfully! Returning to normal flight.")

                        if qr_vision_yaw is not None:
                            cmd_yaw = qr_vision_yaw
                        if current_alt < QR_DIVE_ABORT_ALT_M:
                            qr_mission_state = "PULLOUT"
                            CURRENT_MISSION_MODE = "normal"
                            print("[QR MISSION] Alt < 45m! Aborting dive, pulling out!")

                    elif qr_mission_state == "PULLOUT":
                        cmd_yaw = mathHelpers.get_bearing(current_lat, current_lon, qr_enlem, qr_boylam)
                        cmd_alt = TAKEOFF_ALT_TARGET
                        cmd_pitch = None
                        qr_mission_state = "APPROACH"
                        CURRENT_MISSION_MODE = "normal"
                        print("[QR MISSION] Returning to normal flight.")

            if CURRENT_MISSION_MODE == "normal":
                cmd_pitch = None
                cmd_alt = TAKEOFF_ALT_TARGET
                cmd_speed = FOLLOW_BASE_SPEED
                cmd_yaw = current_yaw
                yaw_source = 'LEVEL'

            if current_lock_candidate:
                if lock_start_time is None:
                    lock_start_time = now
                    lock_sent = False
                    print("[LOCK] Kilitlenme başladı.")
                lock_last_valid_time = now
            elif lock_start_time is not None and (now - lock_last_valid_time) > LOCK_TOLERANCE_SECONDS:
                print(f"[LOCK] Kilit koptu. Süre: {lock_elapsed:.1f}s")
                lock_start_time = None
                lock_last_valid_time = 0.0
                lock_sent = False

            lock_elapsed = now - lock_start_time if lock_start_time is not None else 0.0
            lock_ready = lock_elapsed >= LOCK_REQUIRED_SECONDS
            if lock_ready and not lock_sent:
                lock_sent = True
                print("[LOCK] Başarılı kilit. API'ye gönderiliyor...")
                threading.Thread(target=async_send_lock, args=(telemetry.now_clock(),), daemon=True).start()

            lock_text = f"LOCK {min(lock_elapsed, LOCK_REQUIRED_SECONDS):.1f}/{LOCK_REQUIRED_SECONDS:.0f}s"
            lock_color = (0, 255, 0) if lock_ready else (0, 255, 255)
            cv2.putText(frame, lock_text, (lock_left, max(20, lock_top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lock_color, 2)
            cv2.imshow("YOLOv8 Pose UDP Inference", frame)

            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if success:
                try:
                    send_sock.sendto(buffer.tobytes(), (UDP_OUT_IP, UDP_OUT_PORT))
                except OSError as e:
                    if e.errno == 90:
                        print(f"[UDP HATA] Görüntü paketi çok büyük ({len(buffer.tobytes())} bytes > 65535). Lütfen WIDTH ve HEIGHT değerlerini düşürün veya JPEG_QUALITY'yi azaltın!")
                    else:
                        print(f"[UDP HATA] {e}")

        elif CURRENT_MISSION_MODE == "enemy":
            cmd_pitch = None
            speed_info = speed_from_distance(current_lat, current_lon, target.lat, target.lon)
            cmd_speed = speed_info[0] if speed_info is not None else FOLLOW_BASE_SPEED
            if current_lat is not None and current_lon is not None and target.has_reacquire_data():
                behind_bearing = (target.yaw + 180.0) % 360.0
                behind_lat, behind_lon = mathHelpers.destination_point(target.lat, target.lon, behind_bearing, FOLLOW_DISTANCE_M)
                cmd_yaw = mathHelpers.get_bearing(current_lat, current_lon, behind_lat, behind_lon)
                cmd_alt = target.alt
                yaw_source = 'TRAIL'
            else:
                cmd_alt = TAKEOFF_ALT_TARGET
                yaw_source = 'LEVEL'

            if lock_start_time is not None and (now - lock_last_valid_time) > LOCK_TOLERANCE_SECONDS:
                print(f"[LOCK] Görüntü yok, kilit koptu. Süre: {lock_elapsed:.1f}s")
                lock_start_time = None
                lock_last_valid_time = 0.0
                lock_sent = False

        elif CURRENT_MISSION_MODE == "normal":
            cmd_pitch = None
            cmd_yaw = current_yaw
            cmd_alt = TAKEOFF_ALT_TARGET
            cmd_speed = FOLLOW_BASE_SPEED
            yaw_source = 'LEVEL'

        elif CURRENT_MISSION_MODE == "qr":
            cmd_speed = FOLLOW_BASE_SPEED
            if qr_enlem is not None and qr_boylam is not None and current_lat is not None and current_lon is not None:
                cmd_yaw = mathHelpers.get_bearing(current_lat, current_lon, qr_enlem, qr_boylam)
                if qr_mission_state == "DIVE":
                    cmd_pitch = QR_DIVE_PITCH_DEG
                    cmd_alt = None
                else:
                    cmd_pitch = None
                    cmd_alt = QR_APPROACH_ALT_M
            else:
                cmd_pitch = None
                cmd_yaw = current_yaw
                cmd_alt = TAKEOFF_ALT_TARGET
            yaw_source = 'QR'

        target_lat = target.lat if target.has_position() else None
        target_lon = target.lon if target.has_position() else None
        map_img = draw_minimap(current_lat, current_lon, current_yaw, hss_list, FLIGHT_BOUNDARIES, target_lat, target_lon, qr_enlem, qr_boylam)
        cv2.imshow("Minimap", map_img)
        cv2.waitKey(1)

        if now - last_telemetry_time >= 0.1:
            update_dispatch_state(
                kilit=1 if lock_start_time is not None else 0,
                hx=hedef_x,
                hy=hedef_y,
                hw=hedef_w,
                hh=hedef_h,
            )
            last_telemetry_time = now

        t_pitch, t_yaw, t_alt, t_speed = cmd_pitch, cmd_yaw, cmd_alt, cmd_speed

        if current_lat is not None and current_lon is not None and t_yaw is not None:
            safe_yaw = t_yaw
            if hss_list:
                safe_yaw = mathHelpers.compute_apf_hss(
                    current_lat, current_lon, current_yaw, current_spd, safe_yaw, hss_list
                )
            if FLIGHT_BOUNDARIES:
                safe_yaw = mathHelpers.enforce_flight_boundaries(
                    current_lat, current_lon, current_yaw, current_spd, safe_yaw, FLIGHT_BOUNDARIES
                )
            if abs(mathHelpers.wrap_angle_deg(safe_yaw - t_yaw)) > 1.0:
                t_yaw = safe_yaw
                yaw_source = 'SAFE'

        if prev_meas['alt'] is None:
            prev_meas['alt'] = current_alt
        dA = current_alt - prev_meas['alt']
        a_rate = (DX_CONST * (-dA / dt)) + (PREV_MEAS_RATE_CONST * prev_meas['alt_rate_smoothed'])
        a_rate = mathHelpers.clamp(a_rate, -12.0, 12.0)
        prev_meas['alt_rate_smoothed'] = a_rate
        prev_meas['alt'] = current_alt

        if prev_meas['speed'] is None:
            prev_meas['speed'] = current_spd
        spd_delta = current_spd - prev_meas['speed']
        err_rate = (DX_CONST * (-spd_delta / dt)) + (PREV_MEAS_RATE_CONST * prev_meas['spd_rate_smoothed'])
        err_rate = mathHelpers.clamp(err_rate, -8.0, 8.0)
        prev_meas['spd_rate_smoothed'] = err_rate
        prev_meas['speed'] = current_spd

        desired_pitch = 0.0
        if t_alt is not None:
            alt_error = t_alt - current_alt
            if abs(alt_error) <= ALTITUDE_DEADBAND_M:
                ctrls['altitude'].reset()
            else:
                desired_pitch = mathHelpers.clamp(
                    ctrls['altitude'].compute(alt_error, a_rate, dt),
                    -GPS_REACQUIRE_PITCH_LIMIT_DEG,
                    GPS_REACQUIRE_PITCH_LIMIT_DEG,
                )
                pitch_source = 'ALT'
        elif t_pitch is not None:
            low, high = AXIS_BOUNDS['pitch']
            desired_pitch = mathHelpers.clamp(t_pitch, low, high)
            pitch_source = 'VIS'

        low, high = AXIS_BOUNDS['pitch']
        pitch_rate_limit = VISION_PITCH_CMD_RATE_DPS if t_pitch is not None else GPS_PITCH_CMD_RATE_DPS
        target_pitch_cmd = mathHelpers.clamp(
            rate_limit_value(target_pitch_cmd, desired_pitch, pitch_rate_limit, dt),
            low,
            high,
        )

        if t_speed is not None:
            low, high = AXIS_BOUNDS['speed']
            if t_speed < target_speed_cmd:
                target_speed_cmd = mathHelpers.clamp(t_speed, low, high)
            else:
                target_speed_cmd = mathHelpers.clamp(
                    ctrls['speed'].smooth_value(target_speed_cmd, t_speed, dt),
                    low,
                    high,
                )

        speed_error = target_speed_cmd - current_spd
        desired_throttle = mathHelpers.clamp(
            GUIDED_TRIM_THROTTLE + ctrls['speed'].compute(speed_error, err_rate, dt),
            GUIDED_MIN_THROTTLE,
            GUIDED_MAX_THROTTLE,
        )
        throttle_step_limit = GUIDED_THROTTLE_RATE * dt
        target_throttle_cmd += mathHelpers.clamp(
            desired_throttle - target_throttle_cmd,
            -throttle_step_limit,
            throttle_step_limit,
        )
        target_throttle_cmd = mathHelpers.clamp(target_throttle_cmd, GUIDED_MIN_THROTTLE, GUIDED_MAX_THROTTLE)

        target_yaw_cmd = t_yaw % 360.0 if t_yaw is not None else None
        if now - last_status_time >= STATUS_INTERVAL:
            speed_info = speed_from_distance(current_lat, current_lon, target.lat, target.lon)
            dist_text = f"{speed_info[1]:.1f}m" if speed_info is not None else "n/a"
            target_speed_text = f"{target.speed:.1f}" if target.speed is not None else "n/a"
            target_pitch_text = f"{target.pitch:.1f}" if target.pitch is not None else "n/a"
            debug_parts = [
                f"Mission={CURRENT_MISSION_MODE}",
                f"Vision={visual_source}",
                f"YawSrc={yaw_source}",
                f"TgtYaw={t_yaw}",
                f"CurYaw={current_yaw:.1f}",
                f"CmdYaw={target_yaw_cmd if target_yaw_cmd is not None else 'n/a'}",
                f"TgtPitch={t_pitch}",
                f"TgtPlanePitch={target_pitch_text}",
                f"CurPitch={current_pitch:.1f}",
                f"DesPitch={desired_pitch:.1f}",
                f"CmdPitch={target_pitch_cmd:.1f}",
                f"PitchSrc={pitch_source}",
                f"CmdSpd={t_speed}",
                f"OwnSpd={current_spd:.1f}",
                f"TgtSpd={target_speed_text}",
                f"Thr={target_throttle_cmd:.2f}",
                f"Dist={dist_text}",
                f"Follow={vision_follow_distance:.1f}m",
                f"Conf={visual_score:.2f}",
                f"BoxW={box_w_ratio * 100.0:.1f}%",
                f"BoxH={box_h_ratio * 100.0:.1f}%",
                f"LockIn={int(lock_center_inside)}",
                f"BoxOK={int(lock_size_ok)}",
                f"LockT={lock_elapsed:.1f}",
                f"LockOK={int(lock_ready)}",
                f"YOLO={yolo_hits}/{frame_count}",
                f"TRACK={tracker_hits}",
            ]
            print(" ".join(debug_parts))
            last_status_time = now

        if t_yaw is not None:
            send_guided_heading(target_yaw_cmd)
        send_guided_speed(target_speed_cmd)

        if t_pitch is not None:
            attitude_yaw = target_yaw_cmd if target_yaw_cmd is not None else current_yaw
            send_guided_attitude(current_roll, target_pitch_cmd, attitude_yaw, throttle=target_throttle_cmd)
        elif t_alt is not None:
            send_guided_altitude(t_alt)

        time.sleep(0.05)


if __name__ == '__main__':
    if USE_GUI:
        try:
            _router = LogRouter(sys.stdout)
            sys.stdout = _router
            _ui = MissionUI(_router)
        except Exception as e:
            print(f"[GUI] Açılamadı, terminale geri dönülüyor: {e}")
            USE_GUI = False

    token = telemetry.login(session, BASE_URL, USERNAME, PASSWORD)
    configure_speed_limits()
    wait_for_prearm()
    auto_and_arm()

    if USE_GUI:
        threading.Thread(target=main_loop, daemon=True).start()
        _ui.run()
    else:
        main_loop()
