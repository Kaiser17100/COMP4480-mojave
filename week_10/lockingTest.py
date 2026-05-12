## HOURS WASTED: 99999999999999

# Notes:
# need the send telemetry data

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
import commandState as CS
import requests
import threading
import sys
import select
from datetime import datetime

CURRENT_MISSION_MODE = "normal"
LANDING_STATE = None  # "RETURN" or "LAND"
LANDING_ALT = 10.0  # altitude to descend to during return
RETURN_ALT = 30.0
LANDING_APPROACH_DIST = 30.0  # meters from home to trigger final land
home_lat = None
home_lon = None
BASE_URL = "http://192.168.1.50:10001"
USERNAME = "4"
PASSWORD = "4"
TEAM_NO = 4

## GLOBAL VARIABLES ##
FLIGHT_BOUNDARIES = [
    (38.707147, 27.445404),
    (38.707147, 27.464124),
    (38.691740, 27.464124),
    (38.691740, 27.445404),
]

AXIS_BOUNDS = {
    'pitch': (-35.0, 35.0),
    'roll': (-45.0, 45.0),
    'yaw': (-180.0, 180.0),
    'alt': (50.0, 300.0),
    'speed': (13, 30)
}

MODEL_PATH = Path.home() / "Desktop" / "runs" / "pose" / "train" / "weights" / "best.pt"

TAKEOFF_ALT_TARGET = 50.0
TAKEOFF_ALT_THRESH = 5.0
DX_CONST = 0.25
DT_MIN = 0.01
PREV_MEAS_RATE_CONST = 0.75
CONF = 0.25
IMG_SIZE = 640
FOV_X_DEG = 80.0
FOV_Y_DEG = 60.0
PREARM_CONST = mavutil.mavlink.MAV_SYS_STATUS_PREARM_CHECK
MAX_DISTANCE_BETWEEN_ENEMY = 500.0
AUTONOMOUS_FLIGHT_STATUS = 1
session = requests.Session()
hss_enabled = False

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
LOCK_BBOX_THRESHOLD = 0.15
is_locked_on = False
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
QR_APPROACH_SPEED_MPS = 20.0
QR_DIVE_MIN_START_DIST_M = 150.0
QR_DIVE_MAX_START_DIST_M = 260.0
QR_DIVE_PLANNED_PITCH_DEG = -35.0
QR_DIVE_MIN_PITCH_DEG = -45.0
QR_DIVE_MAX_PITCH_DEG = -15.0
QR_DIVE_LINE_LOOKAHEAD_M = 35.0
QR_DIVE_FINAL_DIRECT_DIST_M = 25.0
QR_DIVE_VISION_LINE_BLEND = 0.35
QR_DIVE_START_MARGIN_M = 8.0
QR_DIVE_MIN_VERTICAL_SPEED_MPS = 3.0
QR_DIVE_MIN_CLOSING_SPEED_MPS = 5.0
QR_DIVE_MAX_HEADING_ERR_DEG = 5.0
QR_DIVE_MAX_LATERAL_SPEED_MPS = 1.5
QR_DIVE_BEARING_STABLE_SECONDS = 0.4
QR_DETECTOR_EPS = 0.35
QR_WARP_SIZE_PX = 420

control = True

SCREENSHOT_DIR = Path.home() / "Desktop" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

def save_screenshot(frame, tag="event"):
    """Save a timestamped screenshot to SCREENSHOT_DIR."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = SCREENSHOT_DIR / f"{tag}_{ts}.jpg"
    try:
        cv2.imwrite(str(filename), frame)
        print(f"[SCREENSHOT] Saved: {filename}")
    except Exception as e:
        print(f"[SCREENSHOT] Failed to save: {e}")

## HELPERS ##

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

    return desired_speed, dist, follow_distance

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

def qr_dive_velocity_components(current_spd, pitch_deg, current_yaw, target_bearing):
    if current_spd is None or current_spd <= 0.0:
        speed = QR_APPROACH_SPEED_MPS
    else:
        speed = mathHelpers.clamp(current_spd, AXIS_BOUNDS['speed'][0], QR_APPROACH_SPEED_MPS)

    pitch_deg = mathHelpers.clamp(pitch_deg, QR_DIVE_MIN_PITCH_DEG, QR_DIVE_MAX_PITCH_DEG)
    pitch_rad = math.radians(abs(pitch_deg))
    horizontal_speed = speed * math.cos(pitch_rad)
    vertical_speed = max(speed * math.sin(pitch_rad), QR_DIVE_MIN_VERTICAL_SPEED_MPS)

    heading_error = math.radians(mathHelpers.wrap_angle_deg(current_yaw - target_bearing))
    closing_speed = horizontal_speed * math.cos(heading_error)
    lateral_speed = horizontal_speed * math.sin(heading_error)
    closing_speed = max(closing_speed, QR_DIVE_MIN_CLOSING_SPEED_MPS)

    return closing_speed, vertical_speed, lateral_speed, speed

def qr_dynamic_dive_start_distance(current_alt, current_spd, current_yaw, target_bearing):
    x_speed, y_speed, lateral_speed, speed = qr_dive_velocity_components(
        current_spd,
        QR_DIVE_PLANNED_PITCH_DEG,
        current_yaw,
        target_bearing,
    )
    dive_time = max(current_alt, 0.0) / y_speed
    start_distance = (x_speed * dive_time) + QR_DIVE_START_MARGIN_M
    start_distance = mathHelpers.clamp(
        start_distance,
        QR_DIVE_MIN_START_DIST_M,
        QR_DIVE_MAX_START_DIST_M,
    )
    return start_distance, dive_time, x_speed, y_speed, lateral_speed, speed

def optional_float(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None

def local_offset_m(origin_lat, origin_lon, lat, lon):
    origin_lat = float(origin_lat)
    origin_lon = float(origin_lon)
    lat = float(lat)
    lon = float(lon)
    meter_per_deg_lat = 111320.0
    meter_per_deg_lon = 111320.0 * math.cos(math.radians(origin_lat))
    east_m = (lon - origin_lon) * meter_per_deg_lon
    north_m = (lat - origin_lat) * meter_per_deg_lat
    return east_m, north_m

def qr_dive_line_guidance(current_lat, current_lon, qr_lat, qr_lon, dive_start_lat, dive_start_lon, dive_bearing):
    direct_yaw = mathHelpers.get_bearing(current_lat, current_lon, qr_lat, qr_lon)
    if None in (dive_start_lat, dive_start_lon, dive_bearing):
        return direct_yaw, 0.0, mathHelpers.get_distance(current_lat, current_lon, qr_lat, qr_lon)

    current_e, current_n = local_offset_m(dive_start_lat, dive_start_lon, current_lat, current_lon)
    qr_e, qr_n = local_offset_m(dive_start_lat, dive_start_lon, qr_lat, qr_lon)
    path_len = math.hypot(qr_e, qr_n)
    if path_len <= 1.0:
        return direct_yaw, 0.0, mathHelpers.get_distance(current_lat, current_lon, qr_lat, qr_lon)

    forward_e = qr_e / path_len
    forward_n = qr_n / path_len
    right_e = forward_n
    right_n = -forward_e

    along = (current_e * forward_e) + (current_n * forward_n)
    cross_track = (current_e * right_e) + (current_n * right_n)
    along_remaining = path_len - along

    if along_remaining <= QR_DIVE_FINAL_DIRECT_DIST_M:
        return direct_yaw, cross_track, along_remaining

    aim_along = mathHelpers.clamp(
        along + QR_DIVE_LINE_LOOKAHEAD_M,
        0.0,
        path_len,
    )
    aim_e = forward_e * aim_along
    aim_n = forward_n * aim_along

    yaw_e = aim_e - current_e
    yaw_n = aim_n - current_n
    if math.hypot(yaw_e, yaw_n) <= 0.5:
        return direct_yaw, cross_track, along_remaining

    line_yaw = (math.degrees(math.atan2(yaw_e, yaw_n)) + 360.0) % 360.0
    return line_yaw, cross_track, along_remaining

def normalize_qr_points(points):
    if points is None:
        return None
    arr = np.asarray(points, dtype=np.float32)
    if arr.size < 8:
        return None
    arr = arr.reshape(-1, 2)
    if len(arr) < 4:
        return None
    return arr[:4]

def order_quad_points(points):
    pts = normalize_qr_points(points)
    if pts is None:
        return None

    rect = np.zeros((4, 2), dtype=np.float32)
    point_sum = pts.sum(axis=1)
    point_diff = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(point_sum)]
    rect[2] = pts[np.argmax(point_sum)]
    rect[1] = pts[np.argmin(point_diff)]
    rect[3] = pts[np.argmax(point_diff)]
    return rect

def warp_qr_from_points(frame, points, size=QR_WARP_SIZE_PX):
    src = order_quad_points(points)
    if src is None:
        return None

    dst = np.array(
        [
            [0.0, 0.0],
            [size - 1.0, 0.0],
            [size - 1.0, size - 1.0],
            [0.0, size - 1.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, matrix, (size, size))
    border = max(20, size // 8)
    return cv2.copyMakeBorder(
        warped,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

def qr_image_variants(image):
    variants = [("raw", image)]
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    variants.append(("gray", gray))
    variants.append(("equalized", cv2.equalizeHist(gray)))
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharpened = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)
    variants.append(("sharp", sharpened))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        5,
    )
    variants.append(("adaptive", adaptive))
    variants.append(("adaptive_inv", cv2.bitwise_not(adaptive)))
    return variants

def qr_try_decode(detector, image):
    best_points = None
    try:
        data, points, _ = detector.detectAndDecode(image)
        best_points = normalize_qr_points(points)
        if data:
            return data, best_points, "single"
    except cv2.error:
        pass

    if hasattr(detector, "detectAndDecodeMulti"):
        try:
            ok, decoded_info, points, _ = detector.detectAndDecodeMulti(image)
            points_arr = np.asarray(points, dtype=np.float32) if points is not None else None
            if ok and points_arr is not None and points_arr.size >= 8:
                points_arr = points_arr.reshape(-1, 4, 2)
                for idx, data in enumerate(decoded_info):
                    data_points = normalize_qr_points(points_arr[idx])
                    if data:
                        return data, data_points, "multi"
                if best_points is None:
                    best_points = normalize_qr_points(points_arr[0])
        except cv2.error:
            pass

    return None, best_points, None

def robust_qr_detect_and_decode(detector, frame):
    data, points, method = qr_try_decode(detector, frame)
    if data:
        return data, points, method

    best_points = points
    if best_points is not None:
        warped = warp_qr_from_points(frame, best_points)
        if warped is not None:
            for variant_name, variant in qr_image_variants(warped):
                data, _, method = qr_try_decode(detector, variant)
                if data:
                    return data, best_points, f"warp_{variant_name}_{method}"

    for variant_name, variant in qr_image_variants(frame)[1:]:
        data, points, method = qr_try_decode(detector, variant)
        if data:
            return data, points if points is not None else best_points, f"{variant_name}_{method}"
        if best_points is None and points is not None:
            best_points = points

    return None, best_points, None

connection = mavutil.mavlink_connection("udpin:0.0.0.0:15000")
connection.wait_heartbeat()
print("Connected to Fixed-Wing Vehicle...")

connection.mav.request_data_stream_send(
    connection.target_system,
    connection.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    20,
    1
)

def input_thread_func():
    global CURRENT_MISSION_MODE, LANDING_STATE, hss_enabled
    while True:
        try:
            val = input("Enter mission mode ('enemy', 'qr', 'normal', 'hss' or 'landing'): ").strip().lower()
            if val in ["enemy", "qr", "normal"]:
                CURRENT_MISSION_MODE = val
                LANDING_STATE = None
                print(f"[MISSION] Mode switched to: {CURRENT_MISSION_MODE}")
            elif val == "hss":
                hss_enabled = not hss_enabled
                print(f"[MISSION] HSS {'enabled' if hss_enabled else 'disabled'}")
            elif val == "landing":
                if home_lat is None or home_lon is None:
                    print("[LANDING] Cannot land: home coordinates not yet captured (no GPS fix).")
                else:
                    CURRENT_MISSION_MODE = "landing"
                    LANDING_STATE = "RETURN"
                    print(f"[LANDING] Landing sequence initiated. Returning to home ({home_lat:.6f}, {home_lon:.6f}) at {LANDING_ALT}m altitude.")
            else:
                print(f"[MISSION] Invalid mode '{val}'. Use 'enemy', 'qr', 'normal', 'hss' or 'landing'.")
        except:
            break


def wait_for_prearm():
    print("Waiting for pre-arm...")
    last_telemetry_time = time.time()
    current_lat, current_lon, current_alt = 0.0, 0.0, 0.0
    current_pitch, current_yaw, current_roll, current_spd, battery = 0.0, 0.0, 0.0, 0.0, 100.0
    while True:
        try:
            msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD', 'SYS_STATUS'], blocking=False)
            prearm_ok = False
            while msg is not None:
                msg_type = msg.get_type()
                if msg_type == 'SYS_STATUS':
                    battery = 1
                    if msg.onboard_control_sensors_health & PREARM_CONST == PREARM_CONST:
                        prearm_ok = True
                elif msg_type == 'ATTITUDE':
                    current_roll = math.degrees(msg.roll)
                    current_pitch = math.degrees(msg.pitch)
                    current_yaw = math.degrees(msg.yaw)
                elif msg_type == 'GLOBAL_POSITION_INT':
                    current_alt = msg.relative_alt / 1000.0
                    current_lat = msg.lat / 1e7
                    current_lon = msg.lon / 1e7
                elif msg_type == 'VFR_HUD':
                    current_spd = msg.airspeed
                msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD', 'SYS_STATUS'], blocking=False)
                
            if prearm_ok:
                print("Pre-arm good...")
                return

            now = time.time()
            if now - last_telemetry_time >= 0.5:
                if current_lat != 0.0 and current_lon != 0.0:
                    telemetry_args = {
                        'session': session, 'base_url': BASE_URL, 'token': token, 'team_no': TEAM_NO,
                        'lat': current_lat, 'lon': current_lon, 'alt': current_alt,
                        'pitch': current_pitch, 'yaw': current_yaw % 360, 'roll': current_roll,
                        'spd': current_spd, 'battery': battery, 'otonom': AUTONOMOUS_FLIGHT_STATUS,
                        'gps_time': telemetry.now_clock(), 'kilit': 0,
                        'hx': 0, 'hy': 0, 'hw': 0, 'hh': 0
                    }
                    threading.Thread(target=async_send_telemetry, args=(telemetry_args,), daemon=True).start()
                last_telemetry_time = now

            time.sleep(0.1)

        except Exception as e:
            print(f'Error in wait_for_prearm: {e}')
            time.sleep(1)
            break


def wait_for_start():
    print("Type 'start' to continue with auto_and_arm: ")
    last_telemetry_time = time.time()
    current_lat, current_lon, current_alt = 0.0, 0.0, 0.0
    current_pitch, current_yaw, current_roll, current_spd, battery = 0.0, 0.0, 0.0, 0.0, 100.0
    while True:
        i, o, e = select.select([sys.stdin], [], [], 0.1)
        if i:
            val = sys.stdin.readline().strip().lower()
            if val == 'start':
                print("Starting auto_and_arm...")
                break
            else:
                print("Invalid input. Please type 'start'.")
                
        try:
            msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD', 'SYS_STATUS'], blocking=False)
            while msg is not None:
                msg_type = msg.get_type()
                if msg_type == 'ATTITUDE':
                    current_roll = math.degrees(msg.roll)
                    current_pitch = math.degrees(msg.pitch)
                    current_yaw = math.degrees(msg.yaw)
                elif msg_type == 'GLOBAL_POSITION_INT':
                    current_alt = msg.relative_alt / 1000.0
                    current_lat = msg.lat / 1e7
                    current_lon = msg.lon / 1e7
                elif msg_type == 'VFR_HUD':
                    current_spd = msg.airspeed
                elif msg_type == 'SYS_STATUS':
                    battery = msg.battery_remaining if msg.battery_remaining > 0 else 0
                msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD', 'SYS_STATUS'], blocking=False)
        except Exception as e:
            print(f'Error reading MAVLink in wait_for_start: {e}')
            
        now = time.time()
        if now - last_telemetry_time >= 0.5:
            if current_lat != 0.0 and current_lon != 0.0:
                telemetry_args = {
                    'session': session, 'base_url': BASE_URL, 'token': token, 'team_no': TEAM_NO,
                    'lat': current_lat, 'lon': current_lon, 'alt': current_alt,
                    'pitch': current_pitch, 'yaw': current_yaw % 360, 'roll': current_roll,
                    'spd': current_spd, 'battery': battery, 'otonom': AUTONOMOUS_FLIGHT_STATUS,
                    'gps_time': telemetry.now_clock(), 'kilit': 0,
                    'hx': 0, 'hy': 0, 'hw': 0, 'hh': 0
                }
                threading.Thread(target=async_send_telemetry, args=(telemetry_args,), daemon=True).start()
            last_telemetry_time = now


def auto_and_arm():
    print("Setting Mode to TAKEOFF...")
    try:
        connection.set_mode('TAKEOFF')
    except Exception as e:
        print(f'Error setting mode: {e}')

    print("Waiting for EKF alignment and arming...")
    while True:
        try:
            connection.mav.command_long_send(
                connection.target_system, connection.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
            )
    
            msg = connection.recv_match(type='HEARTBEAT', blocking=True, timeout=1.0)
    
            if msg and (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                print("Plane is successfully Armed!")
                return
        except Exception as e:
            print(f'Error in auto_and_arm: {e}')
            break

        time.sleep(1.0)


import socket
from urllib.parse import urlparse

# =========================
# RTP/H264 UDP GİRİŞ
# =========================
UDP_IN_PORT = int(os.getenv("IHA_CAMERA_IN_PORT", "5604"))

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
UDP_OUT_PORT = 5428

# =========================
# Görüntü boyutu
# =========================
WIDTH = 1920
HEIGHT = 1080
JPEG_QUALITY = 50

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

UDP_IN_ADDRESS = os.getenv("IHA_CAMERA_IN_ADDRESS", "0.0.0.0")
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


closest_enemy = None

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
        elif resp.status_code == 204:
            print("Error 204: Bad Telemetry")
            print(kwargs)
    except Exception as e:
        print("Telemetry send failed:", e)

def async_send_lock(lock_end_time):
    try:
        telemetry.send_lock(session, BASE_URL, token, AUTONOMOUS_FLIGHT_STATUS, lock_end_time)
    except Exception as e:
        print("[LOCK] API Hatası:", e)

def draw_minimap(current_lat, current_lon, current_yaw, hss_list, flight_boundaries, target_lat=None, target_lon=None, qr_lat=None, qr_lon=None):
    map_w, map_h = 600, 600
    map_img = np.ones((map_h, map_w, 3), dtype=np.uint8) * 30
    
    cx, cy = map_w // 2, map_h // 2
    
    if current_lat is None or current_lon is None:
        cv2.putText(map_img, "Waiting for GPS...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return map_img
        
    meter_per_deg_lat = 111320.0
    meter_per_deg_lon = 111320.0 * math.cos(math.radians(current_lat))
    
    max_dist_m = 100.0
    if flight_boundaries and len(flight_boundaries) >= 3:
        for lat, lon in flight_boundaries:
            dy_m = abs((lat - current_lat) * meter_per_deg_lat)
            dx_m = abs((lon - current_lon) * meter_per_deg_lon)
            max_dist_m = max(max_dist_m, dx_m, dy_m)
            
    for hss in hss_list:
        h_lat = hss.get("hssEnlem", 0.0)
        h_lon = hss.get("hssBoylam", 0.0)
        h_rad = hss.get("hssYaricap", 0.0)
        dy_m = abs((h_lat - current_lat) * meter_per_deg_lat)
        dx_m = abs((h_lon - current_lon) * meter_per_deg_lon)
        max_dist_m = max(max_dist_m, dx_m + h_rad, dy_m + h_rad)
        
    if target_lat is not None and target_lon is not None:
        dy_m = abs((target_lat - current_lat) * meter_per_deg_lat)
        dx_m = abs((target_lon - current_lon) * meter_per_deg_lon)
        max_dist_m = max(max_dist_m, dx_m, dy_m)
        
    if qr_lat is not None and qr_lon is not None:
        dy_m = abs((qr_lat - current_lat) * meter_per_deg_lat)
        dx_m = abs((qr_lon - current_lon) * meter_per_deg_lon)
        max_dist_m = max(max_dist_m, dx_m, dy_m)
        
    max_dist_m *= 1.2
    pixels_per_meter = (map_w / 2.0) / max_dist_m
    
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
        
        px = int(cx + dx_m * pixels_per_meter)
        py = int(cy - dy_m * pixels_per_meter)
        r_px = int(h_rad * pixels_per_meter)
        
        cv2.circle(map_img, (px, py), r_px, (0, 0, 255), 2)
        cv2.drawMarker(map_img, (px, py), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
        cv2.putText(map_img, f"HSS R:{int(h_rad)}m", (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    if target_lat is not None and target_lon is not None:
        dy_m = (target_lat - current_lat) * meter_per_deg_lat
        dx_m = (target_lon - current_lon) * meter_per_deg_lon
        
        px = int(cx + dx_m * pixels_per_meter)
        py = int(cy - dy_m * pixels_per_meter)
        cv2.circle(map_img, (px, py), 6, (0, 255, 0), -1)
        cv2.putText(map_img, "Target", (px+10, py+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if qr_lat is not None and qr_lon is not None:
        dy_m = (qr_lat - current_lat) * meter_per_deg_lat
        dx_m = (qr_lon - current_lon) * meter_per_deg_lon
        
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

    scale_bar_m = 100
    for s in [10, 50, 100, 500, 1000, 2000, 5000, 10000]:
        if s * pixels_per_meter > 50:
            scale_bar_m = s
            break

    cv2.line(map_img, (10, map_h-20), (10 + int(scale_bar_m * pixels_per_meter), map_h-20), (255, 255, 255), 2)
    cv2.putText(map_img, f"{scale_bar_m}m", (10, map_h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return map_img


## CONTROLLER SETUP & MAIN LOOP

# these are product of husein
def make_controllers() -> dict:
    return {
        'pitch_att': controllers.HybridController(
            pid_ctrl=controllers.PIDController(kp=0.45, ki=0.10, kd=0.08, integral_limit=20.0, output_limit=15.0,
                                               integral_zone=18.0, rate_filter_tau=0.10),
            fuzzy_ctrl=controllers.FuzzyGainScheduler(error_range=25.0, rate_range=40.0),
        ),
        'roll_att': controllers.HybridController(
            pid_ctrl=controllers.PIDController(kp=0.50, ki=0.10, kd=0.08, integral_limit=25.0, output_limit=18.0,
                                               integral_zone=20.0, rate_filter_tau=0.10),
            fuzzy_ctrl=controllers.FuzzyGainScheduler(error_range=35.0, rate_range=50.0),
        ),
        'heading': controllers.HybridController(
            pid_ctrl=controllers.PIDController(kp=1.0, ki=0.05, kd=0.08, integral_limit=90.0, output_limit=45.0,
                                               integral_zone=90.0, rate_filter_tau=0.12),
            fuzzy_ctrl=controllers.FuzzyGainScheduler(error_range=120.0, rate_range=40.0),
        ),
        'altitude': controllers.HybridController(
            pid_ctrl=controllers.PIDController(kp=1.2, ki=0.10, kd=0.05, integral_limit=60.0, output_limit=30.0,
                                               integral_zone=35.0, rate_filter_tau=0.18),
            fuzzy_ctrl=controllers.FuzzyGainScheduler(error_range=40.0, rate_range=8.0),
        ),
        'speed': controllers.HybridController(
            pid_ctrl=controllers.PIDController(kp=0.07, ki=0.03, kd=0.01, integral_limit=10.0, output_limit=0.35,
                                               integral_zone=12.0, rate_filter_tau=0.20),
            fuzzy_ctrl=controllers.FuzzyGainScheduler(error_range=12.0, rate_range=6.0),
        ),
        'vision_pan': controllers.PIDController(kp=35.0, ki=5.0, kd=10.0, output_limit=40.0),
        'vision_tilt': controllers.PIDController(kp=20.0, ki=2.0, kd=5.0, output_limit=25.0),
    }


def main_loop():
    global CURRENT_MISSION_MODE, LANDING_STATE, control, closest_enemy, is_locked_on
    cam_thread = CameraThread(start_camera_capture)

    if not MODEL_PATH.exists():
        print("Yolo Model does not exist")
        return

    model = YOLO(str(MODEL_PATH))

    ctrls = make_controllers()

    input_thread = threading.Thread(target=input_thread_func, daemon=True)
    input_thread.start()

    att_msg = connection.recv_match(type='ATTITUDE', blocking=True, timeout=2.0)
    cruise_roll_deg = math.degrees(att_msg.roll) if att_msg else 0.0
    cruise_pitch_deg = math.degrees(att_msg.pitch) if att_msg else 0.0
    cruise_yaw_deg = math.degrees(att_msg.yaw) if att_msg else 0.0

    cmd = CS.CommandState()
    cmd.target_yaw = cruise_yaw_deg

    prev_meas = {'alt': None, 'speed': None, 'alt_rate_smoothed': 0.0, 'spd_rate_smoothed': 0.0}
    prev_time = time.time()
    
    target_pitch_cmd = cruise_pitch_deg
    target_speed_cmd = 15.0
    target_throttle_cmd = GUIDED_TRIM_THROTTLE

    current_roll = cruise_roll_deg
    current_pitch = cruise_pitch_deg
    current_yaw = cruise_yaw_deg
    current_roll_rate = current_pitch_rate = current_yaw_rate = 0.0
    current_alt = 0.0
    current_spd = 15.0
    smoothed_dx = 0.0
    smoothed_dy = 0.0
    filter_alpha = 0.3
    desired_roll = 0.0

    current_lat = None
    current_lon = None
    target_lat = None
    target_lon = None
    target_alt = TAKEOFF_ALT_TARGET
    enemies = []
    qr_resp = telemetry.get_qr(session, BASE_URL, token)
    qr_enlem = optional_float(qr_resp.get("qrEnlem")) if qr_resp else None
    qr_boylam = optional_float(qr_resp.get("qrBoylam")) if qr_resp else None
    if qr_enlem is not None and qr_boylam is not None:
        print(f"[QR MISSION] QR target loaded: lat={qr_enlem:.7f}, lon={qr_boylam:.7f}")
    qr_detector = cv2.QRCodeDetector()
    if hasattr(qr_detector, "setEpsX"):
        qr_detector.setEpsX(QR_DETECTOR_EPS)
    if hasattr(qr_detector, "setEpsY"):
        qr_detector.setEpsY(QR_DETECTOR_EPS)
    qr_mission_state = "APPROACH"
    qr_data = None
    qr_last_data = None
    kamikaze_start_time = {}
    qr_dive_start_lat = None
    qr_dive_start_lon = None
    qr_dive_bearing = None
    qr_bearing_aligned_since = None
    qr_last_approach_status_time = 0.0
    qr_last_line_status_time = 0.0
    qr_last_decode_status_time = 0.0
    
    # Kilitlenme (Lock) variables
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

    target = TargetTelemetry()
    sended_qr = False

    # override = False
    hss_list = []
    last_hss_fetch = 0.0
    last_telemetry_time = 0.0
    battery = 100.0

    takeoff_completed = False
    last_takeoff_print = 0.0
    print(f"[Takeoff] Waiting for plane to climb to {TAKEOFF_ALT_TARGET - TAKEOFF_ALT_THRESH:.0f} m ...")

    while True:
        try:
            hedef_x = hedef_y = hedef_w = hedef_h = 0
            gps_time = telemetry.now_clock()
            now = time.time()
            dt = max(now - prev_time, 0.01)
            prev_time = now

            msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD', 'SYS_STATUS'], blocking=False)
            while msg is not None:
                msg_type = msg.get_type()
                if msg_type == 'ATTITUDE':
                    current_roll = math.degrees(msg.roll)
                    current_pitch = math.degrees(msg.pitch)
                    current_yaw = math.degrees(msg.yaw)
                    current_roll_rate = math.degrees(getattr(msg, 'rollspeed', 0.0))
                    current_pitch_rate = math.degrees(getattr(msg, 'pitchspeed', 0.0))
                    current_yaw_rate = math.degrees(getattr(msg, 'yawspeed', 0.0))
                elif msg_type == 'GLOBAL_POSITION_INT':
                    current_alt = msg.relative_alt / 1000.0 if msg.relative_alt > 0 else 0.0
                    current_lat = msg.lat / 1e7
                    current_lon = msg.lon / 1e7
                    # Capture home coordinates on first valid GPS fix
                    global home_lat, home_lon
                    if home_lat is None and current_lat != 0.0 and current_lon != 0.0:
                        home_lat = current_lat
                        home_lon = current_lon
                        print(f"[HOME] Home coordinates captured: ({home_lat:.6f}, {home_lon:.6f})")
                elif msg_type == 'VFR_HUD':
                    current_spd = msg.airspeed
                elif msg_type == 'SYS_STATUS':
                    battery = msg.battery_remaining if msg.battery_remaining > 0 else 0
                msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD', 'SYS_STATUS'], blocking=False)

            if CURRENT_MISSION_MODE == "enemy":
                target.update_from_server(closest_enemy)

            ## AI RELATED STUFF ##
            ret, frame = cam_thread.read()
            
            if frame is not None:
                h, w = frame.shape[:2]

                if CURRENT_MISSION_MODE == "enemy":
                    results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)
                    result = results[0]
                    now = time.time()
                    dt = max(now - prev_time, DT_MIN)
                    prev_time = now

                if now - last_hss_fetch >= 0.5:
                    try:
                        hss_data = telemetry.get_hss(session, BASE_URL, token)
                        if hss_data:
                            hss_list = hss_data
                    except Exception as e:
                        print("Failed to fetch HSS:", e)
                    last_hss_fetch = now

                cv2.circle(frame, (w // 2, h // 2), 5, (0, 255, 255), -1)
                lock_left, lock_top, lock_right, lock_bottom = lock_area_bounds(w, h)
                cv2.rectangle(frame, (lock_left, lock_top), (lock_right, lock_bottom), (0, 255, 255), 2)

                if CURRENT_MISSION_MODE == "qr":
                    last_vision_yaw = None
                    last_vision_pitch = None

                # if target plane is visible on the screen
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

                if CURRENT_MISSION_MODE != "enemy":
                    tracker = None
                    tracker_active = False

                visual_box = None
                if CURRENT_MISSION_MODE == "enemy":
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

                cmd_yaw = None
                cmd_pitch = None
                cmd_speed = None
                cmd_alt = None

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
                            yaw_control_dx = mathHelpers.clamp(damped_dx, 0.0... (31 KB left)