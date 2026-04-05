## AI MADE THIS LOOKING AT targetPlane.py I WAS TOO LAZY ##

from pymavlink import mavutil
import cv2
import math
import os
import random
import threading
import time
from dataclasses import dataclass


# =========================
# CONFIG
# =========================
TARGET_URI = 'udpin:127.0.0.1:14580'
OBSERVER_URI = 'udpin:127.0.0.1:14581'

TAKEOFF_ALT_TARGET = 70.0
TAKEOFF_ALT_THRESH = 5.0
CONTROL_HZ = 20.0

CAPTURE_DIR = 'dataset/raw'
CAPTURE_FPS = 2.0
CAMERA_FOV_DEG = 78.0

# Replace this with your Gazebo camera pipeline.
# Example for H264 over UDP:
GSTREAMER_PIPELINE = (
    'udpsrc port=5600 '
    'caps="application/x-rtp, media=video, encoding-name=H264, payload=96" ! '
    'rtpjitterbuffer ! rtph264depay ! avdec_h264 ! videoconvert ! appsink drop=true sync=false'
)

VIEW_PRESETS = [
    {'name': 'rear_center', 'forward_m': -90.0, 'right_m': 0.0, 'up_m': 10.0},
    {'name': 'rear_left',   'forward_m': -80.0, 'right_m': -30.0, 'up_m': 15.0},
    {'name': 'rear_right',  'forward_m': -80.0, 'right_m': 30.0, 'up_m': 15.0},
    {'name': 'left_side',   'forward_m': 0.0,   'right_m': -70.0, 'up_m': 8.0},
    {'name': 'right_side',  'forward_m': 0.0,   'right_m': 70.0, 'up_m': 8.0},
    {'name': 'front_high',  'forward_m': 120.0, 'right_m': 0.0, 'up_m': 30.0},
]


# =========================
# HELPERS
# =========================
def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))

def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0

def meters_per_deg_lat() -> float:
    return 111_320.0

def meters_per_deg_lon(lat_deg: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat_deg))

def offset_latlon(lat_deg: float, lon_deg: float, north_m: float, east_m: float):
    dlat = north_m / meters_per_deg_lat()
    dlon = east_m / max(meters_per_deg_lon(lat_deg), 1e-6)
    return lat_deg + dlat, lon_deg + dlon


def ned_offset_between(lat1, lon1, lat2, lon2):
    north = (lat2 - lat1) * meters_per_deg_lat()
    east = (lon2 - lon1) * meters_per_deg_lon((lat1 + lat2) * 0.5)
    return north, east


def horizontal_distance_m(lat1, lon1, lat2, lon2):
    n, e = ned_offset_between(lat1, lon1, lat2, lon2)
    return math.hypot(n, e)


def bearing_deg(lat1, lon1, lat2, lon2):
    north, east = ned_offset_between(lat1, lon1, lat2, lon2)
    return math.degrees(math.atan2(east, north)) % 360.0


def body_to_world_offsets(target_heading_deg: float, forward_m: float, right_m: float):
    yaw = math.radians(target_heading_deg)
    north = forward_m * math.cos(yaw) - right_m * math.sin(yaw)
    east = forward_m * math.sin(yaw) + right_m * math.cos(yaw)
    return north, east


def angle_to_pwm(angle: float, max_angle: float) -> int:
    constrained = clamp(angle, -max_angle, max_angle)
    return int(1500 + (constrained / max_angle) * 500)


def throttle_to_pwm(thrust_0_to_1: float) -> int:
    constrained = clamp(thrust_0_to_1, 0.0, 1.0)
    return int(1000 + constrained * 1000)


# =========================
# PID
# =========================
class PID:
    def __init__(self, kp, ki, kd, integral_limit=None, output_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def update(self, error: float, dt: float) -> float:
        dt = max(dt, 0.01)
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)

        derivative = 0.0
        if self.initialized:
            derivative = (error - self.prev_error) / dt
        else:
            self.initialized = True

        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.output_limit is not None:
            output = clamp(output, -self.output_limit, self.output_limit)
        return output


# =========================
# TELEMETRY
# =========================
@dataclass
class PlaneState:
    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_rate_deg_s: float = 0.0
    pitch_rate_deg_s: float = 0.0
    yaw_rate_deg_s: float = 0.0
    airspeed_mps: float = 15.0
    groundspeed_mps: float = 0.0
    last_update: float = 0.0
    has_fix: bool = False


class PlaneLink:
    def __init__(self, name: str, uri: str):
        self.name = name
        self.uri = uri
        self.conn = mavutil.mavlink_connection(uri)
        self.state = PlaneState()
        self.lock = threading.Lock()
        self.running = True

        print(f'[{self.name}] Connecting on {uri} ...')
        self.conn.wait_heartbeat()
        print(f'[{self.name}] Heartbeat received.')
        self.request_streams()

    def request_streams(self):
        self.conn.mav.request_data_stream_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            20,
            1,
        )

    def telemetry_loop(self):
        while self.running:
            msg = self.conn.recv_match(
                type=['GLOBAL_POSITION_INT', 'ATTITUDE', 'VFR_HUD'],
                blocking=True,
                timeout=1.0,
            )
            if msg is None:
                continue

            with self.lock:
                if msg.get_type() == 'GLOBAL_POSITION_INT':
                    self.state.lat = msg.lat / 1e7
                    self.state.lon = msg.lon / 1e7
                    self.state.alt_m = msg.relative_alt / 1000.0
                    self.state.groundspeed_mps = getattr(msg, 'vx', 0.0) / 100.0
                    self.state.has_fix = True
                elif msg.get_type() == 'ATTITUDE':
                    self.state.roll_deg = math.degrees(msg.roll)
                    self.state.pitch_deg = math.degrees(msg.pitch)
                    self.state.yaw_deg = math.degrees(msg.yaw) % 360.0
                    self.state.roll_rate_deg_s = math.degrees(getattr(msg, 'rollspeed', 0.0))
                    self.state.pitch_rate_deg_s = math.degrees(getattr(msg, 'pitchspeed', 0.0))
                    self.state.yaw_rate_deg_s = math.degrees(getattr(msg, 'yawspeed', 0.0))
                elif msg.get_type() == 'VFR_HUD':
                    self.state.airspeed_mps = getattr(msg, 'airspeed', self.state.airspeed_mps)
                self.state.last_update = time.time()

    def snapshot(self) -> PlaneState:
        with self.lock:
            return PlaneState(**self.state.__dict__)

    def set_mode(self, mode: str):
        self.conn.set_mode(mode)

    def arm(self):
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0,
        )

    def release_rc(self):
        self.conn.mav.rc_channels_override_send(
            self.conn.target_system, self.conn.target_component,
            0, 0, 0, 0, 0, 0, 0, 0,
        )

    def send_rc(self, roll_deg: float, pitch_deg: float, throttle: float, yaw_pwm: int = 1500):
        self.conn.mav.rc_channels_override_send(
            self.conn.target_system,
            self.conn.target_component,
            angle_to_pwm(roll_deg, 45.0),
            angle_to_pwm(pitch_deg, 30.0),
            throttle_to_pwm(throttle),
            yaw_pwm,
            0, 0, 0, 0,
        )


# =========================
# CAMERA CAPTURE
# =========================
class ImageCollector:
    def __init__(self, pipeline: str, out_dir: str):
        self.pipeline = pipeline
        self.out_dir = out_dir
        self.running = True
        self.capture_enabled = False
        self.meta = {
            'view_name': 'unknown',
            'distance_m': 0.0,
            'rel_bearing_deg': 0.0,
            'target_alt_m': 0.0,
            'observer_alt_m': 0.0,
        }
        self.lock = threading.Lock()
        os.makedirs(out_dir, exist_ok=True)
        self.video = None
        self.frame_id = 0
        self.last_save = 0.0

    def update_meta(self, **kwargs):
        with self.lock:
            self.meta.update(kwargs)

    def start(self):
        self.video = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.video.isOpened():
            print('[camera] Failed to open GStreamer pipeline.')
            return

        print('[camera] Capture started.')
        while self.running:
            ok, frame = self.video.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            with self.lock:
                enabled = self.capture_enabled
                meta = dict(self.meta)

            now = time.time()
            if enabled and now - self.last_save >= 1.0 / CAPTURE_FPS:
                filename = os.path.join(self.out_dir, f'img_{self.frame_id:06d}.jpg')
                cv2.imwrite(filename, frame)
                self.write_sidecar(filename, meta)
                self.frame_id += 1
                self.last_save = now

        self.video.release()

    def write_sidecar(self, image_path: str, meta: dict):
        txt_path = os.path.splitext(image_path)[0] + '.txt.meta'
        with open(txt_path, 'w', encoding='utf-8') as f:
            for k, v in meta.items():
                f.write(f'{k}: {v}\n')


# =========================
# VIEW SCHEDULER
# =========================
class ViewScheduler:
    def __init__(self):
        self.active = self.random_view()
        self.switch_time = time.time() + 8.0

    def random_view(self):
        preset = dict(random.choice(VIEW_PRESETS))
        preset['forward_m'] += random.uniform(-15.0, 15.0)
        preset['right_m'] += random.uniform(-10.0, 10.0)
        preset['up_m'] += random.uniform(-5.0, 10.0)
        return preset

    def maybe_update(self):
        now = time.time()
        if now >= self.switch_time:
            self.active = self.random_view()
            self.switch_time = now + random.uniform(6.0, 12.0)
            print(f"[view] -> {self.active['name']}  fwd={self.active['forward_m']:.1f} right={self.active['right_m']:.1f} up={self.active['up_m']:.1f}")
        return self.active


# =========================
# CONTROL
# =========================
def auto_and_arm_observer(observer: PlaneLink):
    observer.set_mode('AUTO')
    print('[observer] Mode -> AUTO')
    observer.arm()
    print('[observer] Arm command sent')


def wait_for_altitude(plane: PlaneLink, target_alt_m: float):
    print(f'[{plane.name}] Waiting for climb to {target_alt_m:.1f} m ...')
    last_print = 0.0
    while True:
        st = plane.snapshot()
        if st.has_fix and st.alt_m >= target_alt_m - TAKEOFF_ALT_THRESH:
            print(f'[{plane.name}] Reached altitude {st.alt_m:.1f} m')
            break
        if time.time() - last_print > 2.0:
            print(f'[{plane.name}] alt={st.alt_m:.1f} m')
            last_print = time.time()
        time.sleep(0.2)


def run_follow(observer: PlaneLink, target: PlaneLink, collector: ImageCollector):
    heading_pid = PID(0.55, 0.02, 0.09, integral_limit=60.0, output_limit=35.0)
    alt_pid = PID(0.10, 0.02, 0.03, integral_limit=100.0, output_limit=18.0)
    speed_pid = PID(0.06, 0.01, 0.00, integral_limit=20.0, output_limit=0.25)
    pitch_att_pid = PID(0.45, 0.05, 0.08, integral_limit=15.0, output_limit=12.0)
    roll_att_pid = PID(0.55, 0.05, 0.08, integral_limit=15.0, output_limit=15.0)

    view_scheduler = ViewScheduler()
    trim_throttle = 0.62
    current_throttle = 0.70
    prev_time = time.time()

    observer.set_mode('FBWA')
    print('[observer] Mode -> FBWA')
    time.sleep(0.5)

    while True:
        obs = observer.snapshot()
        tgt = target.snapshot()

        if not (obs.has_fix and tgt.has_fix):
            time.sleep(0.1)
            continue

        dt = max(time.time() - prev_time, 0.01)
        prev_time = time.time()

        view = view_scheduler.maybe_update()
        north_off, east_off = body_to_world_offsets(
            tgt.yaw_deg,
            view['forward_m'],
            view['right_m'],
        )
        desired_lat, desired_lon = offset_latlon(tgt.lat, tgt.lon, north_off, east_off)
        desired_alt = max(35.0, tgt.alt_m + view['up_m'])

        north_err, east_err = ned_offset_between(obs.lat, obs.lon, desired_lat, desired_lon)
        distance_to_slot = math.hypot(north_err, east_err)
        desired_heading = bearing_deg(obs.lat, obs.lon, desired_lat, desired_lon)

        heading_error = wrap_angle_deg(desired_heading - obs.yaw_deg)
        desired_roll = heading_pid.update(heading_error, dt)

        alt_error = desired_alt - obs.alt_m
        desired_pitch = alt_pid.update(alt_error, dt)

        slot_speed = clamp(max(tgt.airspeed_mps, 13.0) + clamp(distance_to_slot * 0.02, -1.0, 4.0), 13.0, 26.0)
        throttle_cmd = trim_throttle + speed_pid.update(slot_speed - obs.airspeed_mps, dt)

        roll_error = desired_roll - obs.roll_deg
        pitch_error = desired_pitch - obs.pitch_deg
        roll_cmd = desired_roll + roll_att_pid.update(roll_error, dt)
        pitch_cmd = desired_pitch + pitch_att_pid.update(pitch_error, dt)

        roll_cmd = clamp(roll_cmd, -40.0, 40.0)
        pitch_cmd = clamp(pitch_cmd, -20.0, 20.0)
        current_throttle += clamp(throttle_cmd - current_throttle, -0.8 * dt, 0.8 * dt)
        current_throttle = clamp(current_throttle, 0.35, 1.0)

        observer.send_rc(roll_cmd, pitch_cmd, current_throttle)

        target_bearing_from_observer = bearing_deg(obs.lat, obs.lon, tgt.lat, tgt.lon)
        relative_bearing = wrap_angle_deg(target_bearing_from_observer - obs.yaw_deg)
        target_distance = horizontal_distance_m(obs.lat, obs.lon, tgt.lat, tgt.lon)
        capture_ok = abs(relative_bearing) <= (CAMERA_FOV_DEG * 0.45) and 20.0 <= target_distance <= 250.0

        collector.capture_enabled = capture_ok
        collector.update_meta(
            view_name=view['name'],
            distance_m=round(target_distance, 2),
            rel_bearing_deg=round(relative_bearing, 2),
            target_alt_m=round(tgt.alt_m, 2),
            observer_alt_m=round(obs.alt_m, 2),
            slot_error_m=round(distance_to_slot, 2),
        )

        time.sleep(1.0 / CONTROL_HZ)


def main():
    target = PlaneLink('target', TARGET_URI)
    observer = PlaneLink('observer', OBSERVER_URI)

    threading.Thread(target=target.telemetry_loop, daemon=True).start()
    threading.Thread(target=observer.telemetry_loop, daemon=True).start()

    collector = ImageCollector(GSTREAMER_PIPELINE, CAPTURE_DIR)
    threading.Thread(target=collector.start, daemon=True).start()

    time.sleep(2.0)
    auto_and_arm_observer(observer)
    wait_for_altitude(observer, TAKEOFF_ALT_TARGET)
    run_follow(observer, target, collector)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopping observer...')
