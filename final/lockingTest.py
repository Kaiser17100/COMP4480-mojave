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
import subprocess
import socket

CURRENT_MISSION_MODE = "normal"
BASE_URL = "http://192.168.10.2:10001"
USERNAME = "4"
PASSWORD = "4"
TEAM_NO = 4

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
MAX_DISTANCE_BETWEEN_ENEMY = 30.0
AUTONOMOUS_FLIGHT_STATUS = 1
session = requests.Session()
hss_list = []


## HELPERS ##

connection = mavutil.mavlink_connection("udpin:192.168.10.1:14580")
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
    global CURRENT_MISSION_MODE
    while True:
        try:
            val = input("Enter mission mode ('enemy', 'qr' or 'normal'): ").strip().lower()
            if val in ["enemy", "qr", "normal"]:
                CURRENT_MISSION_MODE = val
                print(f"[MISSION] Mode switched to: {CURRENT_MISSION_MODE}")
            else:
                print(f"[MISSION] Invalid mode '{val}'. Use 'enemy', 'qr' or 'normal'.")
        except:
            break


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
        now = time.time()
        if now - last_print >= 2.0:
            print(f"[Takeoff]   alt = {alt_m:.1f} m  (target {TAKEOFF_ALT_TARGET:.0f} m)")
            last_print = now

        if alt_m >= TAKEOFF_ALT_TARGET - TAKEOFF_ALT_THRESH:
            print(f"[Takeoff] Target altitude reached. Switching to FBWA outer loops ...")
            break


# =========================
# RTP/H264 UDP GİRİŞ
# =========================
UDP_IN_PORT = 5600

# =========================
# ÇIKIŞ JPEG UDP
# =========================
UDP_OUT_IP = "127.0.0.1"
UDP_OUT_PORT = 5426

# =========================
# Görüntü boyutu
# =========================
WIDTH = 640
HEIGHT = 480
JPEG_QUALITY = 50

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def create_sdp_file():
    sdp_text = f"""v=0
o=- 0 0 IN IP4 127.0.0.1
s=RTP H264 Stream
c=IN IP4 0.0.0.0
t=0 0
m=video {UDP_IN_PORT} RTP/AVP 96
a=rtpmap:96 H264/90000
a=recvonly
"""
    sdp_path = "/tmp/udp_h264_stream.sdp"
    with open(sdp_path, "w") as f:
        f.write(sdp_text)
    return sdp_path

def print_ffmpeg_errors(process):
    while True:
        line = process.stderr.readline()
        if not line:
            break
        try:
            print("[FFmpeg]", line.decode(errors="ignore").strip())
        except Exception:
            pass

def start_ffmpeg():
    sdp_path = create_sdp_file()
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-protocol_whitelist", "file,udp,rtp",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-analyzeduration", "1000000",
        "-probesize", "1000000",
        "-i", sdp_path,
        "-vf", f"scale={WIDTH}:{HEIGHT}",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-"
    ]
    print("FFmpeg başlatılıyor:")
    print(" ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8
    )
    stderr_thread = threading.Thread(
        target=print_ffmpeg_errors,
        args=(process,),
        daemon=True
    )
    stderr_thread.start()
    return process


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
            pid_ctrl=controllers.PIDController(kp=0.40, ki=0.035, kd=0.05, integral_limit=80.0, output_limit=35.0,
                                               integral_zone=90.0, rate_filter_tau=0.12),
            fuzzy_ctrl=controllers.FuzzyGainScheduler(error_range=120.0, rate_range=40.0),
        ),
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
        'vision_pan': controllers.PIDController(kp=35.0, ki=5.0, kd=10.0, output_limit=40.0),
        'vision_tilt': controllers.PIDController(kp=20.0, ki=2.0, kd=5.0, output_limit=25.0),
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
    def __init__(self, ffmpeg_process):
        self.ffmpeg_process = ffmpeg_process
        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.frame_size = WIDTH * HEIGHT * 3
        self.last_error_time = 0
        # Start the background thread instantly without waiting for a frame
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while True:
            if self.ffmpeg_process.poll() is not None:
                print("FFmpeg kapandı. Stream açılamadı veya giriş formatı uyumsuz.")
                break

            raw_frame = self.ffmpeg_process.stdout.read(self.frame_size)

            if len(raw_frame) != self.frame_size:
                now = time.time()
                if now - self.last_error_time > 1.0:
                    print("Görüntü frame'i eksik geldi veya FFmpeg henüz frame üretmedi.")
                    self.last_error_time = now
                time.sleep(0.01)
                continue

            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((HEIGHT, WIDTH, 3))

            with self.lock:
                self.ret = True
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame

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

def main_loop():
    global CURRENT_MISSION_MODE
    global closest_enemy
    input_thread = threading.Thread(target=input_thread_func, daemon=True)
    input_thread.start()

    ffmpeg_process = start_ffmpeg()

    if not MODEL_PATH.exists():
        print("Yolo Model does not exist")
        return

    model = YOLO(str(MODEL_PATH))

    ctrls = make_controllers()

    wait_for_takeoff()

    if ffmpeg_process.poll() is not None:
        print("[OpenCV] ERROR: Failed to open FFmpeg pipeline.")
        return

    cam_thread = CameraThread(ffmpeg_process)
    threading.Thread(target=fetch_hss_thread, daemon=True).start()

    reader_thread = threading.Thread(target=mavlink_reader_thread, daemon=True)
    reader_thread.start()
    time.sleep(1.0)

    cruise_roll_deg = telemetry_data['roll']
    cruise_pitch_deg = telemetry_data['pitch']
    cruise_yaw_deg = telemetry_data['yaw']

    connection.set_mode('FBWA')
    time.sleep(0.5)

    cmd = CS.CommandState()
    cmd.target_yaw = cruise_yaw_deg

    prev_meas = {'alt': None, 'speed': None, 'alt_rate_smoothed': 0.0, 'spd_rate_smoothed': 0.0}
    prev_time = time.time()
    trim_thrust = 0.60
    current_thrust = 0.8

    current_roll = cruise_roll_deg
    current_pitch = cruise_pitch_deg
    current_yaw = cruise_yaw_deg
    current_roll_rate = current_pitch_rate = current_yaw_rate = 0.0
    current_alt = TAKEOFF_ALT_TARGET
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
    qr_enlem = qr_resp.get("qrEnlem") if qr_resp else None
    qr_boylam = qr_resp.get("qrBoylam") if qr_resp else None
    qr_detector = cv2.QRCodeDetector()
    qr_mission_state = "APPROACH"
    qr_data = None
    kamikaze_start_time = {}
    
    # Kilitlenme (Lock) variables
    is_locked_on = False
    lock_start_time = 0.0
    LOCK_REQUIRED_TIME = 4.0  # Başarılı kilitlenme sayılması için hedefin merkezde kalması gereken minimum süre (sn)
    LOCK_BBOX_THRESHOLD = 0.15 # Kameranın merkezinden %15 sapma payı (Kilitlenme kutusu boyutu)

    last_vision_yaw = None
    last_vision_pitch = None
    sended_qr = False

    # override = False
    global hss_list
    last_telemetry_time = 0.0
    battery = 100.0

    while True:
        now = time.time()
        dt = max(now - prev_time, DT_MIN)
        prev_time = now

        # Retrieve the freshest telemetry data from the background thread
        current_roll = telemetry_data['roll']
        current_pitch = telemetry_data['pitch']
        current_yaw = telemetry_data['yaw']
        current_roll_rate = telemetry_data['roll_rate']
        current_pitch_rate = telemetry_data['pitch_rate']
        current_yaw_rate = telemetry_data['yaw_rate']
        current_alt = telemetry_data['alt']
        current_lat = telemetry_data['lat']
        current_lon = telemetry_data['lon']
        current_spd = telemetry_data['spd']
        battery = telemetry_data['battery']

        hedef_x = hedef_y = hedef_w = hedef_h = 0
        gps_time = telemetry.now_clock()
        ## AI RELATED STUFF ##
        ret, frame = cam_thread.read()
        if not ret:
            print("Frame Cannot be Read")

        if ret:
            h, w = frame.shape[:2]

            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if success:
                send_sock.sendto(buffer.tobytes(), (UDP_OUT_IP, UDP_OUT_PORT))

            if CURRENT_MISSION_MODE == "enemy":
                results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)
                result = results[0]

            # Draw locking zone on screen
            lock_box_w = int(w * LOCK_BBOX_THRESHOLD * 2)
            lock_box_h = int(h * LOCK_BBOX_THRESHOLD * 2)
            box_color = (0, 0, 255) if is_locked_on else (255, 255, 0)
            cv2.rectangle(frame, (w//2 - lock_box_w//2, h//2 - lock_box_h//2), (w//2 + lock_box_w//2, h//2 + lock_box_h//2), box_color, 2)
            cv2.circle(frame, (w // 2, h // 2), 5, (0, 255, 255), -1)

            if CURRENT_MISSION_MODE == "qr":
                last_vision_yaw = None
                last_vision_pitch = None

            # if target plane is visible on the screen
            if CURRENT_MISSION_MODE == "enemy" and result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                best_idx = confs.argmax()

                # get the target plane's coords and the confidence
                x1, y1, x2, y2 = boxes_xyxy[best_idx]
                score = float(confs[best_idx])

                # draw the bounding box and confidence string
                obj_cx, obj_cy, dx_norm, dy_norm = mathHelpers.compute_center_deviation(x1, y1, x2, y2, w, h)

                hedef_x = int(obj_cx)
                hedef_y = int(obj_cy)
                hedef_w = int(x2 - x1)
                hedef_h = int(y2 - y1)

                smoothed_dx = (filter_alpha * dx_norm) + ((1.0 - filter_alpha) * smoothed_dx)
                smoothed_dy = (filter_alpha * dy_norm) + ((1.0 - filter_alpha) * smoothed_dy)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(obj_cx), int(obj_cy)), 5, (0, 0, 255), -1)
                cv2.line(frame, (w // 2, h // 2), (int(obj_cx), int(obj_cy)), (255, 0, 0), 2)

                text1 = f"conf={score:.2f}"
                cv2.putText(frame, text1, (int(x1), max(20, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Kilitlenme Süreci Kontrolü (Hedef merkez kutusunda mı?)
                if abs(dx_norm) < LOCK_BBOX_THRESHOLD and abs(dy_norm) < LOCK_BBOX_THRESHOLD:
                    if not is_locked_on:
                        is_locked_on = True
                        lock_start_time = now
                        print("[LOCK] Kilitlenme BAŞLADI!")
                    else:
                        current_lock_dur = now - lock_start_time
                        cv2.putText(frame, f"LOCKING: {current_lock_dur:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    if is_locked_on:
                        lock_duration = now - lock_start_time
                        is_locked_on = False
                        print(f"[LOCK] Hedef merkezden çıktı. Kilitlenme koptu. Süre: {lock_duration:.1f}s")
                        if lock_duration >= LOCK_REQUIRED_TIME:
                            print(f"[LOCK] {LOCK_REQUIRED_TIME} saniyeyi aştığı için BAŞARILI kilitlenme sayıldı! API'ye gönderiliyor...")
                            try:
                                telemetry.send_lock(session, BASE_URL, token, AUTONOMOUS_FLIGHT_STATUS, telemetry.now_clock())
                            except Exception as e:
                                print("[LOCK] API Hatası:", e)

                # Convert visual offsets to real-world heading and pitch targets
                angle_offset = smoothed_dx * (FOV_X_DEG / 2.0)
                desired_yaw_from_vision = mathHelpers.wrap_angle_deg(current_yaw + angle_offset)
                
                pitch_offset = -smoothed_dy * (FOV_Y_DEG / 2.0)
                desired_pitch_from_vision = current_pitch + pitch_offset
                
                last_vision_yaw = desired_yaw_from_vision
                last_vision_pitch = desired_pitch_from_vision

                cmd.update('yaw', desired_yaw_from_vision)
                cmd.update('pitch', desired_pitch_from_vision)
                cmd.update('roll', None)
                cmd.update('alt', None)

                # Distance approximation heuristic for speed control
                # We use GPS distance to maintain a solid follow distance and keep the target in frame
                FOLLOW_DISTANCE = 25.0  # safe distance in meters
                if current_lat is not None and target_lat is not None:
                    dist = mathHelpers.get_distance(current_lat, current_lon, target_lat, target_lon)
                    dist_error = dist - FOLLOW_DISTANCE
                    speed_correction = dist_error * 0.8
                    desired_speed = mathHelpers.clamp(20.0 + speed_correction, AXIS_BOUNDS['speed'][0], AXIS_BOUNDS['speed'][1])
                    cmd.update('speed', desired_speed)

                else:
                    box_area = (x2 - x1) * (y2 - y1)
                    frame_area = w * h
                    area_ratio = box_area / frame_area

                    target_area_ratio = 0.02  # Assuming target occupies ~2% of the frame when at ideal following distance
                    area_error = target_area_ratio - area_ratio
                    speed_correction = area_error * 100.0

                    desired_speed = mathHelpers.clamp(20.0 + speed_correction, AXIS_BOUNDS['speed'][0], AXIS_BOUNDS['speed'][1])
                    cmd.update('speed', desired_speed)

            else:
                if is_locked_on:
                    lock_duration = now - lock_start_time
                    is_locked_on = False
                    print(f"[LOCK] Hedef kaybedildi. Kilitlenme koptu. Süre: {lock_duration:.1f}s")
                    if lock_duration >= LOCK_REQUIRED_TIME:
                        print(f"[LOCK] {LOCK_REQUIRED_TIME} saniyeyi aştığı için BAŞARILI kilitlenme sayıldı! API'ye gönderiliyor...")
                        try:
                            telemetry.send_lock(session, BASE_URL, token, AUTONOMOUS_FLIGHT_STATUS, telemetry.now_clock())
                        except Exception as e:
                            print("[LOCK] API Hatası:", e)

                # Acknowledgement and memory system
                if last_vision_yaw is not None and last_vision_pitch is not None:
                    yaw_reached = abs(mathHelpers.wrap_angle_deg(current_yaw - last_vision_yaw)) <= 5.0
                    pitch_reached = abs(current_pitch - last_vision_pitch) <= 5.0
                    
                    if yaw_reached and pitch_reached:
                        last_vision_yaw = None
                        last_vision_pitch = None
                    else:
                        cmd.update('yaw', last_vision_yaw)
                        cmd.update('pitch', last_vision_pitch)
                        cmd.update('roll', None)
                        cmd.update('alt', None)
                        cmd.update('speed', 20.0)
                        
                if last_vision_yaw is None: # We either never saw it, or we reached the target and lost it
                    cmd.update('roll', None)
                    cmd.update('pitch', None)
                    cmd.update('speed', 20.0)

                    if current_lat is not None and target_lat is not None:
                        desired_yaw = mathHelpers.get_bearing(current_lat, current_lon, target_lat, target_lon)
                        cmd.update('yaw', desired_yaw)
                        cmd.update('alt', target_alt)
                    else:
                        if cmd.snapshot()[2] is None:
                            cmd.update('yaw', current_yaw)
                        cmd.update('alt', TAKEOFF_ALT_TARGET)

            cv2.imshow("YOLOv8 Pose UDP Inference", frame)

        # Move the minimap and waitKey OUTSIDE the 'if ret:' block!
        if closest_enemy is not None:
            map_img = draw_minimap(current_lat, current_lon, current_yaw, hss_list, FLIGHT_BOUNDARIES, closest_enemy['iha_enlem'], closest_enemy['iha_boylam'], qr_enlem, qr_boylam)
        else:
            map_img = draw_minimap(current_lat, current_lon, current_yaw, hss_list, FLIGHT_BOUNDARIES, None, None, qr_enlem, qr_boylam)
        
        cv2.imshow("Minimap", map_img)
        cv2.waitKey(1)
        ## END OF AI RELATED STUFF ##

        # sending the telemetry data to the server in a background thread
        if now - last_telemetry_time >= 0.5:
            if current_lat is not None and current_lon is not None:
                kilitlenme = 1 if (last_vision_yaw is not None) else 0
                
                # Pack the arguments
                telemetry_args = {
                    'session': session, 'base_url': BASE_URL, 'token': token, 'team_no': TEAM_NO,
                    'lat': current_lat, 'lon': current_lon, 'alt': current_alt,
                    'pitch': current_pitch, 'yaw': current_yaw % 360, 'roll': current_roll,
                    'spd': current_spd, 'battery': battery, 'otonom': AUTONOMOUS_FLIGHT_STATUS,
                    'gps_time': gps_time, 'kilit': kilitlenme,
                    'hx': hedef_x, 'hy': hedef_y, 'hw': hedef_w, 'hh': hedef_h
                }
                
                # Fire and forget! Loop continues immediately.
                threading.Thread(target=async_send_telemetry, args=(telemetry_args,), daemon=True).start()
                
            last_telemetry_time = now


        t_pitch, t_roll, t_yaw, t_alt, t_speed, running = cmd.snapshot()
        if not running: break

        ## QR MISSION
        if CURRENT_MISSION_MODE == "qr":
            if qr_enlem is not None and qr_boylam is not None and current_lat is not None and current_lon is not None:
                dist_to_qr = mathHelpers.get_distance(current_lat, current_lon, qr_enlem, qr_boylam)
                
                if qr_mission_state == "APPROACH":
                    target_lat = qr_enlem
                    target_lon = qr_boylam
                    target_alt = 110.0
                    if dist_to_qr < 170.0 and current_alt > 100.0:
                        kamikaze_start_time = telemetry.now_clock()
                        qr_mission_state = "DIVE"
                        print("[QR MISSION] Close to QR! Initiating DIVE!")
                
                elif qr_mission_state == "DIVE":
                    # Start with default dive pitch, but allow vision to override it
                    t_pitch = -35.0
                    t_alt = None
                    qr_vision_yaw = None
                    qr_data, bbox = None, None
                    
                    if ret:
                        try:
                            qr_data, bbox, _ = qr_detector.detectAndDecode(frame)
                        except cv2.error:
                            pass
                            
                        if bbox is not None:
                            pts = bbox[0]
                            # Calculate the center of the QR Code
                            qr_cx = sum(p[0] for p in pts) / 4.0
                            qr_cy = sum(p[1] for p in pts) / 4.0
                            
                            # --- YAW (X-Axis) CORRECTION ---
                            dx_norm = (qr_cx - w/2) / (w/2)
                            angle_offset = dx_norm * (FOV_X_DEG / 2.0)
                            qr_vision_yaw = mathHelpers.wrap_angle_deg(current_yaw + angle_offset)
                            
                            # --- PITCH (Y-Axis) CORRECTION ---
                            # Calculate how far off-center the QR is vertically
                            dy_norm = (qr_cy - h/2) / (h/2)
                            pitch_offset = -dy_norm * (FOV_Y_DEG / 2.0)
                            
                            # Update target pitch based on vision, clamped to safe dive angles
                            vision_pitch = current_pitch + pitch_offset
                            t_pitch = mathHelpers.clamp(vision_pitch, -45.0, -15.0)
                                
                            # Draw box and center-point for visual feedback
                            pts = np.int32(pts).reshape(-1, 1, 2)
                            cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
                            cv2.circle(frame, (int(qr_cx), int(qr_cy)), 5, (0, 255, 255), -1)

                    if qr_data:
                        print(f"[QR MISSION] QR detected: {qr_data}")
                        try:
                            if qr_resp and not sended_qr:
                                kamikaze_zaman = telemetry.now_clock()
                                telemetry.send_kamikaze(session, BASE_URL, token, TEAM_NO, qr_data, kamikaze_start_time, kamikaze_zaman)
                                sended_qr = True
                                print("[QR MISSION] QR read successfully! Returning to normal flight.")
                                qr_mission_state = "PULLOUT"
                        except Exception as e:
                            print(f"[QR] QR could not be sent: {e}")
                            
                    if qr_vision_yaw is not None:
                        t_yaw = qr_vision_yaw
                    
                    if current_alt < 45.0:
                        qr_mission_state = "PULLOUT"
                        CURRENT_MISSION_MODE = "normal"
                        print("[QR MISSION] Alt < 45m! Aborting dive, pulling out!")
                        
                elif qr_mission_state == "PULLOUT":
                    target_lat = qr_enlem
                    target_lon = qr_boylam
                    target_alt = TAKEOFF_ALT_TARGET
                    print("[QR MISSION] Returning to normal flight.")
                    qr_mission_state = "APPROACH"
                    CURRENT_MISSION_MODE = "normal"
            else:
                target_lat = None
                target_lon = None

        if CURRENT_MISSION_MODE == "normal":
            target_lat = None
            target_lon = None

        ## locating the enemy
        elif closest_enemy is not None and CURRENT_MISSION_MODE == "enemy":
            target_lat = closest_enemy["iha_enlem"]
            target_lon = closest_enemy["iha_boylam"]
            target_alt = closest_enemy["iha_irtifa"]

        # alt rate calc
        if prev_meas['alt'] is None:
            prev_meas['alt'] = current_alt
        dA = current_alt - prev_meas['alt']
        a_rate = (DX_CONST * (-dA / dt)) + (PREV_MEAS_RATE_CONST * prev_meas['alt_rate_smoothed'])
        a_rate = mathHelpers.clamp(a_rate, -12.0, 12.0)
        prev_meas['alt_rate_smoothed'] = a_rate
        prev_meas['alt'] = current_alt

        # speed rate calc
        if prev_meas['speed'] is None:
            prev_meas['speed'] = current_spd
        spd_delta = current_spd - prev_meas['speed']
        err_rate = (DX_CONST * (-spd_delta / dt)) + (PREV_MEAS_RATE_CONST * prev_meas['spd_rate_smoothed'])
        err_rate = mathHelpers.clamp(err_rate, -8.0, 8.0)
        prev_meas['spd_rate_smoothed'] = err_rate
        prev_meas['speed'] = current_spd

        # pitch calc dependent on alt
        if t_alt is not None:
            low, high = AXIS_BOUNDS['pitch']
            desired_pitch = mathHelpers.clamp(ctrls['altitude'].compute(t_alt - current_alt, a_rate, dt), low, high)

        # normal pitch calc
        elif t_pitch is not None:
            low, high = AXIS_BOUNDS['pitch']
            desired_pitch = mathHelpers.clamp(t_pitch, low, high)
        else:
            desired_pitch = 0.0

        if current_lat is not None and current_lon is not None:
            safe_yaw = current_yaw
            if hss_list:
                safe_yaw = mathHelpers.compute_apf_hss(
                    current_lat, current_lon, safe_yaw, current_spd, safe_yaw, hss_list
                )
            if FLIGHT_BOUNDARIES:
                safe_yaw = mathHelpers.enforce_flight_boundaries(
                    current_lat, current_lon, safe_yaw, current_spd, safe_yaw, FLIGHT_BOUNDARIES
                )
            if safe_yaw != current_yaw:
                print("OBSTACLE / BOUNDARY AHEAD")
                cmd.update('yaw', safe_yaw)
                cmd.update('roll', None)  # Override direct roll commands to steer away
                t_yaw = safe_yaw
                t_roll = None

        # roll calc dependent on yaw
        if t_yaw is not None:
            low, high = AXIS_BOUNDS['roll']
            heading_error = mathHelpers.wrap_angle_deg(t_yaw - current_yaw)
            desired_roll = mathHelpers.clamp(ctrls['heading'].compute(heading_error, -current_yaw_rate, dt), low, high)

        # normal roll calc
        elif t_roll is not None:
            low, high = AXIS_BOUNDS['roll']
            desired_roll = mathHelpers.clamp(t_roll, low, high)
        else:
            desired_roll = 0.0

        # put the desired vals at hybrid controller
        pitch_error = desired_pitch - current_pitch
        roll_error = desired_roll - current_roll
        pitch_correction = ctrls['pitch_att'].compute(pitch_error, -current_pitch_rate, dt)
        roll_correction = ctrls['roll_att'].compute(roll_error, -current_roll_rate, dt)

        low, high = AXIS_BOUNDS['pitch']
        target_pitch = mathHelpers.clamp(desired_pitch + pitch_correction, low, high)

        low, high = AXIS_BOUNDS['roll']
        target_roll = mathHelpers.clamp(desired_roll + roll_correction, low, high)

        # speed control / desired thrust calc
        if t_speed is not None:
            desired_thrust = mathHelpers.clamp(
                trim_thrust + ctrls['speed'].compute(t_speed - current_spd, err_rate, dt), 0.2, 1.0)
        else:
            desired_thrust = trim_thrust

        # updated thrust calculations with speed control
        thrust_step_limit = 0.8 * dt
        current_thrust += mathHelpers.clamp(desired_thrust - current_thrust, -thrust_step_limit, thrust_step_limit)
        current_thrust = mathHelpers.clamp(current_thrust, 0.2, 1.0)

        # send the dnew vals to the plane
        connection.mav.rc_channels_override_send(
            connection.target_system, connection.target_component,
            mathHelpers.angle_to_pwm(target_roll, AXIS_BOUNDS['roll'][1]),  # Roll
            mathHelpers.angle_to_pwm(target_pitch, AXIS_BOUNDS['pitch'][1]),  # Pitch
            mathHelpers.throttle_to_pwm(current_thrust),  # Throttle
            1500,  # Yaw
            0, 0, 0, 0
        )


if __name__ == '__main__':
    token = telemetry.login(session, BASE_URL, USERNAME, PASSWORD)
    wait_for_prearm()
    auto_and_arm()
    main_loop()
