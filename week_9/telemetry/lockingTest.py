## HOURS WASTED: 99999999999999

# Notes:
# need the send telemetry data


import math
import os
import time
from pathlib import Path

import cv2
import requests
from pymavlink import mavutil
from ultralytics import YOLO

import commandState as CS
import controllers
import mathHelpers
import telemetry

BASE_URL = "http://[IP_ADDRESS]"
USERNAME = "Mojave"
PASSWORD = "[PASSWORD]"
TEAM_NO = 4

## GLOBAL VARIABLES ##
AXIS_BOUNDS = {
    'pitch': (-30.0, 30.0),
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
AXIS_TURN_STRENGHT = 0.8
FOV_X_DEG = 80.0
FOV_Y_DEG = 60.0
PREARM_CONST = mavutil.mavlink.MAV_SYS_STATUS_PREARM_CHECK
MAX_DISTANCE_BETWEEN_ENEMY = 30.0
AUTONOMOUS_FLIGHT_STATUS = 1
session = requests.Session()

## HELPERS ##

connection = mavutil.mavlink_connection("udpin:127.0.0.1:14580")
connection.wait_heartbeat()
print("Connected to Fixed-Wing Vehicle...")

connection.mav.request_data_stream_send(
    connection.target_system,
    connection.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    20,
    1
)

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


def enable_gazebo_camera():
    print("[Gazebo] Sending 'enable' signal to Gazebo...")
    topic = '/world/runway/model/observer/link/base_link/sensor/nose_camera/image/enable_streaming'
    os.system(f'gz topic -t {topic} -m gz.msgs.Boolean -p "data: true"')

    time.sleep(2)
    pipeline = (
        "udpsrc port=5600 address=127.0.0.1 ! "
        "application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96 ! "
        "rtph264depay ! avdec_h264 ! videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true sync=false max-buffers=1"
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    return cap


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



def main_loop():
    # LOGIN AND OTHER STUFF
    token = telemetry.login(BASE_URL, USERNAME, PASSWORD)

    cam = enable_gazebo_camera()

    if not MODEL_PATH.exists():
        print("Yolo Model does not exist")
        return

    model = YOLO(str(MODEL_PATH))

    ctrls = make_controllers()

    wait_for_takeoff()

    if not cam.isOpened():
        print("[OpenCV] ERROR: Failed to open GStreamer pipeline.")
        return

    att_msg = connection.recv_match(type='ATTITUDE', blocking=True, timeout=2.0)
    cruise_roll_deg = math.degrees(att_msg.roll) if att_msg else 0.0
    cruise_pitch_deg = math.degrees(att_msg.pitch) if att_msg else 0.0
    cruise_yaw_deg = math.degrees(att_msg.yaw) if att_msg else 0.0

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

    current_lat = None
    current_lon = None
    target_lat = None
    target_lon = None
    target_alt = TAKEOFF_ALT_TARGET
    
    last_vision_yaw = None
    last_vision_pitch = None

    while True:
        ## AI RELATED STUFF ##
        ret, frame = cam.read()
        if not ret:
            print("Frame Cannot be Read")
            continue

        h, w = frame.shape[:2]
        results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)
        result = results[0]

        now = time.time()
        dt = max(now - prev_time, DT_MIN)
        prev_time = now

        cv2.circle(frame, (w // 2, h // 2), 5, (0, 255, 255), -1)

        # if target plane is visible on the screen
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            best_idx = confs.argmax()

            # get the target plane's coords and the confidence
            x1, y1, x2, y2 = boxes_xyxy[best_idx]
            score = float(confs[best_idx])

            # draw the bounding box and confidence string
            obj_cx, obj_cy, dx_norm, dy_norm = mathHelpers.compute_center_deviation(x1, y1, x2, y2, w, h)

            smoothed_dx = (filter_alpha * dx_norm) + ((1.0 - filter_alpha) * smoothed_dx)
            smoothed_dy = (filter_alpha * dy_norm) + ((1.0 - filter_alpha) * smoothed_dy)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (int(obj_cx), int(obj_cy)), 5, (0, 0, 255), -1)
            cv2.line(frame, (w // 2, h // 2), (int(obj_cx), int(obj_cy)), (255, 0, 0), 2)

            text1 = f"conf={score:.2f}"
            cv2.putText(frame, text1, (int(x1), max(20, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
        cv2.waitKey(1)
        ## END OF AI RELATED STUFF ##


        t_pitch, t_roll, t_yaw, t_alt, t_speed, running = cmd.snapshot()
        if not running: break
        print(f"Pitch: {t_pitch}, Roll: {t_roll}, Speed: {t_speed}")

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
                current_alt = msg.relative_alt / 1000.0
                current_lat = msg.lat / 1e7
                current_lon = msg.lon / 1e7
            elif msg_type == 'VFR_HUD':
                current_spd = msg.airspeed
            elif msg_type == 'SYS_STATUS':
                battery = msg.battery_remaining
            msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD', 'SYS_STATUS'], blocking=False)
        
        enemies = telemetry.send_telemetry(
            session,
            BASE_URL,
            token,
            TEAM_NO,
            current_lat,
            current_lon,
            current_alt,
            current_pitch,
            current_roll,
            current_yaw,
            current_spd,
            battery,
            AUTONOMOUS_FLIGHT_STATUS,
            GPS_TIME,
        )

        closest_enemy = mathHelpers.find_closest_target(current_lat, current_lon, enemies, MAX_DISTANCE_BETWEEN_ENEMY)

        if closest_enemy is not None:
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
        time.sleep(0.05)  # 20Hz


if __name__ == '__main__':
    wait_for_prearm()
    auto_and_arm()
    main_loop()
