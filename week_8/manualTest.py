from pymavlink import mavutil
import controllers
import mathHelpers
import time
import math
import threading
import os

## GLOBAL VARIABLES ##
AXIS_BOUNDS = {
    'pitch': (-30.0, 30.0),
    'roll': (-45.0, 45.0),
    'yaw': (-180.0, 180.0),
    'alt': (50.0, 300.0),
    'speed': (13, 30)
}

TAKEOFF_ALT_TARGET = 50.0
TAKEOFF_ALT_THRESH = 5.0
DX_CONST = 0.25
DT_MIN = 0.01
PREV_MEAS_RATE_CONST = 0.75
PREARM_CONST = mavutil.mavlink.MAV_SYS_STATUS_PREARM_CHECK

connection = mavutil.mavlink_connection("udpin:127.0.0.1:14580")
connection.wait_heartbeat()
print("Connected to Fixed-Wing Vehicle...")

connection.mav.request_data_stream_send(
    connection.target_system,
    connection.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    20, 1
)

def wait_for_prearm():
    print("Waiting for pre-arm...")
    while True:
        msg = connection.recv_match(type='SYS_STATUS', blocking=True)
        if msg and (msg.onboard_control_sensors_health & PREARM_CONST == PREARM_CONST):
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
    print(f"[Takeoff] Waiting for plane to climb to {TAKEOFF_ALT_TARGET - TAKEOFF_ALT_THRESH:.0f} m ...")
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
            print(f"[Takeoff] Target altitude reached.")
            break

def make_controllers() -> dict:
    return {
        'pitch_att': controllers.HybridController(
            pid_ctrl=controllers.PIDController(kp=0.45, ki=0.10, kd=0.08, integral_limit=20.0, output_limit=15.0, integral_zone=18.0, rate_filter_tau=0.10),
            fuzzy_ctrl=controllers.FuzzyGainScheduler(error_range=25.0, rate_range=40.0),
        ),
        'roll_att': controllers.HybridController(
            pid_ctrl=controllers.PIDController(kp=0.50, ki=0.10, kd=0.08, integral_limit=25.0, output_limit=18.0, integral_zone=20.0, rate_filter_tau=0.10),
            fuzzy_ctrl=controllers.FuzzyGainScheduler(error_range=35.0, rate_range=50.0),
        ),
        'heading': controllers.HybridController(
            pid_ctrl=controllers.PIDController(kp=0.40, ki=0.035, kd=0.05, integral_limit=80.0, output_limit=35.0, integral_zone=90.0, rate_filter_tau=0.12),
            fuzzy_ctrl=controllers.FuzzyGainScheduler(error_range=120.0, rate_range=40.0),
        ),
        'altitude': controllers.HybridController(
            pid_ctrl=controllers.PIDController(kp=0.65, ki=0.08, kd=0.04, integral_limit=60.0, output_limit=18.0, integral_zone=35.0, rate_filter_tau=0.18),
            fuzzy_ctrl=controllers.FuzzyGainScheduler(error_range=40.0, rate_range=8.0),
        ),
        'speed': controllers.HybridController(
            pid_ctrl=controllers.PIDController(kp=0.07, ki=0.03, kd=0.01, integral_limit=10.0, output_limit=0.35, integral_zone=12.0, rate_filter_tau=0.20),
            fuzzy_ctrl=controllers.FuzzyGainScheduler(error_range=12.0, rate_range=6.0),
        )
    }

# Shared state for targets
targets = {
    'roll': None,
    'pitch': None,
    'yaw': None,
    'alt': TAKEOFF_ALT_TARGET,
    'speed': 15.0,
    'running': True
}
targets_lock = threading.Lock()

def input_thread_func():
    print("\n" + "="*40)
    print(" MANUAL CONTROL INTERFACE")
    print("="*40)
    print("Commands:")
    print("  yaw <val>   (e.g., yaw 90)")
    print("  pitch <val> (e.g., pitch 10)")
    print("  roll <val>  (e.g., roll 15)")
    print("  alt <val>   (e.g., alt 100)")
    print("  speed <val> (e.g., speed 25)")
    print("  clear       (resets roll/pitch/yaw to None)")
    print("  status      (prints current targets)")
    print("  quit        (exits script)")
    print("="*40 + "\n")
    
    while targets['running']:
        try:
            cmd_str = input("CMD> ").strip().lower()
            if not cmd_str: continue
            
            parts = cmd_str.split()
            cmd = parts[0]
            
            if cmd == "quit" or cmd == "exit":
                with targets_lock:
                    targets['running'] = False
                break
                
            if cmd == "status":
                with targets_lock:
                    print(f"Current Targets -> Yaw: {targets['yaw']}, Pitch: {targets['pitch']}, Roll: {targets['roll']}, Alt: {targets['alt']}, Speed: {targets['speed']}")
                continue

            if cmd == "clear":
                with targets_lock:
                    targets['roll'] = None
                    targets['pitch'] = None
                    targets['yaw'] = None
                print("Cleared attitude/heading targets.")
                continue

            if len(parts) != 2:
                print("Invalid format. Use: <param> <value>")
                continue
                
            val = float(parts[1])
            with targets_lock:
                if cmd in targets:
                    targets[cmd] = val
                    print(f"Set {cmd} target to {val}")
                else:
                    print(f"Unknown parameter: {cmd}")
        except Exception as e:
            print(f"Error: {e}")

def main_loop():
    ctrls = make_controllers()
    wait_for_takeoff()

    att_msg = connection.recv_match(type='ATTITUDE', blocking=True, timeout=2.0)
    cruise_yaw_deg = math.degrees(att_msg.yaw) if att_msg else 0.0

    print("Switching to FBWA mode for manual controller tests...")
    connection.set_mode('FBWA')
    time.sleep(0.5)

    with targets_lock:
        targets['yaw'] = cruise_yaw_deg

    # Start input thread
    input_t = threading.Thread(target=input_thread_func, daemon=True)
    input_t.start()

    prev_meas = {'alt': None, 'speed': None, 'alt_rate_smoothed': 0.0, 'spd_rate_smoothed': 0.0}
    prev_time = time.time()
    trim_thrust = 0.60
    current_thrust = 0.8

    current_roll = current_pitch = current_yaw = 0.0
    current_roll_rate = current_pitch_rate = current_yaw_rate = 0.0
    current_alt = TAKEOFF_ALT_TARGET
    current_spd = 15.0

    while True:
        with targets_lock:
            if not targets['running']: break
            t_roll = targets['roll']
            t_pitch = targets['pitch']
            t_yaw = targets['yaw']
            t_alt = targets['alt']
            t_speed = targets['speed']

        now = time.time()
        dt = max(now - prev_time, DT_MIN)
        prev_time = now

        msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD'], blocking=False)
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
            elif msg_type == 'VFR_HUD':
                current_spd = msg.airspeed
            msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD'], blocking=False)

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

        # pitch calc
        if t_alt is not None and t_pitch is None:
            low, high = AXIS_BOUNDS['pitch']
            desired_pitch = mathHelpers.clamp(ctrls['altitude'].compute(t_alt - current_alt, a_rate, dt), low, high)
        elif t_pitch is not None:
            low, high = AXIS_BOUNDS['pitch']
            desired_pitch = mathHelpers.clamp(t_pitch, low, high)
        else:
            desired_pitch = 0.0

        # roll calc
        if t_yaw is not None and t_roll is None:
            low, high = AXIS_BOUNDS['roll']
            heading_error = mathHelpers.wrap_angle_deg(t_yaw - current_yaw)
            desired_roll = mathHelpers.clamp(ctrls['heading'].compute(heading_error, -current_yaw_rate, dt), low, high)
        elif t_roll is not None:
            low, high = AXIS_BOUNDS['roll']
            desired_roll = mathHelpers.clamp(t_roll, low, high)
        else:
            desired_roll = 0.0

        pitch_error = desired_pitch - current_pitch
        roll_error = desired_roll - current_roll
        pitch_correction = ctrls['pitch_att'].compute(pitch_error, -current_pitch_rate, dt)
        roll_correction = ctrls['roll_att'].compute(roll_error, -current_roll_rate, dt)

        target_pitch = mathHelpers.clamp(desired_pitch + pitch_correction, AXIS_BOUNDS['pitch'][0], AXIS_BOUNDS['pitch'][1])
        target_roll = mathHelpers.clamp(desired_roll + roll_correction, AXIS_BOUNDS['roll'][0], AXIS_BOUNDS['roll'][1])

        # speed control
        if t_speed is not None:
            desired_thrust = mathHelpers.clamp(trim_thrust + ctrls['speed'].compute(t_speed - current_spd, err_rate, dt), 0.2, 1.0)
        else:
            desired_thrust = trim_thrust

        thrust_step_limit = 0.8 * dt
        current_thrust += mathHelpers.clamp(desired_thrust - current_thrust, -thrust_step_limit, thrust_step_limit)
        current_thrust = mathHelpers.clamp(current_thrust, 0.2, 1.0)

        # send the new vals to the plane
        connection.mav.rc_channels_override_send(
            connection.target_system, connection.target_component,
            mathHelpers.angle_to_pwm(target_roll, AXIS_BOUNDS['roll'][1]),  # Roll
            mathHelpers.angle_to_pwm(target_pitch, AXIS_BOUNDS['pitch'][1]),  # Pitch
            mathHelpers.throttle_to_pwm(current_thrust),  # Throttle
            0,  # Yaw (0 releases override to let ArduPilot auto-coordinate rudder)
            0, 0, 0, 0
        )
        time.sleep(0.05)  # 20Hz

if __name__ == '__main__':
    wait_for_prearm()
    auto_and_arm()
    main_loop()
