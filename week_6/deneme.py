from pymavlink import mavutil
import time
import os
import math
import threading
import cv2  # Added for the camera stream

## GLOBAL VARIABLES ##
AXIS_BOUNDS = {
    'pitch': (-30.0, 30.0),
    'roll': (-45.0, 45.0),
    'yaw': (-180.0, 180.0),
    'alt': (-300.0, 300.0),
    'speed': (13, 30)
}

TAKEOFF_ALT_TARGET = 50.0
TAKEOFF_ALT_THRESH = 5.0

connection = mavutil.mavlink_connection('udp:127.0.0.1:14550')
connection.wait_heartbeat()
print("Connected to Fixed-Wing Vehicle...")

connection.mav.request_data_stream_send(
    connection.target_system,
    connection.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    20,
    1
)

def enable_gazebo_camera():
    """Waits 2 seconds, then tells Gazebo to start streaming."""
    print("[Gazebo] Waiting 2 seconds for OpenCV to get ready...")
    time.sleep(2.0)
    print("[Gazebo] Sending 'enable' signal to Gazebo...")
    topic = '/world/runway/model/uav_1/link/base_link/sensor/nose_camera/image/enable_streaming'
    os.system(f'gz topic -t {topic} -m gz.msgs.Boolean -p "data: 1"')


# ?????????????????????????????????????????????
#  CONTROLLERS
# ?????????????????????????????????????????????

def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


class PIDController:
    def __init__(
        self,
        kp=12.0,
        ki=0.5,
        kd=0.6,
        integral_limit=50.0,
        output_limit=None,
        integral_zone=None,
        rate_filter_tau=0.15,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.integral_zone = integral_zone
        self.rate_filter_tau = rate_filter_tau
        self._integral = 0.0
        self._filtered_rate = 0.0

    def reset(self):
        self._integral = 0.0
        self._filtered_rate = 0.0

    def compute(
        self,
        error: float,
        error_rate: float,
        dt: float,
        kp_scale: float = 1.0,
        ki_scale: float = 1.0,
        kd_scale: float = 1.0,
    ) -> float:
        dt = max(dt, 0.01)

        if self.rate_filter_tau <= 0.0:
            self._filtered_rate = error_rate
        else:
            alpha = dt / (self.rate_filter_tau + dt)
            self._filtered_rate += alpha * (error_rate - self._filtered_rate)

        prev_integral = self._integral
        integral_active = self.integral_zone is None or abs(error) <= self.integral_zone

        if integral_active and self.ki > 0.0 and ki_scale > 0.0:
            self._integral += error * dt
            self._integral = clamp(self._integral, -self.integral_limit, self.integral_limit)
        else:
            self._integral *= 0.98

        p_term = self.kp * kp_scale * error
        i_term = self.ki * ki_scale * self._integral
        d_term = self.kd * kd_scale * self._filtered_rate
        raw_output = p_term + i_term + d_term

        if self.output_limit is None:
            return raw_output

        output = clamp(raw_output, -self.output_limit, self.output_limit)
        if output != raw_output:
            saturated_high = raw_output > self.output_limit and error > 0.0
            saturated_low = raw_output < -self.output_limit and error < 0.0
            if saturated_high or saturated_low:
                self._integral = prev_integral
                i_term = self.ki * ki_scale * self._integral
                raw_output = p_term + i_term + d_term
                output = clamp(raw_output, -self.output_limit, self.output_limit)

        return output


class FuzzyGainScheduler:
    def __init__(self, error_range=180.0, rate_range=90.0):
        self.error_range = error_range
        self.rate_range = rate_range
        self.labels = ['S', 'M', 'L']
        self.kp_rules = [
            [0.75, 0.90, 0.70],
            [1.15, 1.35, 1.10],
            [1.85, 2.20, 1.65],
        ]
        self.ki_rules = [
            [1.60, 1.15, 0.65],
            [0.95, 0.75, 0.45],
            [0.35, 0.25, 0.20],
        ]
        self.kd_rules = [
            [0.75, 1.25, 1.85],
            [0.95, 1.45, 2.10],
            [1.10, 1.75, 2.35],
        ]

    @staticmethod
    def _triangle(x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        if a < x <= b:
            return (x - a) / (b - a)
        if b < x < c:
            return (c - x) / (c - b)
        return 0.0

    @staticmethod
    def _left_shoulder(x, a, b):
        if x <= a:
            return 1.0
        if x >= b:
            return 0.0
        return (b - x) / (b - a)

    @staticmethod
    def _right_shoulder(x, a, b):
        if x <= a:
            return 0.0
        if x >= b:
            return 1.0
        return (x - a) / (b - a)

    def _fuzzify_abs(self, normalized_value):
        x = clamp(normalized_value, 0.0, 1.0)
        return {
            'S': self._left_shoulder(x, 0.20, 0.45),
            'M': self._triangle(x, 0.20, 0.55, 0.85),
            'L': self._right_shoulder(x, 0.55, 0.90),
        }

    def _blend(self, rules, err_mu, rate_mu, default_value=1.0):
        numerator = 0.0
        denominator = 0.0
        for i, err_label in enumerate(self.labels):
            for j, rate_label in enumerate(self.labels):
                weight = min(err_mu[err_label], rate_mu[rate_label])
                if weight <= 0.0:
                    continue
                numerator += weight * rules[i][j]
                denominator += weight
        return default_value if denominator == 0.0 else numerator / denominator

    def compute_scales(self, error: float, error_rate: float) -> dict:
        err_mu = self._fuzzify_abs(abs(error) / self.error_range)
        rate_mu = self._fuzzify_abs(abs(error_rate) / self.rate_range)
        return {
            'kp_scale': self._blend(self.kp_rules, err_mu, rate_mu, 1.0),
            'ki_scale': self._blend(self.ki_rules, err_mu, rate_mu, 1.0),
            'kd_scale': self._blend(self.kd_rules, err_mu, rate_mu, 1.0),
        }


class HybridController:
    def __init__(self, pid_ctrl, fuzzy_ctrl):
        self.pid = pid_ctrl
        self.fuzzy = fuzzy_ctrl

    def reset(self):
        self.pid.reset()

    def compute(self, error: float, rate: float, dt: float) -> float:
        gain_scales = self.fuzzy.compute_scales(error, rate)
        return self.pid.compute(error, rate, dt, **gain_scales)


# ?????????????????????????????????????????????
#  SHARED COMMAND STATE
# ?????????????????????????????????????????????
class CommandState:
    def __init__(self):
        self._lock = threading.Lock()
        self.target_pitch = None
        self.target_roll = None
        self.target_yaw = None
        self.target_alt = None
        self.target_speed = None
        self.running = True
        self.override = False

    def update(self, axis: str, value):
        with self._lock: setattr(self, f'target_{axis}', value)

    def set_override(self, state: bool):
        with self._lock: self.override = state

    def snapshot(self):
        with self._lock:
            return (self.target_pitch, self.target_roll, self.target_yaw, self.target_alt, self.target_speed,
                    self.running, self.override)

    def stop(self):
        with self._lock: self.running = False


# ?????????????????????????????????????????????
#  HELPERS FOR FBWA & RC OVERRIDE
# ?????????????????????????????????????????????
def angle_to_pwm(angle: float, max_angle: float) -> int:
    constrained_angle = max(-max_angle, min(max_angle, angle))
    return int(1500 + (constrained_angle / max_angle) * 500)


def throttle_to_pwm(thrust_0_to_1: float) -> int:
    constrained_thrust = max(0.0, min(1.0, thrust_0_to_1))
    return int(1000 + (constrained_thrust * 1000))


def release_rc_overrides():
    connection.mav.rc_channels_override_send(
        connection.target_system, connection.target_component, 0, 0, 0, 0, 0, 0, 0, 0
    )


def auto_and_arm():
    connection.set_mode('AUTO')
    print("Setting Mode to Auto...")
    connection.mav.command_long_send(
        connection.target_system, connection.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
    )
    print("Arming...")


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


# ?????????????????????????????????????????????
#  THREADS (INPUT & CAMERA)
# ?????????????????????????????????????????????
def input_thread(cmd: CommandState, ctrl_label: str):
    print(f"\n[Input] Controller : {ctrl_label}")
    print("[Input] Commands   : pitch <deg>  roll <deg>  yaw <deg>  alt <val>  speed <val>  reset  quit")
    print("[Input] Note       : pitch/roll are attitude setpoints, yaw is heading hold.\n")
    while cmd.running:
        try:
            raw = input("cmd> ").strip().lower()
        except EOFError:
            break

        if not raw: continue
        if raw in ('quit', 'exit', 'q'):
            print("[Input] Stopping...")
            cmd.stop()
            break

        if cmd.snapshot()[6]:
            print("[Input] COMMAND IGNORED: Plane is currently in emergency pull-up!")
            continue

        if raw == 'reset':
            cmd.update('pitch', 0.0);
            cmd.update('roll', 0.0);
            cmd.update('yaw', None);
            cmd.update('alt', None)
            cmd.update('speed', None)
            continue

        parts = raw.split()
        if len(parts) != 2: continue

        axis, val_str = parts
        if axis not in AXIS_BOUNDS: continue
        try:
            value = float(val_str)
        except ValueError:
            continue

        if axis == 'roll':
            cmd.update('roll', value); cmd.update('yaw', None)
        elif axis == 'yaw':
            cmd.update('yaw', value); cmd.update('roll', None)
        elif axis == 'alt':
            cmd.update('pitch', None); cmd.update('alt', value)
        elif axis == 'pitch':
            cmd.update('alt', None); cmd.update('pitch', value)
        else:
            cmd.update(axis, value)


def camera_thread(cmd: CommandState):
    """ Reads the 1080p GStreamer UDP feed from Gazebo and displays it """
    threading.Thread(target=enable_gazebo_camera, daemon=True).start()
    print("[Camera] Waiting for Gazebo video stream on port 5600...")
    gazebo_stream = (
        'udpsrc port=5600 ! application/x-rtp, encoding-name=H264, payload=96 ! '
        'rtph264depay ! avdec_h264 ! videoconvert ! appsink'
    )
    # Important: Tell OpenCV to use the GStreamer backend
    cap = cv2.VideoCapture(gazebo_stream, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[Camera] Warning: Could not open video stream. Check Gazebo plugins.")
        return

    print("[Camera] Stream active.")
    while cmd.running:
        ret, frame = cap.read()
        if not ret: continue

        cv2.imshow("Talon UAV FPV Camera", frame)

        # Press 'q' inside the video window to cleanly stop everything
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[Camera] 'q' pressed. Shutting down...")
            cmd.stop()
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Camera] Stream closed.")


# ?????????????????????????????????????????????
#  CONTROLLER SETUP & MAIN LOOP
# ?????????????????????????????????????????????
def reset_named_controllers(ctrls: dict, *names: str):
    for name in names:
        ctrl = ctrls.get(name)
        if ctrl is not None:
            ctrl.reset()


def make_controllers(mode: str) -> dict:
    return {
        'pitch_att': HybridController(
            pid_ctrl=PIDController(kp=0.45, ki=0.10, kd=0.08, integral_limit=20.0, output_limit=15.0,
                                   integral_zone=18.0, rate_filter_tau=0.10),
            fuzzy_ctrl=FuzzyGainScheduler(error_range=25.0, rate_range=40.0),
        ),
        'roll_att': HybridController(
            pid_ctrl=PIDController(kp=0.50, ki=0.10, kd=0.08, integral_limit=25.0, output_limit=18.0,
                                   integral_zone=20.0, rate_filter_tau=0.10),
            fuzzy_ctrl=FuzzyGainScheduler(error_range=35.0, rate_range=50.0),
        ),
        'heading': HybridController(
            pid_ctrl=PIDController(kp=0.40, ki=0.035, kd=0.05, integral_limit=80.0, output_limit=35.0,
                                   integral_zone=90.0, rate_filter_tau=0.12),
            fuzzy_ctrl=FuzzyGainScheduler(error_range=120.0, rate_range=40.0),
        ),
        'altitude': HybridController(
            pid_ctrl=PIDController(kp=0.65, ki=0.08, kd=0.04, integral_limit=60.0, output_limit=18.0,
                                   integral_zone=35.0, rate_filter_tau=0.18),
            fuzzy_ctrl=FuzzyGainScheduler(error_range=40.0, rate_range=8.0),
        ),
        'speed': HybridController(
            pid_ctrl=PIDController(kp=0.07, ki=0.03, kd=0.01, integral_limit=10.0, output_limit=0.35,
                                   integral_zone=12.0, rate_filter_tau=0.20),
            fuzzy_ctrl=FuzzyGainScheduler(error_range=12.0, rate_range=6.0),
        ),
    }


def run():
    mode = "Hybrid FPV Mode"
    ctrls = make_controllers(mode)

    wait_for_takeoff()

    att_msg = connection.recv_match(type='ATTITUDE', blocking=True, timeout=2.0)
    cruise_roll_deg = math.degrees(att_msg.roll) if att_msg else 0.0
    cruise_pitch_deg = math.degrees(att_msg.pitch) if att_msg else 0.0
    cruise_yaw_deg = math.degrees(att_msg.yaw) if att_msg else 0.0

    connection.set_mode('FBWA')
    time.sleep(0.5)

    cmd = CommandState()
    cmd.target_yaw = cruise_yaw_deg

    # Start Threads
    threading.Thread(target=input_thread, args=(cmd, mode), daemon=True).start()
    threading.Thread(target=camera_thread, args=(cmd,), daemon=True).start()

    prev_meas = {'alt': None, 'speed': None, 'alt_rate_smoothed': 0.0, 'spd_rate_smoothed': 0.0}
    prev_time = time.time()
    trim_thrust = 0.60
    current_thrust = 0.8

    current_roll = cruise_roll_deg
    current_pitch = cruise_pitch_deg
    current_yaw = cruise_yaw_deg
    current_roll_rate = 0.0
    current_pitch_rate = 0.0
    current_yaw_rate = 0.0
    current_alt = TAKEOFF_ALT_TARGET
    current_spd = 15.0
    prev_override = False
    prev_alt_hold = False
    prev_heading_hold = True
    prev_speed_hold = False

    while True:
        t_pitch, t_roll, t_yaw, t_alt, t_speed, running, override = cmd.snapshot()
        if not running: break

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

        now = time.time()
        dt = max(now - prev_time, 0.01)
        prev_time = now

        if prev_meas['alt'] is None:
            prev_meas['alt'] = current_alt
        alt_delta = current_alt - prev_meas['alt']
        a_rate = (0.25 * (-alt_delta / dt)) + (0.75 * prev_meas['alt_rate_smoothed'])
        a_rate = clamp(a_rate, -12.0, 12.0)
        prev_meas['alt_rate_smoothed'] = a_rate
        prev_meas['alt'] = current_alt

        if prev_meas['speed'] is None:
            prev_meas['speed'] = current_spd
        spd_delta = current_spd - prev_meas['speed']
        err_rate = (0.25 * (-spd_delta / dt)) + (0.75 * prev_meas['spd_rate_smoothed'])
        err_rate = clamp(err_rate, -8.0, 8.0)
        prev_meas['spd_rate_smoothed'] = err_rate
        prev_meas['speed'] = current_spd

        if not override and current_alt < 15.0:
            cmd.set_override(True)
            override = True

        alt_hold_active = t_alt is not None and not override
        heading_hold_active = t_yaw is not None and not override
        speed_hold_active = t_speed is not None and not override

        if override != prev_override:
            reset_named_controllers(ctrls, 'pitch_att', 'roll_att', 'heading', 'altitude', 'speed')
            prev_override = override

        if alt_hold_active != prev_alt_hold:
            reset_named_controllers(ctrls, 'altitude', 'pitch_att')
            prev_alt_hold = alt_hold_active

        if heading_hold_active != prev_heading_hold:
            reset_named_controllers(ctrls, 'heading', 'roll_att')
            prev_heading_hold = heading_hold_active

        if speed_hold_active != prev_speed_hold:
            reset_named_controllers(ctrls, 'speed')
            prev_speed_hold = speed_hold_active

        if override:
            desired_pitch = clamp(ctrls['altitude'].compute(TAKEOFF_ALT_TARGET - current_alt, a_rate, dt), 0.0, 18.0)
            desired_roll = 0.0
            desired_thrust = 1.0
            if current_alt >= 49.0:
                cmd.update('pitch', 0.0)
                cmd.update('roll', 0.0)
                cmd.update('yaw', None)
                cmd.update('alt', None)
                cmd.set_override(False)
                override = False
                reset_named_controllers(ctrls, 'pitch_att', 'roll_att', 'heading', 'altitude')
        else:
            if t_alt is not None:
                desired_pitch = clamp(ctrls['altitude'].compute(t_alt - current_alt, a_rate, dt), -20.0, 20.0)
            elif t_pitch is not None:
                desired_pitch = clamp(t_pitch, -25.0, 25.0)
            else:
                desired_pitch = 0.0

            if t_yaw is not None:
                heading_error = wrap_angle_deg(t_yaw - current_yaw)
                desired_roll = clamp(ctrls['heading'].compute(heading_error, -current_yaw_rate, dt), -35.0, 35.0)
            elif t_roll is not None:
                desired_roll = clamp(t_roll, -45.0, 45.0)
            else:
                desired_roll = 0.0

            if t_speed is not None:
                desired_thrust = clamp(trim_thrust + ctrls['speed'].compute(t_speed - current_spd, err_rate, dt), 0.2, 1.0)
            else:
                desired_thrust = trim_thrust

        pitch_error = desired_pitch - current_pitch
        roll_error = desired_roll - current_roll
        pitch_correction = ctrls['pitch_att'].compute(pitch_error, -current_pitch_rate, dt)
        roll_correction = ctrls['roll_att'].compute(roll_error, -current_roll_rate, dt)

        target_pitch = clamp(desired_pitch + pitch_correction, -30.0, 30.0)
        target_roll = clamp(desired_roll + roll_correction, -45.0, 45.0)

        thrust_step_limit = 0.8 * dt
        current_thrust += clamp(desired_thrust - current_thrust, -thrust_step_limit, thrust_step_limit)
        current_thrust = clamp(current_thrust, 0.2, 1.0)

        # --- Send RC Overrides to Fixed-Wing ---
        connection.mav.rc_channels_override_send(
            connection.target_system, connection.target_component,
            angle_to_pwm(target_roll, 45.0),  # Roll (Ailerons)
            angle_to_pwm(target_pitch, 30.0),  # Pitch (Elevator)
            throttle_to_pwm(current_thrust),  # Throttle
            1500,  # Yaw (Rudder - centered)
            0, 0, 0, 0
        )
        time.sleep(0.05)  # ~20 Hz

    release_rc_overrides()
    print("\nControl loop stopped. RC channels released.")


if __name__ == '__main__':
    # Bypassing init_missions() for testing, but you can uncomment if you want to upload waypoints first
    auto_and_arm()
    run()
