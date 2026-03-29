from pymavlink import mavutil
import time
import math
import threading
import cv2  # Added for the camera stream

## GLOBAL VARIABLES ##
AXIS_BOUNDS = {
    'pitch': (-30.0,  30.0),
    'roll':  (-45.0,  45.0),
    'yaw':   (-180.0, 180.0),
    'alt': (-300.0, 300.0),
    'speed': (13, 30)
}

TAKEOFF_ALT_TARGET = 50.0   
TAKEOFF_ALT_THRESH =  5.0 

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

# ─────────────────────────────────────────────
#  CONTROLLERS
# ─────────────────────────────────────────────

# 1. UNCHANGED PID CONTROLLER
class PIDController:
    def __init__(self, kp=12.0, ki=0.5, kd=0.6):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.time()

    def compute(self, error: float, external_rate: float) -> float:
        now = time.time()
        dt  = max(now - self._prev_time, 0.01)
        
        self._integral += error * dt
        self._integral = max(-50.0, min(50.0, self._integral)) # Anti-windup
        
        out = (self.kp * error) + (self.ki * self._integral) + (self.kd * external_rate)
        
        self._prev_error = error
        self._prev_time  = now
        return out


# 2. IMPROVED FUZZY CONTROLLER (From last week's fix)
class FuzzyController:
    def __init__(self, error_range=180.0, rate_range=90.0, out_range=30.0):
        self.error_range = error_range
        self.rate_range = rate_range
        self.out_range = out_range
        self.labels = ['NL', 'NM', 'NS', 'ZE', 'PS', 'PM', 'PL']
        
        self.rules = [
            ['NL', 'NL', 'NL', 'NL', 'NM', 'NS', 'ZE'],
            ['NL', 'NL', 'NL', 'NM', 'NS', 'ZE', 'PS'],
            ['NL', 'NL', 'NM', 'NS', 'ZE', 'PS', 'PM'],
            ['NL', 'NM', 'NS', 'ZE', 'PS', 'PM', 'PL'],
            ['NM', 'NS', 'ZE', 'PS', 'PM', 'PL', 'PL'],
            ['NS', 'ZE', 'PS', 'PM', 'PL', 'PL', 'PL'],
            ['ZE', 'PS', 'PM', 'PL', 'PL', 'PL', 'PL'],
        ]

    def _triangle_membership(self, x, a, b, c):
        if x <= a or x >= c: return 0.0
        elif a < x <= b: return (x - a) / (b - a)
        elif b < x < c: return (c - x) / (c - b)
        return 0.0

    def _fuzzify(self, value, limit):
        value = max(-limit, min(limit, value))
        step = limit / 3.0
        centers = {'NL': -limit, 'NM': -2*step, 'NS': -step, 'ZE': 0.0, 'PS': step, 'PM': 2*step, 'PL': limit}
        memberships = {}
        for label, c in centers.items():
            if label == 'NL' and value <= c: memberships[label] = 1.0
            elif label == 'PL' and value >= c: memberships[label] = 1.0
            else: memberships[label] = self._triangle_membership(value, c - step, c, c + step)
        return memberships

    def _infer(self, err_fuzz, rate_fuzz):
        out_fuzz = {label: 0.0 for label in self.labels}
        for i, err_label in enumerate(self.labels):
            for j, rate_label in enumerate(self.labels):
                rule_strength = min(err_fuzz[err_label], rate_fuzz[rate_label])
                out_label = self.rules[i][j]
                out_fuzz[out_label] = max(out_fuzz[out_label], rule_strength)
        return out_fuzz

    def _defuzzify(self, out_fuzz):
        step = self.out_range / 3.0
        centers = {'NL': -self.out_range, 'NM': -2*step, 'NS': -step, 'ZE': 0.0, 'PS': step, 'PM': 2*step, 'PL': self.out_range}
        numerator, denominator = 0.0, 0.0
        for label, mu in out_fuzz.items():
            numerator += mu * centers[label]
            denominator += mu
        return 0.0 if denominator == 0.0 else numerator / denominator

    def compute(self, error: float, error_rate: float) -> float:
        err_fuzz = self._fuzzify(error, self.error_range)
        rate_fuzz = self._fuzzify(error_rate, self.rate_range)
        out_fuzz = self._infer(err_fuzz, rate_fuzz)
        return self._defuzzify(out_fuzz)


# 3. NEW HYBRID CONTROLLER
class HybridController:
    """
    Combines PID and Fuzzy logic outputs. 
    Adjust the weights to favor one controller over the other.
    """
    def __init__(self, pid_ctrl, fuzzy_ctrl, pid_weight=1.0, fuzzy_weight=1.0):
        self.pid = pid_ctrl
        self.fuzzy = fuzzy_ctrl
        self.pw = pid_weight
        self.fw = fuzzy_weight

    def compute(self, error: float, rate: float) -> float:
        pid_output = self.pid.compute(error, rate)
        fuzzy_output = self.fuzzy.compute(error, rate)
        # The Hybrid output is a blended sum of both strategies
        return (self.pw * pid_output) + (self.fw * fuzzy_output)

# ─────────────────────────────────────────────
#  SHARED COMMAND STATE
# ─────────────────────────────────────────────
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
            return (self.target_pitch, self.target_roll, self.target_yaw, self.target_alt, self.target_speed, self.running, self.override)

    def stop(self):
        with self._lock: self.running = False


# ─────────────────────────────────────────────
#  HELPERS FOR FBWA & RC OVERRIDE
# ─────────────────────────────────────────────
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
    print(f"[Takeoff] Waiting for plane to climb ≥ {TAKEOFF_ALT_TARGET - TAKEOFF_ALT_THRESH:.0f} m ...")
    last_print = 0.0
    while True:
        msg = connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1.0)
        if msg is None: continue
        
        alt_m = msg.relative_alt / 1000.0   
        now   = time.time()
        if now - last_print >= 2.0:
            print(f"[Takeoff]   alt = {alt_m:.1f} m  (target {TAKEOFF_ALT_TARGET:.0f} m)")
            last_print = now
        
        if alt_m >= TAKEOFF_ALT_TARGET - TAKEOFF_ALT_THRESH:
            print(f"[Takeoff] Target altitude reached. Switching to FBWA outer loops ...")
            break

# ─────────────────────────────────────────────
#  THREADS (INPUT & CAMERA)
# ─────────────────────────────────────────────
def input_thread(cmd: CommandState, ctrl_label: str):
    print(f"\n[Input] Controller : {ctrl_label}")
    print("[Input] Commands   : pitch <deg>  roll <deg>  yaw <deg>  alt <val>  speed <val>  reset  quit")
    while cmd.running:
        try: raw = input("cmd> ").strip().lower()
        except EOFError: break

        if not raw: continue
        if raw in ('quit', 'exit', 'q'):
            print("[Input] Stopping...")
            cmd.stop()
            break

        if cmd.snapshot()[6]: 
            print("[Input] COMMAND IGNORED: Plane is currently in emergency pull-up!")
            continue

        if raw == 'reset':
            cmd.update('pitch', 0.0); cmd.update('roll', 0.0); cmd.update('yaw', None); cmd.update('alt', None)
            continue

        parts = raw.split()
        if len(parts) != 2: continue

        axis, val_str = parts
        if axis not in AXIS_BOUNDS: continue
        try: value = float(val_str)
        except ValueError: continue

        if axis == 'roll': cmd.update('roll', value); cmd.update('yaw', None)  
        elif axis == 'yaw': cmd.update('yaw', value); cmd.update('roll', None)
        elif axis == 'alt': cmd.update('pitch', None); cmd.update('alt', value)
        elif axis == 'pitch': cmd.update('alt', None); cmd.update('pitch', value)
        else: cmd.update(axis, value)

def camera_thread(cmd: CommandState):
    """ Reads the 1080p GStreamer UDP feed from Gazebo and displays it """
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

# ─────────────────────────────────────────────
#  CONTROLLER SETUP & MAIN LOOP
# ─────────────────────────────────────────────
def make_controllers(mode: str) -> dict:
    # Wrap your original PID and improved Fuzzy controllers inside the new HybridController
    return {
        'yaw': HybridController(
            pid_ctrl=PIDController(kp=1.0, ki=0.01, kd=0.1),
            fuzzy_ctrl=FuzzyController(error_range=180.0, rate_range=90.0, out_range=20.0),
            pid_weight=0.5, fuzzy_weight=1.0
        ),
        'alt': HybridController(
            pid_ctrl=PIDController(kp=0.5, ki=0.01, kd=0.1),
            fuzzy_ctrl=FuzzyController(error_range=50.0, rate_range=10.0, out_range=15.0),
            pid_weight=0.5, fuzzy_weight=1.0
        ),
        'speed': HybridController(
            pid_ctrl=PIDController(kp=0.05, ki=0.0, kd=0.0),
            fuzzy_ctrl=FuzzyController(error_range=20.0, rate_range=40.0, out_range=0.5),
            pid_weight=0.5, fuzzy_weight=1.0
        ),
    }

def run():
    mode = "Hybrid FPV Mode"
    ctrls = make_controllers(mode)

    wait_for_takeoff()

    att_msg = connection.recv_match(type='ATTITUDE', blocking=True, timeout=2.0)
    cruise_yaw_deg = math.degrees(att_msg.yaw) if att_msg else 0.0
    
    connection.set_mode('FBWA')
    time.sleep(0.5)

    cmd = CommandState()
    cmd.target_yaw = cruise_yaw_deg

    # Start Threads
    threading.Thread(target=input_thread, args=(cmd, mode), daemon=True).start()
    threading.Thread(target=camera_thread, args=(cmd,), daemon=True).start()

    prev_meas = {'yaw': None, 'alt': None, 'speed': None, 'alt_rate_smoothed': 0.0, 'spd_rate_smoothed': 0.0}
    prev_time = time.time()
    current_thrust = 0.8

    current_yaw, current_alt, current_spd = cruise_yaw_deg, TAKEOFF_ALT_TARGET, 15.0

    while True:
        t_pitch, t_roll, t_yaw, t_alt, t_speed, running, override = cmd.snapshot()
        if not running: break

        msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD'], blocking=False)
        while msg is not None:
            msg_type = msg.get_type()
            if msg_type == 'ATTITUDE': current_yaw = math.degrees(msg.yaw)
            elif msg_type == 'GLOBAL_POSITION_INT': current_alt = msg.relative_alt / 1000.0
            elif msg_type == 'VFR_HUD': current_spd = msg.airspeed
            msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD'], blocking=False)

        now = time.time()
        dt  = max(now - prev_time, 0.01)
        prev_time = now

        # --- Calculate Rates ---
        if prev_meas['yaw'] is None: prev_meas['yaw'] = current_yaw
        yaw_delta = (current_yaw - prev_meas['yaw'] + 180) % 360 - 180
        y_rate = max(-60.0, min(60.0, -yaw_delta / dt))
        prev_meas['yaw'] = current_yaw

        if prev_meas['alt'] is None: prev_meas['alt'] = current_alt
        alt_delta = current_alt - prev_meas['alt']
        a_rate = (0.2 * (-alt_delta / dt)) + (0.8 * prev_meas['alt_rate_smoothed'])
        prev_meas['alt_rate_smoothed'], prev_meas['alt'] = a_rate, current_alt

        if prev_meas['speed'] is None: prev_meas['speed'] = current_spd
        spd_delta = current_spd - prev_meas['speed']
        err_rate = (0.2 * (-spd_delta / dt)) + (0.8 * prev_meas['spd_rate_smoothed'])
        prev_meas['spd_rate_smoothed'], prev_meas['speed'] = err_rate, current_spd

        # --- Outer Loop Processing (Using Hybrid Controllers) ---
        
        # 1. Altitude -> Target Pitch
        if not override and current_alt < 15.0:
            cmd.set_override(True); override = True
            
        if override:
            a_err = 50.0 - current_alt
            target_pitch = max(0.0, min(20.0, ctrls['alt'].compute(a_err, a_rate)))
            current_thrust = 1.0 
            if current_alt >= 49.0:
                cmd.update('pitch', 0.0); cmd.update('roll', 0.0); cmd.update('yaw', None)
                cmd.set_override(False); override = False
        else:
            if t_alt is not None:
                target_pitch = max(-25.0, min(25.0, ctrls['alt'].compute(t_alt - current_alt, a_rate)))
            elif t_pitch is not None: target_pitch = t_pitch
            else: target_pitch = 0.0

            # 2. Speed -> Target Throttle
            if t_speed is not None:
                current_thrust += (ctrls['speed'].compute(t_speed - current_spd, err_rate) * dt)
                current_thrust = max(0.2, min(1.0, current_thrust))
            else: current_thrust = 0.6

        # 3. Heading -> Target Roll
        if t_yaw is not None and not override:
            y_err = (t_yaw - current_yaw + 180) % 360 - 180
            target_roll = max(-45.0, min(45.0, ctrls['yaw'].compute(y_err, y_rate)))
        elif t_roll is not None: target_roll = t_roll
        else: target_roll = 0.0

        # --- Send RC Overrides to Fixed-Wing ---
        connection.mav.rc_channels_override_send(
            connection.target_system, connection.target_component,
            angle_to_pwm(target_roll, 45.0),   # Roll (Ailerons)
            angle_to_pwm(target_pitch, 30.0),  # Pitch (Elevator)
            throttle_to_pwm(current_thrust),   # Throttle
            1500,                              # Yaw (Rudder - centered)
            0, 0, 0, 0
        )
        time.sleep(0.05)   # ~20 Hz

    release_rc_overrides()
    print("\nControl loop stopped. RC channels released.")

if __name__ == '__main__':
    # Bypassing init_missions() for testing, but you can uncomment if you want to upload waypoints first
    auto_and_arm()
    run()
