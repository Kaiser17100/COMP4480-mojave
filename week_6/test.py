from pymavlink import mavutil
import cv2
import time
import math
import threading
import os

# PID
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
        out = (self.kp * error) + (self.ki * self._integral) + (self.kd * external_rate)
        
        self._prev_error = error
        self._prev_time  = now
        return out


# FUZZY
class FuzzyController:
    RESOLUTION = 200
    def __init__(self, error_range=180.0, rate_range=90.0, out_range=30.0):
        self.er  = error_range
        self.rr  = rate_range
        self.out = out_range
        
        self._out_universe = [
            i * (2 * out_range / (self.RESOLUTION - 1)) - out_range
            for i in range(self.RESOLUTION)
        ]
        
        self._rules = [
            ['NL', 'NL', 'NL', 'NL', 'NM', 'NS', 'ZE'],
            ['NL', 'NL', 'NL', 'NM', 'NS', 'ZE', 'PS'],
            ['NL', 'NL', 'NM', 'NS', 'ZE', 'PS', 'PM'],
            ['NL', 'NM', 'NS', 'ZE', 'PS', 'PM', 'PL'],
            ['NM', 'NS', 'ZE', 'PS', 'PM', 'PL', 'PL'],
            ['NS', 'ZE', 'PS', 'PM', 'PL', 'PL', 'PL'],
            ['ZE', 'PS', 'PM', 'PL', 'PL', 'PL', 'PL'],
        ]
        
        self._labels = ['NL', 'NM', 'NS', 'ZE', 'PS', 'PM', 'PL']

    @staticmethod
    def _tri(x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        return (x - a) / (b - a) if x <= b else (c - x) / (c - b)

    def _fuzzify(self, value, universe_half):
        u = universe_half
        centres = [-u, -2*u/3, -u/3, 0, u/3, 2*u/3, u]
        step = u / 3
        
        memberships = {
            label: self._tri(value, c - step, c, c + step)
            for label, c in zip(self._labels, centres)
        }

        memberships['NL'] = max(memberships['NL'], 1.0 if value <= -u else 0.0)
        memberships['PL'] = max(memberships['PL'], 1.0 if value >=  u else 0.0)

        return memberships

    def _defuzzify(self, activation):
        u    = self.out
        step = u / 3
        centres = [-u, -2*u/3, -u/3, 0, u/3, 2*u/3, u]
        num = den = 0.0
        
        for x in self._out_universe:
            mu = 0.0
            for label, c in zip(self._labels, centres):
                mu = max(mu, min(activation[label], self._tri(x, c - step, c, c + step)))
            num += x * mu
            den += mu
        
        if den == 0:
            return 0.0
        return num / den

    def _infer(self, err_mu, rate_mu):
        out = {label: 0.0 for label in self._labels}
        
        for i, e in enumerate(self._labels):
            for j, r in enumerate(self._labels):
                strength  = min(err_mu[e], rate_mu[r])
                out_label = self._rules[i][j]
                out[out_label] = max(out[out_label], strength)
        
        return out

    def compute(self, error: float, error_rate: float) -> float:
        error      = max(-self.er, min(self.er, error))
        error_rate = max(-self.rr, min(self.rr, error_rate))
        
        return self._defuzzify(
            self._infer(
                self._fuzzify(error,      self.er),
                self._fuzzify(error_rate, self.rr)
            )
        )
## GLOBAL VARIABLES ##

TAKEOFF_ALT_TARGET = 50.0   
TAKEOFF_ALT_THRESH =  5.0 

bounds = {
    'pitch': (-30.0,  30.0),
    'roll':  (-45.0,  45.0),
    'yaw':   (-180.0, 180.0),
    'alt': ( 55.0, 300.0),
    'speed': (13, 30)
}

## CODE START ##
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

## SHARED COMMAND STATE ##

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
        
        # Tracking Variables
        self.tracking_active = False
        self.tracker = None
        self.bbox = None

    def update(self, axis: str, value):
        with self._lock:
            setattr(self, f'target_{axis}', value)

    def set_override(self, state: bool):
        with self._lock:
            self.override = state

    def start_tracking(self, frame, x, y):
        with self._lock:
            # Create a 60x60 bounding box around the click
            size = 60
            self.bbox = (int(x - size/2), int(y - size/2), size, size)
            
            # Initialize a CSRT Tracker
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(frame, self.bbox)
            self.tracking_active = True
            
    def stop_tracking(self):
        with self._lock:
            self.tracking_active = False
            self.tracker = None

    def snapshot(self):
        with self._lock:
            return (self.target_pitch, self.target_roll, self.target_yaw, self.target_alt, self.target_speed, self.running,  self.override)

    def stop(self):
        with self._lock:
            self.running = False

## HELPERS ##

def angle_to_pwm(val: float) -> int:
    return int(max(1000, min(2000, 1500 + val)))

def enable_gazebo_camera():
    print("[Camera] Sending enable signal to Gazebo...")
    topic = '/world/runway/model/uav_1/link/base_link/sensor/nose_camera/image/enable_streaming'
    os.system(f'gz topic -t {topic} -m gz.msgs.Boolean -p "data: 1"')
    time.sleep(1)

class CameraStream:
    def __init__(self):
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.pipeline = "udpsrc port=5600 ! application/x-rtp, payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink drop=true sync=false"
        
    def start(self):
        enable_gazebo_camera()
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print("[Camera] WARNING: Failed to open GStreamer pipeline in OpenCV.")
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
            else:
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def stop(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()

def init_missions():
    print("Adding missions...")
    
    connection.mav.mission_clear_all_send(
        connection.target_system,
        connection.target_component
    )
    
    time.sleep(1)

    mission_list = read_missions()

    connection.mav.mission_count_send(
        connection.target_system,
        connection.target_component,
        len(mission_list)
    )
    
    for m in mission_list:
        connection.mav.mission_item_send(
            connection.target_system, connection.target_component,
            m['seq'], m['frame'], m['command'], m['current'], m['autocontinue'],
            m['param1'], m['param2'], m['param3'], m['param4'],
            m['lat'], m['lon'], m['alt']
        )
    
    while True:
        ack = connection.recv_match(type='MISSION_ACK', blocking=True)
        if ack:
            print("Mission accepted." if ack.type == 0 else "Mission failed.")
        break

def read_missions():
    missions = []
    if not os.path.exists("test.waypoints"):
        return missions
    with open("test.waypoints", 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('QGC'):
                continue
            parts = line.split('\t')
            if len(parts) < 12:
                continue
            missions.append({
                'seq':          int(parts[0]),
                'current':      int(parts[1]),
                'frame':        int(parts[2]),
                'command':      int(parts[3]),
                'param1':       float(parts[4]),
                'param2':       float(parts[5]),
                'param3':       float(parts[6]),
                'param4':       float(parts[7]),
                'lat':          float(parts[8]),
                'lon':          float(parts[9]),
                'alt':          float(parts[10]),
                'autocontinue': int(parts[11]),
            })
    return missions

def auto_and_arm():
    connection.set_mode('AUTO')
    print("Setting Mode to Auto...")
    connection.mav.command_long_send(
        connection.target_system, 
        connection.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 
        0, 1, 0, 0, 0, 0, 0, 0
    )
    print("Arming...")

def input_thread(cmd: CommandState, ctrl_label: str):
    print(f"\n[Input] Controller : {ctrl_label}")
    print("[Input] Commands   : pitch <deg>  roll <deg>  yaw <deg>  alt <val>  speed <val>  reset  quit")
    print("[Input] Note       : Typing 'roll' disables auto-heading. Typing 'yaw' disables manual roll.\n")

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
            cmd.stop_tracking()
            cmd.update('pitch', 0.0)
            cmd.update('roll', 0.0)
            cmd.update('yaw', None)
            cmd.update('alt', None)
            print("[Input] Reset → Pitch=0° Roll=0° (Wings level, tracking stopped, heading free)")
            continue

        parts = raw.split()
        if len(parts) != 2:
            print("[Input] Usage: <axis> <value>  or  reset  or  quit")
            continue

        axis, val_str = parts
        if axis not in bounds:
            print(f"[Input] Unknown axis '{axis}'. Choose from: {list(bounds)}")
            continue

        try:
            value = float(val_str)
        except ValueError:
            continue

        lo, hi = bounds[axis]
        if not (lo <= value <= hi):
            print(f"[Input] {axis} must be in [{lo}, {hi}].")
            continue

        # Stop tracking if manual roll/pitch/yaw/alt is entered
        cmd.stop_tracking()

        if axis == 'roll':
            cmd.update('roll', value)
            cmd.update('yaw', None)  
        elif axis == 'yaw':
            cmd.update('yaw', value)
            cmd.update('roll', None)
        elif axis == 'alt':
            cmd.update('pitch', None)
            cmd.update('alt', value)
        elif axis == 'pitch':
            cmd.update('alt', None)
            cmd.update('pitch', value)
        else:
            cmd.update(axis, value)

        print(f"[Input] Target {axis} → {value:+.1f}°")

def wait_for_takeoff():
    print(f"[Takeoff] Waiting for plane to climb ≥ {TAKEOFF_ALT_TARGET - TAKEOFF_ALT_THRESH:.0f} m ...")
    last_print = 0.0
    while True:
        msg = connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1.0)
        
        if msg is None: 
            continue
        
        alt_m = msg.relative_alt / 1000.0
        now   = time.time()
        if now - last_print >= 2.0:
            print(f"[Takeoff]   alt = {alt_m:.1f} m  (target {TAKEOFF_ALT_TARGET:.0f} m)")
            last_print = now
        
        if alt_m >= TAKEOFF_ALT_TARGET - TAKEOFF_ALT_THRESH:
            print(f"[Takeoff] Target altitude reached. Switching to GUIDED ...")
            break

def make_controllers() -> dict:
    return {
        # MUST FINE TUNE THESE
        'pitch': PIDController(kp=23.0, ki=0.55, kd=0.3),
        'roll': PIDController(kp=20.0, ki=0.5, kd=0.3),
        'yaw': PIDController(kp=2.0, ki=0.01, kd=0.1),
        'alt':   FuzzyController(error_range=50.0,  rate_range=10.0, out_range=25.0),
        'speed': FuzzyController(error_range=20.0,  rate_range=40.0,  out_range=1.0),
    }

## MAIN STUFF ##

def run():
    ctrls = make_controllers()

    wait_for_takeoff()

    print("Initializing Camera Stream on UDP 5600...")
    cam = CameraStream()
    cam.start()

    att_msg = connection.recv_match(type='ATTITUDE', blocking=True, timeout=2.0)
    cruise_yaw_deg = math.degrees(att_msg.yaw) if att_msg else 0.0
    
    connection.set_mode('FBWA')
    time.sleep(0.5)

    cmd = CommandState()
    cmd.target_yaw = cruise_yaw_deg

    t_input = threading.Thread(target=input_thread, args=(cmd, "Flying"), daemon=True)
    t_input.start()

    # OpenCV Window and Mouse Callback
    cv2.namedWindow("Talon Nose Camera")
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = cam.get_frame()
            if frame is not None:
                print(f"\n[Camera] Target acquired at ({x}, {y}). Engaging visual tracking!\ncmd> ", end="", flush=True)
                cmd.start_tracking(frame, x, y)
                # Clear previous manual targets
                cmd.update('yaw', None)
                cmd.update('alt', None)

    cv2.setMouseCallback("Talon Nose Camera", mouse_callback)

    prev_meas = {
        'yaw': None, 'pitch': None, 'roll': None, 
        'alt': None, 'alt_rate_smoothed':None, 
        'speed': None, 'spd_rate_smooth': None, 
        'time': time.time()
    }
    
    roll_pwm = pitch_pwm = yaw_pwm = 1500
    throttle_pwm = 1800
    current_thrust = 0.8
    
    print(f"\n| Fixed-Wing GUIDED cruise loop active at 20 Hz.")
    print(f"| Altitude is controlled by pitch (Hard deck: 15m)")
    print(f"| Telemetry printing is disabled to allow console input.\n")

    while True:
        t_pitch, t_roll, t_yaw, t_alt, t_speed, running, override = cmd.snapshot()

        frame = cam.get_frame()

        # --- VISUAL TRACKING LOGIC ---
        if cmd.tracking_active and not override and frame is not None:
            # Update the tracker
            success, bbox = cmd.tracker.update(frame)
            if success:
                h, w = frame.shape[:2]
                cx, cy = w / 2.0, h / 2.0
                
                # Center of the tracked object
                obj_x = bbox[0] + (bbox[2] / 2.0)
                obj_y = bbox[1] + (bbox[3] / 2.0)
                
                # Normalize error (-1.0 to 1.0)
                x_err = (obj_x - cx) / cx
                y_err = (cy - obj_y) / cy  # Positive is up
                
                # Calculate dynamic roll and pitch to center the target
                target_roll = x_err * 35.0
                target_pitch = y_err * 25.0
                
                # Clamp the values
                target_roll = max(bounds['roll'][0], min(bounds['roll'][1], target_roll))
                target_pitch = max(bounds['pitch'][0], min(bounds['pitch'][1], target_pitch))
                
                cmd.update('roll', target_roll)
                cmd.update('pitch', target_pitch)
                
                # Draw visual feedback
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                cv2.circle(frame, (int(obj_x), int(obj_y)), 4, (0, 0, 255), -1)
                cv2.line(frame, (int(cx), int(cy)), (int(obj_x), int(obj_y)), (255, 0, 0), 2)
                
            else:
                # Target lost (flew past or occluded)
                print("\n[Camera] Target lost. Leveling out.\ncmd> ", end="", flush=True)
                cmd.stop_tracking()
                cmd.update('roll', 0.0)
                cmd.update('pitch', 0.0)
        # -----------------------------

        if frame is not None:
            cv2.imshow("Talon Nose Camera", frame)
            cv2.waitKey(1)

        now = time.time()
        dt  = max(now - prev_meas['time'], 0.01)
        prev_meas['time'] = now

        if not running: 
            break

        msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD'], blocking=True, timeout=0.5)
        
        if msg is None: 
            continue

        if msg.get_type() == 'ATTITUDE':
            current_pitch = math.degrees(msg.pitch)
            current_roll = math.degrees(msg.roll)
            current_yaw = math.degrees(msg.yaw)

            # pitch calc
            if t_pitch is not None:
                if prev_meas['pitch'] is None:
                    prev_meas['pitch'] = current_pitch
                
                p_delta = current_pitch - prev_meas['pitch']
                p_rate  = max(-60.0, min(60.0, p_delta / dt))
                prev_meas['pitch'] = current_pitch
                p_err = t_pitch - current_pitch
                pitch_pwm = angle_to_pwm(current_pitch + ctrls['pitch'].compute(p_err, p_rate))

            # roll calc
            if t_roll is not None:
                if prev_meas['roll'] is None:
                    prev_meas['roll'] = current_roll
                
                r_delta = current_roll - prev_meas['roll']
                r_rate  = max(-60.0, min(60.0, r_delta / dt))
                prev_meas['roll'] = current_roll
                r_err = t_roll - current_roll

                roll_pwm = angle_to_pwm(ctrls['roll'].compute(r_err, r_rate))

            # yaw calc
            if t_yaw is not None:
                if prev_meas['yaw'] is None:
                    prev_meas['yaw'] = current_yaw
                
                yaw_delta = (current_yaw - prev_meas['yaw'] + 180) % 360 - 180
                prev_meas['yaw'] = current_yaw
                y_rate = max(-60.0, min(60.0, -yaw_delta / dt))
                y_err = (t_yaw - current_yaw + 180) % 360 - 180

                yaw_pwm = angle_to_pwm(ctrls['yaw'].compute(y_err, y_rate))

        elif msg.get_type() == 'VFR_HUD':
            airspeed = msg.airspeed

            if t_speed is not None:
                if prev_meas['speed'] is None:
                    prev_meas['speed'] = airspeed
                    prev_meas['spd_rate_smoothed'] = 0.0

                spd_delta = airspeed - prev_meas['speed']
                prev_meas['speed'] = airspeed
                
                spd_err = t_speed - airspeed
                raw_rate = -spd_delta / dt
                
                smoothed_rate = (0.2 * raw_rate) + (0.8 * prev_meas.get('spd_rate_smoothed', 0.0))
                prev_meas['spd_rate_smoothed'] = smoothed_rate
                err_rate = max(-10.0, min(10.0, smoothed_rate))

                thrust_shift = ctrls['speed'].compute(spd_err, err_rate)
                current_thrust += thrust_shift * dt
                current_thrust = max(0.2, min(1.0, current_thrust))

            else:
                prev_meas['speed'] = None
                if 'spd_rate_smoothed' in prev_meas:
                    del prev_meas['spd_rate_smoothed']

        elif msg.get_type() == 'GLOBAL_POSITION_INT':
            alt_m = msg.relative_alt / 1000.0        
            
            if prev_meas['alt'] is None:
                prev_meas['alt'] = alt_m
                prev_meas['alt_rate_smoothed'] = 0.0
                
            alt_delta = alt_m - prev_meas['alt']
            prev_meas['alt'] = alt_m
            
            raw_a_rate = -alt_delta / dt
            
            smoothed_a_rate = (0.2 * raw_a_rate) + (0.8 * prev_meas.get('alt_rate_smoothed', 0.0))
            prev_meas['alt_rate_smoothed'] = smoothed_a_rate
            a_rate = smoothed_a_rate
            
            if not override and alt_m < 15.0:
                print(f"\n[EMERGENCY] Altitude dropped to {alt_m:.1f}m!")
                print("[EMERGENCY] Hard deck breached. Max throttle and pulling up to 50m.\ncmd> ", end="", flush=True)
                cmd.set_override(True)
                cmd.stop_tracking() # Cancel tracking on emergency
                
            if override:
                a_err  = 50.0 - alt_m
                if abs(a_err) <= 1.0:
                    a_err = 0.0
                
                override_pitch = max(0.0, min(20.0, ctrls['alt'].compute(a_err, a_rate)))
                cmd.update('pitch', override_pitch)
                current_thrust = 1.0 

                if alt_m >= 49.0:
                    print(f"\n[EMERGENCY] Reached {alt_m:.1f}m. Leveling out and restoring free flight.\ncmd> ", end="", flush=True)
                    cmd.update('pitch', 0.0)
                    cmd.update('roll', 0.0) 
                    cmd.update('yaw', None)
                    cmd.set_override(False)
            
            else:
                if t_alt is not None:
                    a_err  = t_alt - alt_m
                    if abs(a_err) <= 1.0:
                        a_err = 0.0

                    alt_pitch = ctrls['alt'].compute(a_err, a_rate)
                    alt_pitch = max(-25.0, min(25.0, alt_pitch))
                    cmd.update('pitch', alt_pitch)
                
                else:
                    if t_speed is None:
                        current_thrust = 0.6

        connection.mav.rc_channels_override_send(
            connection.target_system,
            connection.target_component,
            roll_pwm if t_roll is not None else 65535,   # Roll
            pitch_pwm if t_pitch is not None else 65535, # Pitch
            throttle_pwm,                                # Throttle
            yaw_pwm if t_yaw is not None else 65535,     # Yaw
            65535, 65535, 65535, 65535                   # rest is blank
        )

        time.sleep(0.02)   # 20 Hz

    cam.stop()
    cv2.destroyAllWindows()
    print("\nControl loop stopped.")

if __name__ == "__main__":
    init_missions()
    auto_and_arm()
    run()
