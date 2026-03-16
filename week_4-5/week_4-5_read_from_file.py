from pymavlink import mavutil
import time
import math
import threading

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

# PID
class PIDController:
    def __init__(self, kp=12.0, ki=0.5, kd=0.6):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.time()

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.time()

    # external_rate is for ease of use for controller logic, fuzzy controller needs it, easier to code
    def compute(self, error: float, external_rate: float) -> float:
        now = time.time()
        dt  = max(now - self._prev_time, 0.01)
        
        self._integral += error * dt
        # Kp * e + integral(e) + derivative(e)
        out = (self.kp * error) + (self.ki * self._integral) + (self.kd * dt)
        
        self._prev_error = error
        self._prev_time  = now
        return out


# FUZZY
class FuzzyController:
    RESOLUTION = 200
    # error_range = sensivity to target, rate_range = sensivitiy to change
    def __init__(self, error_range=180.0, rate_range=90.0, out_range=30.0):
        self.er  = error_range
        self.rr  = rate_range
        self.out = out_range
        
        # grid for defuzzify to use
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
        
        # Negative Large, Negative Medium, Negative Small, Zero, Positive Small, Positive Medium, Positive Large
        self._labels = ['NL', 'NM', 'NS', 'ZE', 'PS', 'PM', 'PL']

    # function to calculation where it sits on the label triangle
    @staticmethod
    def _tri(x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        return (x - a) / (b - a) if x <= b else (c - x) / (c - b)

    
    def _fuzzify(self, value, universe_half):
        u = universe_half
        centres = [-u, -2*u/3, -u/3, 0, u/3, 2*u/3, u]
        step = u / 3
        
        # calculate how much it's on(?) each label
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
        
        # find the return value from out table for our label
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
        with self._lock:
            setattr(self, f'target_{axis}', value)

    def set_override(self, state: bool):
        with self._lock:
            self.override = state

    def snapshot(self):
        with self._lock:
            return (self.target_pitch, self.target_roll, self.target_yaw, self.target_alt, self.target_speed, self.running,  self.override)

    def stop(self):
        with self._lock:
            self.running = False


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def init_missions():
    print("Adding missions...")
    
    #clearing prev missions if had any
    connection.mav.mission_clear_all_send(
        connection.target_system, #  plane id
        connection.target_component # plane's brain
    )
    
    time.sleep(1)

    mission_list = read_missions()

    connection.mav.mission_count_send(
        connection.target_system, # plane id
        connection.target_component, # plane's brain
        len(mission_list)
    )
    
    for m in mission_list:
        connection.mav.mission_item_send(
            connection.target_system, connection.target_component,
            m['seq'], # wp no 
            m['frame'], #
            m['command'], # command
            m['current'], # current mission flag
            m['autocontinue'], # autocontinue flag
            m['param1'], m['param2'], m['param3'], m['param4'], # param 1-4
            m['lat'], m['lon'], m['alt'] # param 5-8 coords
        )
    
    # waiting for ACK to check whether our mission is successfully send
    while True:
        ack = connection.recv_match(type='MISSION_ACK', blocking=True)
        if ack:
            print("Mission accepted." if ack.type == 0 else "Mission failed.")
        break

def read_missions():
    missions = []
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


# mavlink euler_to_quat does not work this is the replacement
def euler_to_quat(rpy: list) -> list:
    roll, pitch, yaw = rpy
    cr, cp, cy = math.cos(roll/2), math.cos(pitch/2), math.cos(yaw/2)
    sr, sp, sy = math.sin(roll/2), math.sin(pitch/2), math.sin(yaw/2)
    return [
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ]

# this allows cmd to input while displaying something, it works with threading heyyyyy işletim dersi
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
            cmd.update('pitch', 0.0)
            cmd.update('roll', 0.0)
            cmd.update('yaw', None)
            print("[Input] Reset → Pitch=0° Roll=0° (Wings level, heading free)")
            continue

        parts = raw.split()
        if len(parts) != 2:
            print("[Input] Usage: <axis> <value>  or  reset  or  quit")
            continue

        axis, val_str = parts
        if axis not in AXIS_BOUNDS:
            print(f"[Input] Unknown axis '{axis}'. Choose from: {list(AXIS_BOUNDS)}")
            continue

        try:
            value = float(val_str)
        except ValueError:
            continue

        lo, hi = AXIS_BOUNDS[axis]
        if not (lo <= value <= hi):
            print(f"[Input] {axis} must be in [{lo}, {hi}].")
            continue

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


def select_controller() -> str:
    while True:
        choice = input("Select Outer-Loop Controller \n1) PID \n2) Fuzzy Logic \nChoice: ").strip()
        
        if choice == '1': return 'pid'
        if choice == '2': return 'fuzzy'
        
        print("Please enter 1 or 2.")


def make_controllers(mode: str) -> dict:
    if mode == 'pid':
        return {
            # MUST FINE TUNE THESE
            'pitch': PIDController(kp=1.5, ki=0.02, kd=0.3),
            'roll': PIDController(kp=0.8, ki=0.01, kd=0.3),
            'yaw': PIDController(kp=2.0, ki=0.01, kd=0.1),
            'alt': PIDController(kp=0.4, ki=0.01, kd=0.1),
            'speed': PIDController(kp=0.05, ki=0.005, kd=0.01),
        }
    
    return {
        # MUST FINE TUNE THESE
        'pitch': FuzzyController(error_range=45.0, rate_range=60.0, out_range=12.0),
        'roll':  FuzzyController(error_range=45.0, rate_range=60.0, out_range=12.0),
        'yaw':   FuzzyController(error_range=180.0, rate_range=60.0, out_range=20.0),
        'alt':   FuzzyController(error_range=50.0,  rate_range=10.0, out_range=8.0),
        'speed': FuzzyController(error_range=20.0,  rate_range=5.0,  out_range=0.4),
    }


# ─────────────────────────────────────────────
#  Takeoff monitor
# ─────────────────────────────────────────────

def run():
    mode  = select_controller()
    ctrls = make_controllers(mode)

    wait_for_takeoff()

    # to make sure plane doesn't crash
    att_msg = connection.recv_match(type='ATTITUDE', blocking=True, timeout=2.0)
    cruise_yaw_deg = math.degrees(att_msg.yaw) if att_msg else 0.0
    
    connection.set_mode('GUIDED')
    time.sleep(0.5)

    cmd = CommandState()
    cmd.target_yaw = cruise_yaw_deg

    inp = threading.Thread(target=input_thread, args=(cmd, mode), daemon=True)
    inp.start()

    # to track measurements here instead of error states
    prev_meas = {'yaw': None, 'pitch': None, 'roll': None, 'alt': None, 'speed': None}
    
    prev_time_att = time.time()
    prev_time_alt = time.time()
    prev_time_spd = time.time()
    
    current_thrust = 0.6  

    print(f"\n[{mode}] Fixed-Wing GUIDED cruise loop active at 20 Hz.")
    print(f"[{mode}] Altitude is controlled by pitch (Hard deck: 15m)")
    print(f"[{mode}] Telemetry printing is disabled to allow console input.\n")

    while True:
        t_pitch, t_roll, t_yaw, t_alt, t_speed, running, override = cmd.snapshot()
        
        if not running: 
            break

        msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD'], blocking=True, timeout=0.5)
        
        if msg is None: 
            continue

        if msg.get_type() == 'ATTITUDE':
            now = time.time()
            dt  = max(now - prev_time_att, 0.01)
            prev_time_att = now
            
            current_pitch = math.degrees(msg.pitch)
            current_roll = math.degrees(msg.roll)
            current_yaw = math.degrees(msg.yaw)
            
            roll_deg = yaw_deg = pitch_deg = 0

            # pitch calc
            if t_pitch is not None:
                if prev_meas['pitch'] is None:
                    prev_meas['pitch'] = current_pitch
                
                p_delta = current_pitch - prev_meas['pitch']
                p_rate  = max(-60.0, min(60.0, p_delta / dt))
                prev_meas['pitch'] = current_pitch

                p_err = t_pitch - current_pitch

                # if we do axis_deg = ctrls[axis].compute() plane crashes
                pitch_deg = max(-30.0, min(30.0, t_pitch + ctrls['pitch'].compute(p_err, p_rate)))

            # roll calc
            if t_roll is not None:
                if prev_meas['roll'] is None:
                    prev_meas['roll'] = current_roll
                
                r_delta = current_roll - prev_meas['roll']
                r_rate  = max(-60.0, min(60.0, r_delta / dt))
                prev_meas['roll'] = current_roll

                r_err    = t_roll - current_roll

                roll_deg = max(-45.0, min(45.0, t_roll + ctrls['roll'].compute(r_err, r_rate)))

            # yaw calc
            if t_yaw is not None:
                if prev_meas['yaw'] is None:
                    prev_meas['yaw'] = current_yaw
                
                yaw_delta = (current_yaw - prev_meas['yaw'] + 180) % 360 - 180
                prev_meas['yaw'] = current_yaw
                
                y_rate = max(-60.0, min(60.0, -yaw_delta / dt))

                y_err   = (t_yaw - current_yaw + 180) % 360 - 180

                yaw_deg = current_yaw + ctrls['yaw'].compute(y_err, y_rate)

            pitch_rad = math.radians(pitch_deg)
            roll_rad = math.radians(roll_deg)
            yaw_rad = math.radians(yaw_deg)

            
            connection.mav.set_attitude_target_send(
                int(time.time() * 1000) & 0xFFFFFFFF,
                connection.target_system,
                connection.target_component,
                0b00000111,
                euler_to_quat([roll_rad, pitch_rad, yaw_rad]),
                0, 0, 0,
                current_thrust
            )

        elif msg.get_type() == 'VFR_HUD':
            now = time.time()
            dt  = max(now - prev_time_spd, 0.01)
            prev_time_spd = now

            airspeed = msg.airspeed

            # speec calc
            if t_speed is not None:
                if prev_meas['speed'] is None:
                    prev_meas['speed'] = airspeed

                spd_delta = airspeed - prev_meas['speed']
                spd_rate  = max(-5.0, min(5.0, spd_delta / dt))
                prev_meas['speed'] = airspeed

                spd_err        = t_speed - airspeed
                thrust_delta   = ctrls['speed'].compute(spd_err, spd_rate)

                current_thrust = max(0.2, min(1.0, 0.6 + thrust_delta))
            else:
                prev_meas['speed'] = None

        elif msg.get_type() == 'GLOBAL_POSITION_INT':
            now = time.time()
            dt  = max(now - prev_time_alt, 0.01)
            prev_time_alt = now

            alt_m = msg.relative_alt / 1000.0       
            
            # emergency override addition of hüseyin
            if not override and alt_m < 15.0:
                print(f"\n[EMERGENCY] Altitude dropped to {alt_m:.1f}m!")
                print("[EMERGENCY] Hard deck breached. Max throttle and pulling up to 50m.\ncmd> ", end="")
                cmd.set_override(True)
                
            if override:
                if prev_meas['alt'] is None:
                    prev_meas['alt'] = alt_m
                
                alt_delta = alt_m - prev_meas['alt']
                prev_meas['alt'] = alt_m
                
                a_rate = -alt_delta / dt
                a_err  = 50.0 - alt_m
                
                override_pitch = max(0.0, min(20.0, ctrls['alt'].compute(a_err, a_rate)))
                cmd.update('pitch', override_pitch)
                current_thrust = 1.0 

                if alt_m >= 49.0:
                    print(f"\n[EMERGENCY] Reached {alt_m:.1f}m. Leveling out and restoring free flight.\ncmd> ", end="")
                    cmd.update('pitch', 0.0)
                    cmd.update('roll', 0.0) 
                    cmd.update('yaw', None)
                    cmd.set_override(False)
            
            # normal part
            else:
                # alt calc with pitch
                if t_alt is not None:
                    if prev_meas['alt'] is None:
                        prev_meas['alt'] = alt_m

                    alt_delta = alt_m - prev_meas['alt']
                    prev_meas['alt'] = alt_m
                    a_rate = -alt_delta / dt
                    a_err  = t_alt - alt_m

                    alt_pitch = ctrls['alt'].compute(a_err, a_rate)
                    alt_pitch = max(-15.0, min(15.0, alt_pitch))
                    cmd.update('pitch', alt_pitch)

                else:
                    if t_speed is None:
                        current_thrust = 0.6
                    prev_meas['alt'] = None

        time.sleep(0.05)   # 20 Hz

    print("\nControl loop stopped.")

if __name__ == '__main__':
    init_missions()
    auto_and_arm()
    run()