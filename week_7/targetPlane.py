from pymavlink import mavutil
import time
import random
import math
import threading
from mathHelpers import *
from controllers import *

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
TIMER_MAX = 4.0
DX_CONST = 0.25
DT_MIN = 0.01
PREV_MEAS_RATE_CONST = 0.75

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

    def update(self, axis: str, value):
        with self._lock: setattr(self, f'target_{axis}', value)

    def set_override(self, state: bool):
        with self._lock: self.override = state

    def snapshot(self):
        with self._lock:
            return (self.target_pitch, self.target_roll, self.target_yaw, self.target_alt, self.target_speed,
                    self.running)

## PLANE START ##
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

## HELPERS FOR FBWA & RC OVERRIDE ##

# rng movement decider resets other axises
def randomMovement(cmd: CommandState):
    i = random.randint(0, 3)
    axis = list(AXIS_BOUNDS.keys())[i]
    lo, high = AXIS_BOUNDS[axis]
    val = random.uniform(lo, high)
    print(f"Random Manuever: {axis} {val}")
    cmd.update(axis, val)
    for x in list(AXIS_BOUNDS.keys()):
        if x != axis:
            cmd.update(x, None)

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


## CONTROLLER SETUP & MAIN LOOP

# these are product of husein
def make_controllers() -> dict:
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
    ctrls = make_controllers()

    wait_for_takeoff()

    att_msg = connection.recv_match(type='ATTITUDE', blocking=True, timeout=2.0)
    cruise_roll_deg = math.degrees(att_msg.roll) if att_msg else 0.0
    cruise_pitch_deg = math.degrees(att_msg.pitch) if att_msg else 0.0
    cruise_yaw_deg = math.degrees(att_msg.yaw) if att_msg else 0.0

    connection.set_mode('FBWA')
    time.sleep(0.5)

    cmd = CommandState()
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
    timer = 0.0

    override = True

    while True:
        t_pitch, t_roll, t_yaw, t_alt, t_speed, running = cmd.snapshot()
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
            msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD'], blocking=False)

        now = time.time()
        dt = max(now - prev_time, DT_MIN)
        prev_time = now

        timer += dt
        if timer >= TIMER_MAX and not override:
            timer = 0.0
            randomMovement(cmd)

        # alt rate calc
        if prev_meas['alt'] is None:
            prev_meas['alt'] = current_alt
        dA = current_alt - prev_meas['alt']
        a_rate = (DX_CONST * (-dA / dt)) + (PREV_MEAS_RATE_CONST * prev_meas['alt_rate_smoothed'])
        a_rate = clamp(a_rate, -12.0, 12.0)
        prev_meas['alt_rate_smoothed'] = a_rate
        prev_meas['alt'] = current_alt

        # pitch calc dependent on alt
        if t_alt is not None:
            low, high = AXIS_BOUNDS['pitch']
            desired_pitch = clamp(ctrls['altitude'].compute(t_alt - current_alt, a_rate, dt), low, high)

        # normal pitch calc
        elif t_pitch is not None:
            low, high = AXIS_BOUNDS['pitch']
            desired_pitch = clamp(t_pitch, low, high)
        else:
            desired_pitch = 0.0

        # roll calc dependent on yaw
        if t_yaw is not None:
            low, high = AXIS_BOUNDS['roll']
            heading_error = wrap_angle_deg(t_yaw - current_yaw)
            desired_roll = clamp(ctrls['heading'].compute(heading_error, -current_yaw_rate, dt), low, high)
        
        # normal roll calc
        elif t_roll is not None:
            low, high = AXIS_BOUNDS['roll']
            desired_roll = clamp(t_roll, low, high)
        else:
            desired_roll = 0.0

        # if plane decides to go to low just in case
        if current_alt < TAKEOFF_ALT_TARGET:
            cmd.update('pitch', 0.0)
            cmd.update('roll', 0.0)
            cmd.update('alt', TAKEOFF_ALT_TARGET)
            override = True

        elif current_alt >= TAKEOFF_ALT_TARGET and override:
            override = False

        # put the desired vals at hybrid controller 
        pitch_error = desired_pitch - current_pitch
        roll_error = desired_roll - current_roll
        pitch_correction = ctrls['pitch_att'].compute(pitch_error, -current_pitch_rate, dt)
        roll_correction = ctrls['roll_att'].compute(roll_error, -current_roll_rate, dt)

        low, high = AXIS_BOUNDS['pitch']
        target_pitch = clamp(desired_pitch + pitch_correction, low, high)

        low, high = AXIS_BOUNDS['roll']
        target_roll = clamp(desired_roll + roll_correction, low, high)

        # i will not touch calc related to thrust this shit crashes the plane 
        thrust_step_limit = 0.8 * dt
        current_thrust += clamp(trim_thrust - current_thrust, -thrust_step_limit, thrust_step_limit)
        current_thrust = clamp(current_thrust, 0.2, 1.0)

        # send the dnew vals to the plane
        connection.mav.rc_channels_override_send(
            connection.target_system, connection.target_component,
            angle_to_pwm(target_roll, AXIS_BOUNDS['roll'][1]),  # Roll
            angle_to_pwm(target_pitch, AXIS_BOUNDS['pitch'][1]),  # Pitch
            throttle_to_pwm(current_thrust),  # Throttle
            1500,  # Yaw (Rudder - centered)
            0, 0, 0, 0
        )
        time.sleep(0.05)  # ~20 Hz

if __name__ == '__main__':
    auto_and_arm()
    run()