from pymavlink import mavutil
import time
import math

missions = [
    [0, 1, 0, 16, 0, 0, 0, 0, 38.7009336, 27.4539365, 10.14, 1],
    [1, 0, 3, 22, 0, 0, 0, 0, 0, 0, 50.0, 1],
    [2, 0, 3, 16, 0, 0, 0, 0, 38.7023076, 27.4582350, 50.0, 1],
    [3, 0, 3, 16, 0, 0, 0, 0, 38.7043757, 27.4559176, 50.0, 1],
    [4, 0, 3, 16, 0, 0, 0, 0, 38.7055311, 27.4528062, 50.0, 1],
    [5, 0, 3, 16, 0, 0, 0, 0, 38.7057488, 27.4495769, 50.0, 1],
    [6, 0, 3, 16, 0, 0, 0, 0, 38.7054474, 27.4465084, 50.0, 1],
    [7, 0, 3, 16, 0, 0, 0, 0, 38.7033375, 27.4457037, 35.0, 1],
    [8, 0, 3, 16, 0, 0, 0, 0, 38.7017048, 27.4496305, 25.0, 1],
    [9, 0, 3, 21, 0, 0, 0, 0, 38.7010015, 27.4534285, 0.0, 1]
]

# Global PID Variables
prev_time = time.time()
p_integral = 0
p_prev_error = 0
a_integral = 0
a_prev_error = 0

kp_p = 15.0
kd_p = 5.0
ki_p = 0.6

kp_a = 15.0
kd_a = 0.5
ki_a = 5.0

connection = mavutil.mavlink_connection('udp:127.0.0.1:14550')
connection.wait_heartbeat()
print("Connected...")

def addMissions():
    print("Uploading Missions...")
    connection.mav.mission_clear_all_send(connection.target_system, connection.target_component)
    time.sleep(0.5)
    connection.mav.mission_count_send(connection.target_system, connection.target_component, len(missions))
    for i in range(len(missions)):
        msg = connection.recv_match(type='MISSION_REQUEST', blocking=True)
        connection.mav.mission_item_send(
            connection.target_system, connection.target_component,
            missions[i][0], missions[i][2], missions[i][3], missions[i][1], missions[i][11],
            missions[i][4], missions[i][5], missions[i][6], missions[i][7],
            missions[i][8], missions[i][9], missions[i][10]
        )
    print("Mission Accepted.")

def control_pitch_and_altitude(target_pitch, current_pitch, target_alt, current_alt):
    global p_prev_error, p_integral, a_prev_error, a_integral, prev_time
    
    now = time.time()
    dt = now - prev_time
    if dt <= 0: dt = 0.1
    
    p_error = target_pitch - current_pitch
    p_integral += p_error * dt
    p_deriv = (p_error - p_prev_error) / dt
    p_out = (kp_p * p_error) + (ki_p * p_integral) + (kd_p * p_deriv)
    elevator_pwm = int(1500 + p_out)
    elevator_pwm = max(1000, min(2000, elevator_pwm))
    print(elevator_pwm / 1000.0)

    a_error = target_alt - current_alt
    a_integral += a_error * dt
    a_deriv = (a_error - a_prev_error) / dt
    a_out = (kp_a * a_error) + (ki_a * a_integral) + (kd_a * a_deriv)
    throttle_pwm = int(1500 + a_out)    
    throttle_pwm = max(1000, min(2000, throttle_pwm))

    connection.mav.rc_channels_override_send(
        connection.target_system, connection.target_component,
        65535, elevator_pwm, throttle_pwm,
        65535, 65535, 65535, 65535, 65535
    )
    
    p_prev_error = p_error
    a_prev_error = a_error
    prev_time = now

def release_control():
    """Hand all channels back to the autopilot."""
    connection.mav.rc_channels_override_send(
        connection.target_system, connection.target_component,
        0, 0, 0, 0, 0, 0, 0, 0
    )

def do_wp3_guided_loiter():
    """
    Switch to GUIDED, hold 10° pitch at 50 m for 10 seconds,
    release RC overrides, then resume AUTO from the next waypoint (wp4).
    """
    print("WP3 reached — switching to GUIDED for 10s pitch hold...")
    connection.set_mode('GUIDED')
    time.sleep(0.3)   # give FC time to accept mode change

    loiter_start = time.time()
    loiter_duration = 20.0   # seconds

    while (time.time() - loiter_start) < loiter_duration:
        msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT'], blocking=True, timeout=1.0)
        if msg is None:
            continue

        if msg.get_type() == 'ATTITUDE':
            pitch = math.degrees(msg.pitch)
            alt_msg = connection.messages.get('GLOBAL_POSITION_INT')
            if alt_msg:
                curr_alt = alt_msg.relative_alt / 1000.0
                elapsed = time.time() - loiter_start
                print(f"  GUIDED loiter {elapsed:.1f}s/{loiter_duration:.0f}s | Pitch {pitch:.1f}°, Alt {curr_alt:.1f}m")
                control_pitch_and_altitude(10.0, pitch, 50.0, curr_alt)

        time.sleep(0.1)

    print("WP3 loiter complete — releasing overrides and resuming AUTO...")
    release_control()
    time.sleep(0.2)

    # Resume AUTO; the FC will continue from wp4 (the next unvisited item)
    connection.set_mode('AUTO')

def run_flight_loop():
    current_wp = 0
    prev_wp = 0
    wp3_loiter_done = False

    while True:
        m_msg = connection.recv_match(type='MISSION_CURRENT', blocking=False)
        if m_msg:
            current_wp = m_msg.seq

        # Detect the exact moment the autopilot advances past wp3 → wp4
        if not wp3_loiter_done and prev_wp == 3 and current_wp == 4:
            wp3_loiter_done = True
            do_wp3_guided_loiter()

        prev_wp = current_wp

        msg = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT'], blocking=True)

        if msg.get_type() == 'ATTITUDE':
            pitch = math.degrees(msg.pitch)
            alt_msg = connection.messages.get('GLOBAL_POSITION_INT')
            if alt_msg:
                curr_alt = alt_msg.relative_alt / 1000.0
                # Normal AUTO flight — no RC override needed
                print(f"WP{current_wp} | Pitch {pitch:.1f}°, Alt {curr_alt:.1f}m")

        time.sleep(0.1)

# Start
addMissions()
connection.set_mode('AUTO')
connection.mav.command_long_send(
    connection.target_system, connection.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)

run_flight_loop()



