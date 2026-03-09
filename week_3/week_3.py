from pymavlink import mavutil
import time
import math

missions = [
    [0, 1, 0, 16, 0, 0, 0, 0, 38.7009336, 27.4539365, 10.14, 1], # home wp
    [1, 0, 3, 22, 0, 0, 0, 0, 0, 0, 50.0, 1],                    # takeoff wp
    [2, 0, 3, 16, 0, 0, 0, 0, 38.7023076, 27.4582350, 50.0, 1],  # wp1
    [3, 0, 3, 16, 0, 0, 0, 0, 38.7043757, 27.4559176, 50.0, 1],  # wp2
    [4, 0, 3, 16, 0, 0, 0, 0, 38.7055311, 27.4528062, 50.0, 1],  # wp3
    [5, 0, 3, 16, 0, 0, 0, 0, 38.7057488, 27.4495769, 50.0, 1],  # wp4
    [6, 0, 3, 16, 0, 0, 0, 0, 38.7054474, 27.4465084, 50.0, 1],  # wp5
    [7, 0, 3, 16, 0, 0, 0, 0, 38.7033375, 27.4457037, 35.0, 1],  # wp6
    [8, 0, 3, 16, 0, 0, 0, 0, 38.7017048, 27.4496305, 25.0, 1],  # wp7
    [9, 0, 3, 21, 0, 0, 0, 0, 38.7010015, 27.4534285, 0.0, 1]    # land wp
]

# Global PID Variables
prev_time = time.time()
p_integral = p_prev_error = 0
a_integral = a_prev_error = 0

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
    print("Adding Missions...")
    
    # clearing prev mission if had any
    connection.mav.mission_clear_all_send(
        connection.target_system, # our plane's id
        connection.target_component # plane's brain 
    )
    time.sleep(1)

    connection.mav.mission_count_send(
        connection.target_system, # plane id
        connection.target_component, # plane's brain
        len(missions))

    for i in range(len(missions)):
        connection.mav.mission_item_send(
            connection.target_system,    # id
            connection.target_component, # brain
            missions[i][0], # wp no
            missions[i][2], #
            missions[i][3], # command
            missions[i][1], # current mission flag
            missions[i][11], # autocontinue flag
            missions[i][4], missions[i][5], missions[i][6], missions[i][7],
            missions[i][8], missions[i][9], missions[i][10]
        )

    # waiting for ACK to check whether our mission send failed or not
    while True:
        ack = connection.recv_match(type='MISSION_ACK', blocking=True)
        if ack:
            if ack.type == 0:
                print("Mission accepted \nContinuing with AUTO and arm...")
            else:
                print("Mission rejected or failed.")
        break

def controlPitchAndAltitude(target_pitch, current_pitch, target_alt, current_alt):
    global p_prev_error, p_integral, a_prev_error, a_integral, prev_time

    # time to calc change in time
    now = time.time()
    dt = now - prev_time

    # divide by zero problem
    if dt <= 0: dt = 0.1


    # u(t) = Kp*e(t) + * Ki*integral(e(t)) + Kd*derivative(e(t))
    p_error = target_pitch - current_pitch # e(t)
    p_integral += p_error * dt # integral(e(t))
    p_deriv = (p_error - p_prev_error) / dt # derivative(e(t))
    p_out = (kp_p * p_error) + (ki_p * p_integral) + (kd_p * p_deriv) # u(t)

    # to limit the outcome
    elevator_pwm = int(1500 + p_out)
    elevator_pwm = max(1000, min(2000, elevator_pwm))

    # same as before
    a_error = target_alt - current_alt
    a_integral += a_error * dt
    a_deriv = (a_error - a_prev_error) / dt
    a_out = (kp_a * a_error) + (ki_a * a_integral) + (kd_a * a_deriv)
    throttle_pwm = int(1500 + a_out)    
    throttle_pwm = max(1000, min(2000, throttle_pwm))

    # override the pitch value
    connection.mav.rc_channels_override_send(
        connection.target_system, connection.target_component,
        65535, elevator_pwm, throttle_pwm,
        65535, 65535, 65535, 65535, 65535
    )

    p_prev_error = p_error
    a_prev_error = a_error
    prev_time = now

def guidedLoiter():
    loiter_duration = 20.0
    print(f"Switching to GUIDED for {loiter_duration} pitch hold...")
    connection.set_mode('GUIDED')
    time.sleep(0.1)

    loiter_start = time.time()


    while time.time() - loiter_start < loiter_duration:
        mes = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT'], blocking=True)
        if not mes:
            continue

        if mes.get_type() == 'ATTITUDE':
            pitch = math.degrees(mes.pitch)
            alt_msg = connection.messages.get('GLOBAL_POSITION_INT')
            if alt_msg:
                curr_alt = alt_msg.relative_alt / 1000.0
                print(f"Pitch {pitch}, Alt {curr_alt}")
                controlPitchAndAltitude(10.0, pitch, 50.0, curr_alt)

        time.sleep(0.1)

    print("Resuming AUTO...")
    connection.mav.rc_channels_override_send(
        connection.target_system, connection.target_component,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    time.sleep(0.1)

    # Resume AUTO; the FC will continue from wp4 (the next unvisited item)
    connection.set_mode('AUTO')

def run():
    pitch = curr_alt = current_wp = 0
    loiter_done = False

    while True:
        mission_mes = connection.recv_match(type='MISSION_CURRENT', blocking=False)
        if mission_mes:
            current_wp = mission_mes.seq

        if not wp3_loiter_done and current_wp == 4:
            loiter_done = True
            guidedLoiter()

        mes = connection.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT'], blocking=True)

        if mes.get_type() == 'ATTITUDE':
            pitch = math.degrees(mes.pitch)
            print(f"Pitch {pitch}")

        if mes.get_type() == 'GLOBAL_POSITION_INT':
            curr_alt = mes.relative_alt / 1000.0
            print(f"Alt {curr_alt}")


        time.sleep(0.1)

# Start
addMissions()
connection.set_mode('AUTO')
connection.mav.command_long_send(
    connection.target_system, connection.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)

run()
