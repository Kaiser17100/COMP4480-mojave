from pymavlink import mavutil
import time
import math



connection = mavutil.mavlink_connection('udp:127.0.0.1:14550')
connection.wait_heartbeat()

print("Connected...")

# 10Hz data stream
connection.mav.request_data_stream_send(
    connection.target_system,
    connection.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    10,
    1
)

def addMissions():
    print("Adding Missions...")
    missionCount = 9
    
    # clearing prev mission if had any
    connection.mav.mission_clear_all_send(
        connection.target_system, # our plane's id
        connection.target_component # plane's brain 
    )
    time.sleep(1)
    
    # how many wp does our mission have and sending it
    connection.mav.mission_count_send(
        connection.target_system, # plane id
        connection.target_component, # plane's brain
        missionCount
    )
    
    
    # actually sending the wps
    # ADD MORE WP
    for i in range(missionCount):
        mes = connection.recv_match(type='MISSION_REQUEST', blocking=True)
        print(mes.seq)
        if mes.seq == 0:
            # home wp
            connection.mav.mission_item_send(
                connection.target_system, # id
                connection.target_component, # brain
                0, # waypoint 0
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, # our command
                0, # current mission
                0, # autocontinue
                0, 0, 0, 0, -35.36323543, 149.1652924, 0
            )
        if mes.seq == 1:
            # takeoff wp
            connection.mav.mission_item_send(
                connection.target_system, #id 
                connection.target_component, # brain
                1, # waypoint 1
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, # our command
                1, # current mission
                1, # autocontinue 
                15, 0, 0, 0,-35.36328112, 149.16473343, 50
            )
        elif mes.seq == 2:
            # wp 2
            connection.mav.mission_item_send(
                connection.target_system, #id 
                connection.target_component, # brain
                2, # waypoint 2
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, # our command
                0, # current mission
                1, # autocontinue
                0, 30, 0, 0, -35.36157265, 149.16642059, 50  # alti
            )
        elif mes.seq == 3:
            # wp 3
            connection.mav.mission_item_send(
                connection.target_system, #id 
                connection.target_component, # brain
                3, # waypoint 3
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, # our command
                0, # current mission
                1, # autocontinue
                0, 30, 0, 0, -35.36043321, 149.16751750, 50
            )
        elif mes.seq == 4:
            # wp 4
            connection.mav.mission_item_send(
                connection.target_system, #id 
                connection.target_component, # brain
                4, # waypoint 4
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, # our command
                0, # current mission
                1, # autocontinue
                0, 30, 0, 0, -35.35933390, 149.16589693, 50
            )
        elif mes.seq == 5:
            # wp 5
            connection.mav.mission_item_send(
                connection.target_system, #id 
                connection.target_component, # brain
                5, # waypoint 5
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, # our command
                0, # current mission
                1, # autocontinue
                0, 30, 0, 0, -35.36026028, 149.16483674, 50
            )
        elif mes.seq == 6:
            # wp 6
            connection.mav.mission_item_send(
                connection.target_system, #id 
                connection.target_component, # brain
                6, # waypoint 6
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, # our command
                0, # current mission
                1, # autocontinue
                0, 10, 0, 0, -35.36121136, 149.16497305, 35
            )
        elif mes.seq == 7:
            # wp 7
            connection.mav.mission_item_send(
                connection.target_system, #id 
                connection.target_component, # brain
                7, # waypoint 8
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, # our commmand
                0, # current mission
                1, # autocontinue
                0, 5, 0, 0, -35.36220681, 149.16505294, 20
            )
        elif mes.seq == 8:
            # landing wp
            connection.mav.mission_item_send(
                connection.target_system, # id 
                connection.target_component, # brain
                8, # waypoint 8
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_LAND, # our command
                0, # current mission
                0, # autocontinue 
                0, 15, 0, 0, -35.36304432, 149.16519650, 0
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

def autoAndArm():
    connection.set_mode_auto()
    print("Mode Set to Auto...")
    connection.arducopter_arm()
    print("Arming...")

def printValues():
    # variables for storing parameters
    lat = 0
    lon = 0
    alti_ft = 0
    alti = 0
    airSpeed = 0
    groundSpeed = 0
    heading = 0
    roll = 0
    pitch = 0
    yaw = 0
    while True:
        # getting the flight parameters
        mes = connection.recv_match(type = ['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD'], blocking = True)
        
        if not mes:
            continue
        # calcing / storing the parameters
        if mes.get_type() == 'GLOBAL_POSITION_INT':
            lat = mes.lat / 10000000
            lon = mes.lon / 10000000
            alti = (mes.relative_alt / 1000)
            alti_ft = (mes.relative_alt / 1000) * 3.2808399
            
        if mes.get_type() == 'ATTITUDE':
            roll = math.degrees(mes.roll)
            yaw = math.degrees(mes.yaw)
            pitch = math.degrees(mes.pitch)
        
        if mes.get_type() == 'VFR_HUD':
            airSpeed = mes.airspeed
            groundSpeed = mes.groundspeed
            heading = mes.heading
        
        # printing the parameters
        print("----------------------")
        print(f"Latitude: {lat}")
        print(f"Longitude: {lon}")
        print(f"Altitude (ft): {alti_ft}")
        print(f"Altitude (m): {alti}")
        print(f"Roll: {roll}")
        print(f"Pitch: {pitch}")
        print(f"Yaw: {yaw}")
        print(f"Air Speed: {airSpeed}")
        print(f"Ground Speed: {groundSpeed}")
        print(f"Heading: {heading}")
        time.sleep(0.1)

addMissions()
autoAndArm()
printValues()
