def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))

def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0

def angle_to_pwm(angle: float, max_angle: float) -> int:
    constrained_angle = clamp(angle, -max_angle, max_angle)
    return int(1500 + (constrained_angle / max_angle) * 500)

def throttle_to_pwm(thrust: float) -> int:
    thrust = clamp(thrust, 0.0, 1.0)
    return int(1000 + thrust * 1000)

def compute_center_deviation(x1, y1, x2, y2, frame_w, frame_h):
    obj_cx = (x1 + x2) / 2.0
    obj_cy = (y1 + y2) / 2.0
    img_cx = frame_w / 2.0
    img_cy = frame_h / 2.0

    dx = obj_cx - img_cx
    dy = obj_cy - img_cy

    dx_norm = dx / img_cx if img_cx != 0 else 0.0
    dy_norm = dy / img_cy if img_cy != 0 else 0.0

    return obj_cx, obj_cy, dx_norm, dy_norm

def get_bearing(lat1, lon1, lat2, lon2):
    import math
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon_diff_rad = math.radians(lon2 - lon1)
    
    x = math.sin(lon_diff_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon_diff_rad)
    
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    return (initial_bearing + 360) % 360

def get_distance(lat1, lon1, lat2, lon2):
    import math
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def find_closest_target(current_lat, current_lon, targets, max_distance):
    closest_target = None
    closest_distance = float('inf')
    for target in targets:
        distance = get_distance(current_lat, current_lon, target['iha_enlem'], target['iha_boylam'])
        if distance < closest_distance:
            closest_distance = distance
            closest_target = target

    if closest_distance <= max_distance:
        return closest_target

    return None


def compute_apf_hss(current_lat, current_lon, current_yaw, current_spd, desired_yaw, hss_list, lookahead_time=2.0, safety_margin=50.0, eta=75000.0, k_att=1.0):
    import math
    
    # Drone's local future position based on current velocity
    yaw_rad = math.radians(current_yaw)
    v_x = current_spd * math.sin(yaw_rad) # East
    v_y = current_spd * math.cos(yaw_rad) # North
    
    p_future_x = v_x * lookahead_time
    p_future_y = v_y * lookahead_time
    
    total_rep_x = 0.0
    total_rep_y = 0.0
    
    for hss in hss_list:
        hss_lat = hss.get("hssEnlem", 0.0)
        hss_lon = hss.get("hssBoylam", 0.0)
        hss_radius = hss.get("hssYaricap", 0.0)
        
        # Calculate distance and bearing from drone to HSS
        d_center = get_distance(current_lat, current_lon, hss_lat, hss_lon)
        b_center = get_bearing(current_lat, current_lon, hss_lat, hss_lon)
        
        b_rad = math.radians(b_center)
        h_x = d_center * math.sin(b_rad)
        h_y = d_center * math.cos(b_rad)
        
        # Distance from future position to HSS center
        dx = p_future_x - h_x
        dy = p_future_y - h_y
        d_future = math.sqrt(dx**2 + dy**2)
        
        # Distance to the boundary of the HSS area
        d_obs = d_future - hss_radius
        
        # If within the safety margin (influence radius of the obstacle)
        if d_obs < safety_margin:
            d_obs_clamped = max(d_obs, 0.1)  # prevent singularity if inside or exactly on the boundary
            
            # Calculate repulsive force magnitude based on distance to boundary
            f_rep = eta * (1.0 / d_obs_clamped - 1.0 / safety_margin) * (1.0 / (d_obs_clamped**2))
            
            # Direction from HSS center to future position (pushing radially away)
            if d_future > 0.001:
                dir_x = dx / d_future
                dir_y = dy / d_future
            else:
                dir_x = 1.0
                dir_y = 0.0
            
            total_rep_x += f_rep * dir_x
            total_rep_y += f_rep * dir_y

    if total_rep_x == 0 and total_rep_y == 0:
        return desired_yaw # No repulsive forces acting
        
    # Attractive force vector (where the drone wants to go)
    desired_yaw_rad = math.radians(desired_yaw)
    f_att_mag = k_att * max(current_spd, 15.0) # Base vector length on speed
    att_x = f_att_mag * math.sin(desired_yaw_rad)
    att_y = f_att_mag * math.cos(desired_yaw_rad)
    
    # Combined vector
    total_x = att_x + total_rep_x
    total_y = att_y + total_rep_y
    
    new_yaw = math.degrees(math.atan2(total_x, total_y))
    return (new_yaw + 360.0) % 360.0
