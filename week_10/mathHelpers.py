import math

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


def find_closest_target(current_lat, current_lon, targets, max_distance, team_no):
    closest_target = None
    closest_distance = float('inf')
    for target in targets:
        distance = get_distance(current_lat, current_lon, target['iha_enlem'], target['iha_boylam'])
        if distance < closest_distance and target.get('takim_numarasi') != team_no:
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
            
            # Vortex/Tangential field to prevent getting stuck in straight lines (local minima)
            # 90 degrees clockwise rotation of the radial vector
            tan_x = dir_y
            tan_y = -dir_x
            
            total_rep_x += f_rep * (dir_x + 0.8 * tan_x)
            total_rep_y += f_rep * (dir_y + 0.8 * tan_y)

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


def enforce_flight_boundaries(current_lat, current_lon, current_yaw, current_spd, desired_yaw, boundaries, lookahead_time=2.0, safety_margin=50.0, eta=75000.0, k_att=1.0):
    if len(boundaries) < 3:
        return desired_yaw
        
    origin_lat = boundaries[0][0]
    origin_lon = boundaries[0][1]
    
    meter_per_deg_lat = 111320.0
    meter_per_deg_lon = 111320.0 * math.cos(math.radians(origin_lat))
    
    def latlon_to_xy(lat, lon):
        x = (lon - origin_lon) * meter_per_deg_lon
        y = (lat - origin_lat) * meter_per_deg_lat
        return x, y
        
    p_x, p_y = latlon_to_xy(current_lat, current_lon)
    
    # Build polygon in XY
    polygon = []
    for lat, lon in boundaries:
        polygon.append(latlon_to_xy(lat, lon))
        
    # Ray casting to check if drone is inside
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > p_y) != (yj > p_y)) and (p_x < (xj - xi) * (p_y - yi) / (yj - yi) + xi)
        if intersect:
            inside = not inside
        j = i
        
    # Calculate centroid of polygon
    cx = sum(p[0] for p in polygon) / n
    cy = sum(p[1] for p in polygon) / n
    
    if not inside:
        # OUT OF BOUNDS: override everything and point directly to the center
        return (math.degrees(math.atan2(cx - p_x, cy - p_y)) + 360.0) % 360.0
        
    # INSIDE: check if we are approaching a wall
    yaw_rad = math.radians(current_yaw)
    v_x = current_spd * math.sin(yaw_rad)
    v_y = current_spd * math.cos(yaw_rad)
    
    p_future_x = p_x + v_x * lookahead_time
    p_future_y = p_y + v_y * lookahead_time
    
    desired_yaw_rad = math.radians(desired_yaw)
    vec_x = math.sin(desired_yaw_rad)
    vec_y = math.cos(desired_yaw_rad)
    
    repulsion_applied = False
    
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        
        l2 = (bx - ax)**2 + (by - ay)**2
        if l2 == 0:
            continue
            
        t = max(0.0, min(1.0, ((p_future_x - ax) * (bx - ax) + (p_future_y - ay) * (by - ay)) / l2))
        proj_x = ax + t * (bx - ax)
        proj_y = ay + t * (by - ay)
        
        dx = p_future_x - proj_x
        dy = p_future_y - proj_y
        d_obs = math.sqrt(dx**2 + dy**2)
        
        if d_obs < safety_margin:
            # We are close. Calculate inward normal.
            nx = -(by - ay)
            ny = (bx - ax)
            nl = math.sqrt(nx**2 + ny**2)
            if nl > 0:
                nx /= nl
                ny /= nl
                
            # Ensure it points towards centroid (inward)
            mx = (ax + bx) / 2.0
            my = (ay + by) / 2.0
            if (nx * (cx - mx) + ny * (cy - my)) < 0:
                nx = -nx
                ny = -ny
                
            # Strength increases as we get closer
            d_obs_clamped = max(d_obs, 0.1)
            strength = (safety_margin / d_obs_clamped) - 1.0
            
            # Tangent logic: push towards the middle of the wall to avoid corner traps
            tx = (bx - ax) / math.sqrt(l2)
            ty = (by - ay) / math.sqrt(l2)
            
            dot_mid = tx * (mx - proj_x) + ty * (my - proj_y)
            
            if abs(dot_mid) > 0.1:
                if dot_mid < 0:
                    tx = -tx
                    ty = -ty
            else:
                if (v_x * tx + v_y * ty) < 0:
                    tx = -tx
                    ty = -ty
                
            # Add inward push + a slight tangent to turn towards the safest area
            push_x = nx + 0.6 * tx
            push_y = ny + 0.6 * ty
            
            vec_x += push_x * strength * 2.0
            vec_y += push_y * strength * 2.0
            repulsion_applied = True
            
    if repulsion_applied:
        new_yaw = math.degrees(math.atan2(vec_x, vec_y))
        return (new_yaw + 360.0) % 360.0

    return desired_yaw

