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

    dx = obj_cx - img_cx   # + ise sağda, - ise solda
    dy = obj_cy - img_cy

    dx_norm = dx / img_cx if img_cx != 0 else 0.0
    dy_norm = dy / img_cy if img_cy != 0 else 0.0

    return obj_cx, obj_cy, dx_norm, dy_norm