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