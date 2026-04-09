from mathHelpers import *

class PIDController:
    def __init__(
        self,
        kp=12.0,
        ki=0.5,
        kd=0.6,
        integral_limit=50.0,
        output_limit=None,
        integral_zone=None,
        rate_filter_tau=0.15,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.integral_zone = integral_zone
        self.rate_filter_tau = rate_filter_tau
        self._integral = 0.0
        self._filtered_rate = 0.0

    def compute(
        self,
        error: float,
        error_rate: float,
        dt: float,
        kp_scale: float = 1.0,
        ki_scale: float = 1.0,
        kd_scale: float = 1.0,
    ) -> float:
        dt = max(dt, 0.01)

        if self.rate_filter_tau <= 0.0:
            self._filtered_rate = error_rate
        else:
            alpha = dt / (self.rate_filter_tau + dt)
            self._filtered_rate += alpha * (error_rate - self._filtered_rate)

        prev_integral = self._integral
        integral_active = self.integral_zone is None or abs(error) <= self.integral_zone

        if integral_active and self.ki > 0.0 and ki_scale > 0.0:
            self._integral += error * dt
            self._integral = clamp(self._integral, -self.integral_limit, self.integral_limit)
        else:
            self._integral *= 0.98

        p_term = self.kp * kp_scale * error
        i_term = self.ki * ki_scale * self._integral
        d_term = self.kd * kd_scale * self._filtered_rate
        raw_output = p_term + i_term + d_term

        if self.output_limit is None:
            return raw_output

        output = clamp(raw_output, -self.output_limit, self.output_limit)
        if output != raw_output:
            saturated_high = raw_output > self.output_limit and error > 0.0
            saturated_low = raw_output < -self.output_limit and error < 0.0
            if saturated_high or saturated_low:
                self._integral = prev_integral
                i_term = self.ki * ki_scale * self._integral
                raw_output = p_term + i_term + d_term
                output = clamp(raw_output, -self.output_limit, self.output_limit)

        return output


class FuzzyGainScheduler:
    def __init__(self, error_range=180.0, rate_range=90.0):
        self.error_range = error_range
        self.rate_range = rate_range
        self.labels = ['S', 'M', 'L']
        self.kp_rules = [
            [0.75, 0.90, 0.70],
            [1.15, 1.35, 1.10],
            [1.85, 2.20, 1.65],
        ]
        self.ki_rules = [
            [1.60, 1.15, 0.65],
            [0.95, 0.75, 0.45],
            [0.35, 0.25, 0.20],
        ]
        self.kd_rules = [
            [0.75, 1.25, 1.85],
            [0.95, 1.45, 2.10],
            [1.10, 1.75, 2.35],
        ]

    @staticmethod
    def _triangle(x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        if a < x <= b:
            return (x - a) / (b - a)
        if b < x < c:
            return (c - x) / (c - b)
        return 0.0

    @staticmethod
    def _left_shoulder(x, a, b):
        if x <= a:
            return 1.0
        if x >= b:
            return 0.0
        return (b - x) / (b - a)

    @staticmethod
    def _right_shoulder(x, a, b):
        if x <= a:
            return 0.0
        if x >= b:
            return 1.0
        return (x - a) / (b - a)

    def _fuzzify_abs(self, normalized_value):
        x = clamp(normalized_value, 0.0, 1.0)
        return {
            'S': self._left_shoulder(x, 0.20, 0.45),
            'M': self._triangle(x, 0.20, 0.55, 0.85),
            'L': self._right_shoulder(x, 0.55, 0.90),
        }

    def _blend(self, rules, err_mu, rate_mu, default_value=1.0):
        numerator = 0.0
        denominator = 0.0
        for i, err_label in enumerate(self.labels):
            for j, rate_label in enumerate(self.labels):
                weight = min(err_mu[err_label], rate_mu[rate_label])
                if weight <= 0.0:
                    continue
                numerator += weight * rules[i][j]
                denominator += weight
        return default_value if denominator == 0.0 else numerator / denominator

    def compute_scales(self, error: float, error_rate: float) -> dict:
        err_mu = self._fuzzify_abs(abs(error) / self.error_range)
        rate_mu = self._fuzzify_abs(abs(error_rate) / self.rate_range)
        return {
            'kp_scale': self._blend(self.kp_rules, err_mu, rate_mu, 1.0),
            'ki_scale': self._blend(self.ki_rules, err_mu, rate_mu, 1.0),
            'kd_scale': self._blend(self.kd_rules, err_mu, rate_mu, 1.0),
        }


class HybridController:
    def __init__(self, pid_ctrl, fuzzy_ctrl):
        self.pid = pid_ctrl
        self.fuzzy = fuzzy_ctrl

    def reset(self):
        self.pid.reset()

    def compute(self, error: float, rate: float, dt: float) -> float:
        gain_scales = self.fuzzy.compute_scales(error, rate)
        return self.pid.compute(error, rate, dt, **gain_scales)