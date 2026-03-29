import time

# PID
class PIDController:
    def __init__(self, kp=12.0, ki=0.5, kd=0.6):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.time()

    def compute(self, error: float, external_rate: float) -> float:
        now = time.time()
        dt  = max(now - self._prev_time, 0.01)
        
        self._integral += error * dt
        out = (self.kp * error) + (self.ki * self._integral) + (self.kd * external_rate)
        
        self._prev_error = error
        self._prev_time  = now
        return out


# FUZZY
class FuzzyController:
    RESOLUTION = 200
    def __init__(self, error_range=180.0, rate_range=90.0, out_range=30.0):
        self.er  = error_range
        self.rr  = rate_range
        self.out = out_range
        
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
        
        self._labels = ['NL', 'NM', 'NS', 'ZE', 'PS', 'PM', 'PL']

    @staticmethod
    def _tri(x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        return (x - a) / (b - a) if x <= b else (c - x) / (c - b)

    def _fuzzify(self, value, universe_half):
        u = universe_half
        centres = [-u, -2*u/3, -u/3, 0, u/3, 2*u/3, u]
        step = u / 3
        
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