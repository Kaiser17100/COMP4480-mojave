import threading

## SHARED COMMAND STATE ##
class CommandState:
    def __init__(self):
        self._lock = threading.Lock()
        self.target_pitch = None
        self.target_roll = None
        self.target_yaw = None
        self.target_alt = None
        self.target_speed = None
        self.override = False

    def update(self, axis: str, value):
        with self._lock: setattr(self, f'target_{axis}', value)

    def set_override(self, state: bool):
        with self._lock: self.override = state

    def snapshot(self):
        with self._lock:
            return (self.target_pitch, self.target_roll, self.target_yaw, self.target_alt,
                    self.target_speed)