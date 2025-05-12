class PIDFeedforwardController:
    def __init__(self, Kp, Ki, Kd, Kff=0.0, dt=0.01):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kff = Kff  # ganho do feedforward
        self.dt = dt

        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint, measurement):
        error = setpoint - measurement
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt

        u_pid = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        u_ff = self.Kff * setpoint  # alimentação direta baseada na referência

        self.prev_error = error
        return u_pid + u_ff