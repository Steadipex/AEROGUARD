import random


class EnvironmentSimulator:

    def __init__(self):

        # Start in normal conditions
        self.mode = "normal"

        self.temperature = 10.0
        self.pressure = 850.0
        self.vibration = 0.20

    def update(self):

        # -----------------------------
        # NORMAL CONDITIONS
        # -----------------------------
        if self.mode == "normal":

            target_temp = 10
            target_pressure = 850
            target_vibration = 0.20

        # -----------------------------
        # HIGH-ALTITUDE CONDITIONS
        # -----------------------------
        elif self.mode == "high":

            target_temp = -15
            target_pressure = 650
            target_vibration = 0.50

        # -----------------------------
        # EXTREME CONDITIONS
        # -----------------------------
        elif self.mode == "extreme":

            target_temp = -35
            target_pressure = 450
            target_vibration = 1.00

        # Gradually move toward target conditions
        self.temperature += (
            target_temp - self.temperature
        ) * 0.10

        self.pressure += (
            target_pressure - self.pressure
        ) * 0.10

        self.vibration += (
            target_vibration - self.vibration
        ) * 0.10

        # Add small random variations
        self.temperature += random.uniform(-1, 1)
        self.pressure += random.uniform(-5, 5)
        self.vibration += random.uniform(-0.05, 0.05)

        # Prevent negative vibration
        self.vibration = max(0, self.vibration)

        return {
            "temperature": self.temperature,
            "pressure": self.pressure,
            "vibration": self.vibration
        }

    def set_mode(self, mode):

        if mode in ["normal", "high", "extreme"]:
            self.mode = mode

        else:
            print("Invalid mode")