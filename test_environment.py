from environment import EnvironmentSimulator
from health import calculate_health
import time

environment = EnvironmentSimulator()

# Start with normal conditions
environment.set_mode("normal")

counter = 0

while True:

    # Change environment every 15 seconds
    if counter == 15:
        environment.set_mode("high")
        print("\n===== HIGH ALTITUDE CONDITIONS =====\n")

    if counter == 30:
        environment.set_mode("extreme")
        print("\n===== EXTREME CONDITIONS =====\n")

    data = environment.update()

    health = calculate_health(
        data["temperature"],
        data["pressure"],
        data["vibration"]
    )

    print(
        f"Temperature: {data['temperature']:.1f} °C | "
        f"Pressure: {data['pressure']:.1f} hPa | "
        f"Vibration: {data['vibration']:.2f} g | "
        f"System Health: {health}%"
    )

    counter += 1

    time.sleep(1)