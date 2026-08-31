def calculate_health(temperature, pressure, vibration):

    health = 100

    # Temperature penalty
    if temperature < -10:
        health -= 15
    elif temperature < 0:
        health -= 5

    # Pressure penalty
    if pressure < 600:
        health -= 15
    elif pressure < 650:
        health -= 5

    # Vibration penalty
    if vibration > 0.8:
        health -= 20
    elif vibration > 0.5:
        health -= 10

    health = max(0, min(100, health))

    return health