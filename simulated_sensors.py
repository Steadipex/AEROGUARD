import random


def get_rf_confidence(environment_health):
    """
    Simulates RF detection confidence.
    RF is relatively independent of camera visibility.
    """

    base = 0.80

    # Add small variation
    variation = random.uniform(-0.05, 0.05)

    confidence = base + variation

    # Slight degradation at extreme conditions
    if environment_health < 40:
        confidence -= 0.10

    return max(0.0, min(1.0, confidence))


def get_radar_confidence(environment_health):
    """
    Simulates radar detection confidence.
    Radar remains relatively reliable in degraded visibility.
    """

    base = 0.90

    variation = random.uniform(-0.04, 0.04)

    confidence = base + variation

    if environment_health < 30:
        confidence -= 0.05

    return max(0.0, min(1.0, confidence))