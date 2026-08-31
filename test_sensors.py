from simulated_sensors import (
    get_rf_confidence,
    get_radar_confidence
)


for health in [100, 80, 60, 40, 20]:

    rf = get_rf_confidence(health)
    radar = get_radar_confidence(health)

    print(
        f"Health: {health}% | "
        f"RF: {rf * 100:.1f}% | "
        f"Radar: {radar * 100:.1f}%"
    )