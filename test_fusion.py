from sensor_fusion import calculate_fused_confidence

camera = 0.90
rf = 0.60
radar = 0.80
tracking = 0.85

for health in [100, 80, 60, 40, 20]:

    result = calculate_fused_confidence(
        camera,
        rf,
        radar,
        tracking,
        health
    )

    print(
        f"System Health: {health}% | "
        f"Fused Confidence: {result * 100:.1f}%"
    )