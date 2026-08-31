def calculate_fused_confidence(
    camera_confidence,
    rf_confidence,
    radar_confidence,
    tracking_confidence,
    health
):

    if health >= 80:
        mode = "NORMAL"

        camera_weight = 0.40
        rf_weight = 0.20
        radar_weight = 0.25
        tracking_weight = 0.15

    elif health >= 60:
        mode = "DEGRADED"

        camera_weight = 0.20
        rf_weight = 0.30
        radar_weight = 0.35
        tracking_weight = 0.15

    else:
        mode = "CRITICAL"

        camera_weight = 0.10
        rf_weight = 0.35
        radar_weight = 0.40
        tracking_weight = 0.15

    fused_confidence = (
        camera_confidence * camera_weight +
        rf_confidence * rf_weight +
        radar_confidence * radar_weight +
        tracking_confidence * tracking_weight
    )

    return fused_confidence, mode