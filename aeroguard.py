from ultralytics import YOLO
from environment import EnvironmentSimulator
from health import calculate_health

import cv2
import time
import math


# ============================================================
# LOAD MODELS
# ============================================================

model = YOLO("aeroguard_drone.pt")

environment = EnvironmentSimulator()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera error")
    exit()


# ============================================================
# TRACKING DATA
# ============================================================

previous_positions = {}
trajectories = {}


# ============================================================
# FPS
# ============================================================

previous_time = time.time()


# ============================================================
# ENVIRONMENT UPDATE
# ============================================================

start_time = time.time()
last_environment_update = 0

environment_data = {
    "temperature": 10,
    "pressure": 850,
    "vibration": 0.2
}

system_health = 100


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break


    # ========================================================
    # UPDATE ENVIRONMENT ONCE PER SECOND
    # ========================================================

    current_time = time.time()
    elapsed = current_time - start_time

    if current_time - last_environment_update >= 1:

        # 0-15 seconds = normal
        if elapsed < 15:

            environment.set_mode("normal")

        # 15-30 seconds = high altitude
        elif elapsed < 30:

            environment.set_mode("high")

        # 30+ seconds = extreme
        else:

            environment.set_mode("extreme")


        # Get environmental data
        environment_data = environment.update()


        # Calculate system health
        system_health = calculate_health(
            environment_data["temperature"],
            environment_data["pressure"],
            environment_data["vibration"]
        )


        last_environment_update = current_time


    # ========================================================
    # YOLO + BYTETRACK
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.50,
        verbose=False
    )


    # ========================================================
    # DRAW DETECTIONS
    # ========================================================

    if results[0].boxes.id is not None:

        boxes = results[0].boxes

        track_ids = boxes.id.int().cpu().tolist()


        for box, track_id in zip(boxes, track_ids):

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = model.names[class_id]


            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Width / height
            width = x2 - x1
            height = y2 - y1


            # Center
            center_x = x1 + width // 2
            center_y = y1 + height // 2


            # =================================================
            # SPEED
            # =================================================

            velocity = 0

            if track_id in previous_positions:

                old_x, old_y, old_time = previous_positions[track_id]

                distance = math.sqrt(
                    (center_x - old_x) ** 2 +
                    (center_y - old_y) ** 2
                )

                time_difference = current_time - old_time

                if time_difference > 0:

                    velocity = distance / time_difference


            previous_positions[track_id] = (
                center_x,
                center_y,
                current_time
            )


            # =================================================
            # TRAJECTORY
            # =================================================

            if track_id not in trajectories:

                trajectories[track_id] = []


            trajectories[track_id].append(
                (center_x, center_y)
            )


            # Keep last 50 points
            if len(trajectories[track_id]) > 50:

                trajectories[track_id].pop(0)


            # Draw trajectory
            points = trajectories[track_id]

            for i in range(1, len(points)):

                cv2.line(
                    frame,
                    points[i - 1],
                    points[i],
                    (255, 0, 0),
                    2
                )


            # =================================================
            # BOUNDING BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # Center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )


            # =================================================
            # DRONE INFORMATION
            # =================================================

            label = (
                f"ID: {track_id} | "
                f"DRONE | "
                f"{confidence * 100:.1f}%"
            )


            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 30, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            position = (
                f"Pos: ({center_x}, {center_y})"
            )


            cv2.putText(
                frame,
                position,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )


            speed = (
                f"Speed: {velocity:.1f} px/s"
            )


            cv2.putText(
                frame,
                speed,
                (x1, y2 + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )


    # ========================================================
    # FPS
    # ========================================================

    fps = 1 / max(
        current_time - previous_time,
        0.001
    )

    previous_time = current_time


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "AEROGUARD - TARGET TRACKING",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ========================================================
    # ENVIRONMENT PANEL
    # ========================================================

    panel_x = 20
    panel_y = frame.shape[0] - 175

    # Background rectangle
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (430, frame.shape[0] - 15),
        (20, 20, 20),
        -1
    )


    cv2.putText(
        frame,
        "ENVIRONMENT",
        (panel_x + 10, panel_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # Temperature
    cv2.putText(
        frame,
        f"Temperature: "
        f"{environment_data['temperature']:.1f} C",
        (panel_x + 10, panel_y + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # Pressure
    cv2.putText(
        frame,
        f"Pressure: "
        f"{environment_data['pressure']:.1f} hPa",
        (panel_x + 10, panel_y + 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # Vibration
    cv2.putText(
        frame,
        f"Vibration: "
        f"{environment_data['vibration']:.2f} g",
        (panel_x + 10, panel_y + 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SYSTEM HEALTH
    # ========================================================

    if system_health >= 90:

        status = "NORMAL"

    elif system_health >= 70:

        status = "DEGRADED"

    else:

        status = "CRITICAL"


    cv2.putText(
        frame,
        f"System Health: {system_health}%",
        (panel_x + 10, panel_y + 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Status: {status}",
        (panel_x + 10, panel_y + 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "AEROGUARD",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()