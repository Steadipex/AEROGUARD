from ultralytics import YOLO
from sensor_fusion import calculate_fused_confidence
from simulated_sensors import (
    get_rf_confidence,
    get_radar_confidence
)
from environment import EnvironmentSimulator
from health import calculate_health
import cv2
import time
import math

# Load trained AEROGUARD model
model = YOLO("aeroguard_drone.pt")
environment = EnvironmentSimulator()

# Open webcam
cap = cv2.VideoCapture(0)

# FPS
previous_time = time.time()

# Previous positions
previous_positions = {}

# Trajectories
trajectories = {}
lost_frames = {}
predicted_positions = {}
mode = "NORMAL"
# Number of consecutive frames required
# before we confirm a drone
CONFIRM_FRAMES = 4

# Store how many consecutive frames each ID has appeared
track_confirmations = {}

# Keep track of the last frame each ID was seen
last_seen_frame = {}

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break
    # Get environmental conditions
    data = environment.update()

    # Calculate system health
    health = calculate_health(
        data["temperature"],
        data["pressure"],
        data["vibration"]
    )

    frame_number += 1

    # ------------------------------------------------
    # YOLO + ByteTrack
    # ------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",

        # Low enough to detect distant drones
        conf=0.25,

        verbose=False
    )

    boxes = results[0].boxes

    # ------------------------------------------------
    # Process detections
    # ------------------------------------------------
    

    if boxes.id is not None:
        detected_ids = boxes.id.int().cpu().tolist()
    else:
        detected_ids = []
    
    if boxes.id is not None:

        track_ids = boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):

            # Confidence
            camera_confidence = float(box.conf[0])
            # Simulate environmental degradation
            if health < 60:
                camera_confidence *= 0.55

            elif health < 80:
                camera_confidence *= 0.75
            # Simulated RF and radar confidence
            rf_confidence = get_rf_confidence(health)
            radar_confidence = get_radar_confidence(health)

            # Tracking confidence
            tracking_confidence = 0.90
            # Calculate fused confidence
            fused_confidence, mode = calculate_fused_confidence(
                camera_confidence,
                rf_confidence,
                radar_confidence,
                tracking_confidence,
                health
            )
            
            # Adaptive fused-confidence filtering
            if mode == "NORMAL":
                minimum_fused_confidence = 0.75

            elif mode == "DEGRADED":
                minimum_fused_confidence = 0.68

            else:
                minimum_fused_confidence = 0.60

            if fused_confidence < minimum_fused_confidence:
                continue
            
            # Sensor fusion status
            if mode == "NORMAL":
                sensor_status = "CAMERA + RF + RADAR"
            elif mode == "DEGRADED":
                sensor_status = "CAMERA + RF + RADAR (ADAPTIVE)"
            else:
                sensor_status = "RF + RADAR PRIORITY"
            # Class
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # We only have one class: drone
            if class_name != "drone":
                continue

            # ------------------------------------------------
            # CONFIRMATION SYSTEM
            # ------------------------------------------------

            # Was this ID seen in the immediately previous frame?
            if last_seen_frame.get(track_id) == frame_number - 1:

                track_confirmations[track_id] = \
                    track_confirmations.get(track_id, 0) + 1

            else:

                # New detection / lost track
                track_confirmations[track_id] = 1

            last_seen_frame[track_id] = frame_number

            confirmation_count = track_confirmations[track_id]

            # Is this drone confirmed?
            confirmed = confirmation_count >= CONFIRM_FRAMES
            
            if confirmed:
                lost_frames[track_id] = 0

            # ------------------------------------------------
            # Don't display weak/unconfirmed detections
            # ------------------------------------------------

            if not confirmed:
                continue

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )
            
            # Frame dimensions
            frame_height, frame_width = frame.shape[:2]

            # Bounding box area
            box_area = (x2 - x1) * (y2 - y1)

            # Frame area
            frame_area = frame_width * frame_height

            # How much of the frame the object occupies
            area_ratio = box_area / frame_area

            # Adaptive confidence filtering
            # Small/far drones can have lower confidence.
            # Very large detections need higher confidence.
            if area_ratio > 0.25 and fused_confidence < 0.65:
                continue
            
            width = x2 - x1
            height = y2 - y1

            # Center
            center_x = x1 + width // 2
            center_y = y1 + height // 2

            # ------------------------------------------------
            # Velocity
            # ------------------------------------------------

            current_time = time.time()

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

            # Save position
            previous_positions[track_id] = (
                center_x,
                center_y,
                current_time
            )

            # ------------------------------------------------
            # Trajectory
            # ------------------------------------------------

            if track_id not in trajectories:
                trajectories[track_id] = []

            trajectories[track_id].append(
                (center_x, center_y)
            )

            # Keep last 50 points
            if len(trajectories[track_id]) > 50:
                trajectories[track_id].pop(0)

            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Draw center
            # ------------------------------------------------

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # ------------------------------------------------
            # Draw trajectory
            # ------------------------------------------------

            points = trajectories[track_id]

            for i in range(1, len(points)):

                cv2.line(
                    frame,
                    points[i - 1],
                    points[i],
                    (255, 0, 0),
                    2
                )

            # ------------------------------------------------
            # Information
            # ------------------------------------------------

            label = (
                f"ID: {track_id} | "
                f"{class_name} | "
                f"Fused: {fused_confidence * 100:.1f}%"
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

            # Confirmation status
            confirmed_text = "CONFIRMED"

            cv2.putText(
                frame,
                confirmed_text,
                (x1, y2 + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"Sensors: {sensor_status}",
                (x1, y2 + 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )
    # ------------------------------------------------
    # Temporary track prediction
    # ------------------------------------------------

    for track_id in list(last_seen_frame.keys()):

        # Skip tracks detected in the current frame
        if track_id in detected_ids:
            continue

        # Only predict confirmed tracks
        if track_id not in previous_positions:
            continue

        lost_frames[track_id] = lost_frames.get(track_id, 0) + 1

        # Predict only for a short period
        if lost_frames[track_id] <= 10:

            center_x, center_y, current_time = previous_positions[track_id]

            # Estimate movement from previous trajectory
            predicted_x = center_x
            predicted_y = center_y

            if len(trajectories.get(track_id, [])) >= 2:

                previous_point = trajectories[track_id][-2]
                current_point = trajectories[track_id][-1]

                dx = current_point[0] - previous_point[0]
                dy = current_point[1] - previous_point[1]

                predicted_x = int(center_x + dx)
                predicted_y = int(center_y + dy)

            # Keep prediction inside frame
            predicted_x = max(0, min(frame_width - 1, predicted_x))
            predicted_y = max(0, min(frame_height - 1, predicted_y))

            # Draw predicted position
            cv2.circle(
                frame,
                (predicted_x, predicted_y),
                8,
                (0, 165, 255),
                2
            )

            cv2.putText(
                frame,
                f"ID: {track_id} | PREDICTED",
                (predicted_x + 10, predicted_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2
            )

            cv2.putText(
                frame,
                f"Lost: {lost_frames[track_id]} frames",
                (predicted_x + 10, predicted_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2
            )
            
    # ------------------------------------------------
    # Remove old tracks from memory
    # ------------------------------------------------

    old_ids = []

    for track_id in last_seen_frame:

        if frame_number - last_seen_frame[track_id] > 30:
            old_ids.append(track_id)

    for track_id in old_ids:

        last_seen_frame.pop(track_id, None)
        track_confirmations.pop(track_id, None)
        previous_positions.pop(track_id, None)
        trajectories.pop(track_id, None)

    # ------------------------------------------------
    # FPS
    # ------------------------------------------------

    current_time = time.time()

    time_difference = current_time - previous_time

    if time_difference > 0:
        fps = 1 / time_difference
    else:
        fps = 0

    previous_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.putText(
        frame,
        f"System Health: {health}%",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    cv2.putText(
        frame,
        f"MODE: {mode}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )
    # Environment scenario
    environment_mode = environment.mode.upper()
    cv2.putText(
        frame,
        f"ENVIRONMENT: {environment_mode}",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 200, 255),
        2
    )
    cv2.putText(
        frame,
        f"Temp: {data['temperature']:.1f} C",
        (20, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Pressure: {data['pressure']:.1f} hPa",
        (20, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Vibration: {data['vibration']:.2f} g",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )
    cv2.putText(
        frame,
        f"Camera: {camera_confidence * 100:.0f}%",
        (250, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"RF: {rf_confidence * 100:.0f}%",
        (250, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Radar: {radar_confidence * 100:.0f}%",
        (250, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )
    cv2.putText(
        frame,
        f"FUSION: {sensor_status}",
        (250, 265),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )
    

    # ------------------------------------------------
    # Title
    # ------------------------------------------------

    cv2.putText(
        frame,
        "AEROGUARD - TARGET TRACKING",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ------------------------------------------------
    # Show frame
    # ------------------------------------------------

    cv2.imshow(
        "AEROGUARD",
        frame
    )

    # ------------------------------------------------
    # Keyboard controls
    # ------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("n"):
        environment.set_mode("normal")

    elif key == ord("h"):
        environment.set_mode("high")

    elif key == ord("e"):
        environment.set_mode("extreme")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()