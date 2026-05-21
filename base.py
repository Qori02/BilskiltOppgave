import cv2
from ultralytics import YOLO
import torch

# ==========================================================
# SETTINGS
# ==========================================================

VIDEO_SOURCE = "test4.mp4"

VEHICLE_MODEL_PATH = "yolov8n.pt"
PLATE_MODEL_PATH = "license_plate_detector.pt"

# COCO vehicle classes:
# car = 2, motorcycle = 3, bus = 5, truck = 7
VEHICLE_CLASSES = [2, 3, 5, 7]

CONF_VEHICLE = 0.25
CONF_PLATE = 0.30

# Processing resolution
PROCESS_WIDTH = 1920
PROCESS_HEIGHT = 1080

# Display resolution
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Speed measurement lines
LINE_A_Y = 500
LINE_B_Y = 830

REAL_DISTANCE_METERS = 10
SPEED_LIMIT_KMH = 50

MIN_BOX_WIDTH = 50

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ==========================================================
# LOAD MODELS
# ==========================================================

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

vehicle_model = YOLO(VEHICLE_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)

# ==========================================================
# VIDEO
# ==========================================================

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError("Could not open video file.")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Running speed + vehicle + plate detection.")
print(f"Original video: {width}x{height}")
print(f"FPS: {fps}")
print("Press Q or ESC to stop.")

# ==========================================================
# STORAGE
# ==========================================================

first_line = {}
first_frame = {}
measured = set()
previous_positions = {}
speed_results = {}

frame_count = 0

# ==========================================================
# LOOP
# ==========================================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    frame_count += 1

    frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    clean_frame = frame.copy()

    # ======================================================
    # VEHICLE TRACKING
    # ======================================================

    vehicle_results = vehicle_model.track(
        frame,
        persist=True,
        classes=VEHICLE_CLASSES,
        conf=CONF_VEHICLE,
        verbose=False,
        device=DEVICE
    )

    # ======================================================
    # DRAW SPEED LINES
    # ======================================================

    cv2.line(frame, (0, LINE_A_Y), (frame.shape[1], LINE_A_Y), (0, 255, 255), 3)
    cv2.line(frame, (0, LINE_B_Y), (frame.shape[1], LINE_B_Y), (255, 0, 255), 3)

    cv2.putText(
        frame,
        "LINE A",
        (20, LINE_A_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "LINE B",
        (20, LINE_B_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2
    )

    # ======================================================
    # PROCESS VEHICLES
    # ======================================================

    if vehicle_results and vehicle_results[0].boxes is not None:

        for box in vehicle_results[0].boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])
            class_id = int(box.cls[0])
            class_name = vehicle_model.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if (x2 - x1) < MIN_BOX_WIDTH:
                continue

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            prev_y = previous_positions.get(track_id, center_y)

            # ==================================================
            # SPEED MEASUREMENT - BOTH DIRECTIONS
            # ==================================================

            crossed_a = (
                (prev_y < LINE_A_Y and center_y >= LINE_A_Y) or
                (prev_y > LINE_A_Y and center_y <= LINE_A_Y)
            )

            crossed_b = (
                (prev_y < LINE_B_Y and center_y >= LINE_B_Y) or
                (prev_y > LINE_B_Y and center_y <= LINE_B_Y)
            )

            if track_id not in first_line:

                if crossed_a:
                    first_line[track_id] = "A"
                    first_frame[track_id] = frame_count
                    print(f"Track {track_id} crossed LINE A first")

                elif crossed_b:
                    first_line[track_id] = "B"
                    first_frame[track_id] = frame_count
                    print(f"Track {track_id} crossed LINE B first")

            elif track_id not in measured:

                if first_line[track_id] == "A" and crossed_b:
                    frame_diff = frame_count - first_frame[track_id]
                    elapsed = frame_diff / fps

                    if elapsed > 0:
                        speed_kmh = (REAL_DISTANCE_METERS / elapsed) * 3.6
                        status = "SPEEDING" if speed_kmh > SPEED_LIMIT_KMH else "OK"

                        speed_results[track_id] = {
                            "speed": speed_kmh,
                            "status": status,
                            "direction": "A->B"
                        }

                        print(
                            f"Track {track_id} | Direction A->B | "
                            f"Speed: {speed_kmh:.2f} km/h | {status}"
                        )

                        measured.add(track_id)

                elif first_line[track_id] == "B" and crossed_a:
                    frame_diff = frame_count - first_frame[track_id]
                    elapsed = frame_diff / fps

                    if elapsed > 0:
                        speed_kmh = (REAL_DISTANCE_METERS / elapsed) * 3.6
                        status = "SPEEDING" if speed_kmh > SPEED_LIMIT_KMH else "OK"

                        speed_results[track_id] = {
                            "speed": speed_kmh,
                            "status": status,
                            "direction": "B->A"
                        }

                        print(
                            f"Track {track_id} | Direction B->A | "
                            f"Speed: {speed_kmh:.2f} km/h | {status}"
                        )

                        measured.add(track_id)

            previous_positions[track_id] = center_y

            # ==================================================
            # DRAW VEHICLE BOX
            # ==================================================

            vehicle_color = (0, 255, 0)

            if track_id in speed_results:
                if speed_results[track_id]["status"] == "SPEEDING":
                    vehicle_color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                vehicle_color,
                2
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # ==================================================
            # PLATE DETECTION INSIDE VEHICLE
            # ==================================================

            vehicle_crop = clean_frame[y1:y2, x1:x2]

            if vehicle_crop.size > 0:
                plate_results = plate_model(
                    vehicle_crop,
                    conf=CONF_PLATE,
                    verbose=False,
                    device=DEVICE
                )

                if plate_results and plate_results[0].boxes is not None:

                    for plate_box in plate_results[0].boxes:

                        px1, py1, px2, py2 = map(int, plate_box.xyxy[0])

                        real_x1 = x1 + px1
                        real_y1 = y1 + py1
                        real_x2 = x1 + px2
                        real_y2 = y1 + py2

                        cv2.rectangle(
                            frame,
                            (real_x1, real_y1),
                            (real_x2, real_y2),
                            (255, 0, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            "plate",
                            (real_x1, real_y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            2
                        )

            # ==================================================
            # LABEL
            # ==================================================

            label = f"{class_name} ID:{track_id}"

            if track_id in speed_results:
                speed = speed_results[track_id]["speed"]
                status = speed_results[track_id]["status"]
                label += f" {speed:.1f}km/h {status}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                vehicle_color,
                2
            )

    # ======================================================
    # DISPLAY
    # ======================================================

    display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("Speed + Vehicle + Plate", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        print("Stopping...")
        break

# ==========================================================
# CLEANUP
# ==========================================================

cap.release()
cv2.destroyAllWindows()

print("Program stopped.")