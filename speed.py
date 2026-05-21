import cv2
from ultralytics import YOLO

#Setting

VIDEO_SOURCE = "test4.mp4"

vehicle_model = YOLO("yolov8n.pt")

#COCO vehicle classes
#car = 2, motorcycle = 3, bus = 5, truck = 7
VEHICLE_CLASSES = [2, 3, 5, 7]

PROCESS_EVERY_N_FRAMES = 1

CONF_THRESHOLD = 0.25

MIN_BOX_WIDTH = 40

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

LINE_A_Y = 340
LINE_B_Y = 550

REAL_DISTANCE_METERS = 10

SPEED_LIMIT_KMH = 50

#Video

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError("Could not open video file.")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

print("Video FPS:", fps)
print("Running speed-only both-direction version.")
print("Press Q or ESC to stop.")

#Storage

first_line = {}
first_frame = {}
measured = set()
previous_positions = {}

frame_count = 0
last_results = None

#Loop

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    frame_count += 1


    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    #YOLO

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:

        last_results = vehicle_model.track(
            frame,
            persist=True,
            classes=VEHICLE_CLASSES,
            conf=CONF_THRESHOLD,
            verbose=False
        )

    #Draw lines

    cv2.line(
        frame,
        (0, LINE_A_Y),
        (frame.shape[1], LINE_A_Y),
        (0, 255, 255),
        3
    )

    cv2.line(
        frame,
        (0, LINE_B_Y),
        (frame.shape[1], LINE_B_Y),
        (255, 0, 255),
        3
    )

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

    #Processing detections

    if last_results and last_results[0].boxes is not None:

        for box in last_results[0].boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])

            class_id = int(box.cls[0])
            class_name = vehicle_model.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            box_width = x2 - x1

            # Ignore extremely tiny detections
            if box_width < MIN_BOX_WIDTH:
                continue

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            prev_y = previous_positions.get(track_id, center_y)

            #Speed and direction

            crossed_a = (
                (prev_y < LINE_A_Y and center_y >= LINE_A_Y) or
                (prev_y > LINE_A_Y and center_y <= LINE_A_Y)
            )

            crossed_b = (
                (prev_y < LINE_B_Y and center_y >= LINE_B_Y) or
                (prev_y > LINE_B_Y and center_y <= LINE_B_Y)
            )

            #First line crossed
            if track_id not in first_line:

                if crossed_a:
                    first_line[track_id] = "A"
                    first_frame[track_id] = frame_count

                    print(f"Track {track_id} crossed LINE A first")

                elif crossed_b:
                    first_line[track_id] = "B"
                    first_frame[track_id] = frame_count

                    print(f"Track {track_id} crossed LINE B first")

            # econd crossing
            elif track_id not in measured:

                # A -> B
                if first_line[track_id] == "A" and crossed_b:

                    frame_diff = frame_count - first_frame[track_id]
                    elapsed = frame_diff / fps

                    if elapsed > 0:

                        speed_kmh = (
                            REAL_DISTANCE_METERS / elapsed
                        ) * 3.6

                        status = "OK"

                        if speed_kmh > SPEED_LIMIT_KMH:
                            status = "SPEEDING"

                        print(
                            f"Track {track_id} | "
                            f"Direction A->B | "
                            f"Speed: {speed_kmh:.2f} km/h | "
                            f"{status}"
                        )

                        measured.add(track_id)

                # B -> A
                elif first_line[track_id] == "B" and crossed_a:

                    frame_diff = frame_count - first_frame[track_id]
                    elapsed = frame_diff / fps

                    if elapsed > 0:

                        speed_kmh = (
                            REAL_DISTANCE_METERS / elapsed
                        ) * 3.6

                        status = "OK"

                        if speed_kmh > SPEED_LIMIT_KMH:
                            status = "SPEEDING"

                        print(
                            f"Track {track_id} | "
                            f"Direction B->A | "
                            f"Speed: {speed_kmh:.2f} km/h | "
                            f"{status}"
                        )

                        measured.add(track_id)

            #Save previous position
            previous_positions[track_id] = center_y

            #Draw Vehicle

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            label = f"{class_name} ID:{track_id}"

            if track_id in measured:
                label += " MEASURED"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    #Show Window

    cv2.imshow("Speed Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        print("Stopping...")
        break

#Cleanup

cap.release()
cv2.destroyAllWindows()

print("Program stopped.")