import cv2
from ultralytics import YOLO
import torch
import easyocr
from config import *
from speed_utils import (
    crossed_line,
    calculate_speed_kmh,
    get_speed_status
)
from plate_utils import clean_plate_text
from log_utils import (
    setup_excel,
    save_speeding_screenshot,
    log_event
)

#GPU

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

#Loading models

vehicle_model = YOLO(VEHICLE_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)

reader = easyocr.Reader(
    ["en"],
    gpu=torch.cuda.is_available()
)

#Excel logging
wb, ws = setup_excel()

#Video

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Running final system.")
print(f"Video: {width}x{height}")
print(f"FPS: {fps}")

#storage

first_line = {}
first_frame = {}

measured = set()

previous_positions = {}

speed_results = {}

last_plate_text = {}

screenshots_taken = 0

frame_count = 0

#loop

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    frame_count += 1

    #resizing

    frame = cv2.resize(
        frame,
        (PROCESS_WIDTH, PROCESS_HEIGHT)
    )

    clean_frame = frame.copy()

    #vehicle tracking

    vehicle_results = vehicle_model.track(
        frame,
        persist=True,
        classes=VEHICLE_CLASSES,
        conf=CONF_VEHICLE,
        verbose=False,
        device=DEVICE
    )

    #lines

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

    #processing

    if vehicle_results and vehicle_results[0].boxes is not None:

        for box in vehicle_results[0].boxes:

            if box.id is None:
                continue

            #vehicle info

            track_id = int(box.id[0])

            class_id = int(box.cls[0])

            class_name = vehicle_model.names[class_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            if (x2 - x1) < MIN_BOX_WIDTH:
                continue

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            prev_y = previous_positions.get(
                track_id,
                center_y
            )

            #line crossing

            crossed_a = crossed_line(
                prev_y,
                center_y,
                LINE_A_Y
            )

            crossed_b = crossed_line(
                prev_y,
                center_y,
                LINE_B_Y
            )

            new_event = None

            #first line

            if track_id not in first_line:

                if crossed_a:
                    first_line[track_id] = "A"
                    first_frame[track_id] = frame_count

                elif crossed_b:
                    first_line[track_id] = "B"
                    first_frame[track_id] = frame_count

            #speed measurement

            elif track_id not in measured:

                direction = None

                if (
                    first_line[track_id] == "A"
                    and crossed_b
                ):
                    direction = "A->B"

                elif (
                    first_line[track_id] == "B"
                    and crossed_a
                ):
                    direction = "B->A"

                if direction is not None:

                    frame_diff = (
                        frame_count
                        - first_frame[track_id]
                    )

                    speed_kmh = calculate_speed_kmh(
                        REAL_DISTANCE_METERS,
                        frame_diff,
                        fps
                    )

                    if speed_kmh is not None:

                        status = get_speed_status(
                            speed_kmh,
                            SPEED_LIMIT_KMH
                        )

                        speed_results[track_id] = {
                            "speed": speed_kmh,
                            "status": status,
                            "direction": direction
                        }

                        new_event = {
                            "track_id": track_id,
                            "direction": direction,
                            "speed_kmh": speed_kmh,
                            "speed_limit_kmh": SPEED_LIMIT_KMH,
                            "status": status,
                            "frame_number": frame_count,
                            "line_a_y": LINE_A_Y,
                            "line_b_y": LINE_B_Y,
                            "distance_meters": REAL_DISTANCE_METERS,
                            "plate_text": "",
                            "screenshot": ""
                        }

                        measured.add(track_id)

            previous_positions[track_id] = center_y

            #vehicle color

            vehicle_color = (0, 255, 0)

            if track_id in speed_results:

                if (
                    speed_results[track_id]["status"]
                    == "SPEEDING"
                ):
                    vehicle_color = (0, 0, 255)

            #draw box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                vehicle_color,
                2
            )

            #plate detection

            vehicle_crop = clean_frame[y1:y2, x1:x2]

            plate_text = last_plate_text.get(
                track_id,
                ""
            )

            if vehicle_crop.size > 0:

                plate_results = plate_model(
                    vehicle_crop,
                    conf=CONF_PLATE,
                    verbose=False,
                    device=DEVICE
                )

                if (
                    plate_results
                    and plate_results[0].boxes is not None
                ):

                    for plate_box in plate_results[0].boxes:

                        px1, py1, px2, py2 = map(
                            int,
                            plate_box.xyxy[0]
                        )

                        real_x1 = x1 + px1
                        real_y1 = y1 + py1
                        real_x2 = x1 + px2
                        real_y2 = y1 + py2

                        #draw plate box

                        cv2.rectangle(
                            frame,
                            (real_x1, real_y1),
                            (real_x2, real_y2),
                            (255, 0, 0),
                            2
                        )

                        #ocr

                        plate_crop = clean_frame[
                            real_y1:real_y2,
                            real_x1:real_x2
                        ]

                        if (
                            frame_count % OCR_EVERY_N_FRAMES == 0
                            and plate_crop.size > 0
                        ):

                            ocr_results = reader.readtext(
                                plate_crop
                            )

                            for detection in ocr_results:

                                raw_text = detection[1]

                                confidence = detection[2]

                                cleaned = clean_plate_text(
                                    raw_text
                                )

                                if (
                                    len(cleaned) >= 4
                                    and confidence > 0.25
                                ):

                                    plate_text = cleaned

                                    last_plate_text[
                                        track_id
                                    ] = cleaned

                                    break

                        #draw ocr text

                        if plate_text:

                            cv2.putText(
                                frame,
                                plate_text,
                                (real_x1, real_y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 0, 0),
                                2
                            )

            #label

            label = f"{class_name} ID:{track_id}"

            if plate_text:
                label += f" Plate:{plate_text}"

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

            #logging

            if new_event is not None:

                new_event["plate_text"] = plate_text

                if (
                    new_event["status"] == "SPEEDING"
                    and screenshots_taken < MAX_SCREENSHOTS
                ):

                    screenshot_path = save_speeding_screenshot(
                        frame,
                        SCREENSHOT_FOLDER,
                        track_id,
                        new_event["speed_kmh"]
                    )

                    new_event["screenshot"] = screenshot_path

                    screenshots_taken += 1

                log_event(ws, new_event)

                print(
                    f"Track {track_id} | "
                    f"Plate:{plate_text} | "
                    f"{new_event['direction']} | "
                    f"{new_event['speed_kmh']:.2f} km/h | "
                    f"{new_event['status']}"
                )

    #display

    display = cv2.resize(
        frame,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    )

    cv2.imshow(
        "Final Speed Camera System",
        display
    )

    #exit

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        print("Stopping...")
        break

#cleanup

cap.release()

cv2.destroyAllWindows()

wb.save(EXCEL_FILE)

print("Program stopped.")
print(f"Excel saved: {EXCEL_FILE}")
print(f"Screenshots saved: {screenshots_taken}")