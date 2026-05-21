import cv2
from ultralytics import YOLO
import torch
import easyocr

#Settings

VIDEO_SOURCE = "test2.mp4"

VEHICLE_MODEL_PATH = "yolov8n.pt"
PLATE_MODEL_PATH = "license_plate_detector.pt"

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

CONF_VEHICLE = 0.25
CONF_PLATE = 0.30

PROCESS_WIDTH = 1920
PROCESS_HEIGHT = 1080

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

OCR_EVERY_N_FRAMES = 15

DEVICE = 0 if torch.cuda.is_available() else "cpu"

#Loading Models

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

vehicle_model = YOLO(VEHICLE_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)
reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

#Video

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError("Could not open video file.")

print("Running vehicle + plate + OCR only.")
print("Press Q or ESC to stop.")

#Storage

frame_count = 0
last_plate_text = {}

#Helpers

def clean_plate_text(text):
    return (
        text.replace(" ", "")
            .replace("-", "")
            .replace(".", "")
            .replace("_", "")
            .replace(":", "")
            .replace("[", "")
            .replace("]", "")
            .replace("|", "")
            .upper()
    )

#Loop

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    frame_count += 1

    frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    clean_frame = frame.copy()

    vehicle_results = vehicle_model.track(
        frame,
        persist=True,
        classes=VEHICLE_CLASSES,
        conf=CONF_VEHICLE,
        verbose=False,
        device=DEVICE
    )

    if vehicle_results and vehicle_results[0].boxes is not None:

        for box in vehicle_results[0].boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])
            class_id = int(box.cls[0])
            class_name = vehicle_model.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            vehicle_crop = clean_frame[y1:y2, x1:x2]

            if vehicle_crop.size == 0:
                continue

            plate_text = last_plate_text.get(track_id, "")

            #Plate and vehicle

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

                    plate_crop = clean_frame[real_y1:real_y2, real_x1:real_x2]

                    #OCR only every N frames
                    if frame_count % OCR_EVERY_N_FRAMES == 0 and plate_crop.size > 0:
                        ocr_results = reader.readtext(plate_crop)

                        for detection in ocr_results:
                            raw_text = detection[1]
                            confidence = detection[2]

                            cleaned = clean_plate_text(raw_text)

                            if len(cleaned) >= 4 and confidence > 0.25:
                                plate_text = cleaned
                                last_plate_text[track_id] = cleaned
                                print(f"Track {track_id} | Plate: {plate_text}")
                                break

                    #Draw plate
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

            #Draw vehicle

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"{class_name} ID:{track_id}"

            if plate_text:
                label += f" Plate:{plate_text}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("Vehicle + Plate + OCR", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        print("Stopping...")
        break

#Cleanup

cap.release()
cv2.destroyAllWindows()

print("Program stopped.")