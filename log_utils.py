import os
import cv2
from openpyxl import Workbook


def setup_excel():
    wb = Workbook()
    ws = wb.active
    ws.title = "Speed Log"

    ws.append([
        "track_id",
        "plate_text",
        "direction",
        "speed_kmh",
        "speed_limit_kmh",
        "status",
        "frame_number",
        "line_a_y",
        "line_b_y",
        "distance_meters",
        "screenshot"
    ])

    return wb, ws


def save_speeding_screenshot(frame, folder, track_id, speed_kmh):
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(
        folder,
        f"speeding_id{track_id}_{speed_kmh:.1f}kmh.jpg"
    )

    cv2.imwrite(path, frame)

    return path


def log_event(ws, event):
    ws.append([
        event["track_id"],
        event["plate_text"],
        event["direction"],
        round(event["speed_kmh"], 2),
        event["speed_limit_kmh"],
        event["status"],
        event["frame_number"],
        event["line_a_y"],
        event["line_b_y"],
        event["distance_meters"],
        event["screenshot"]
    ])