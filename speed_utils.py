def crossed_line(prev_y, center_y, line_y):
    return (
        (prev_y < line_y and center_y >= line_y) or
        (prev_y > line_y and center_y <= line_y)
    )


def calculate_speed_kmh(distance_meters, frame_diff, fps):
    elapsed = frame_diff / fps

    if elapsed <= 0:
        return None

    return (distance_meters / elapsed) * 3.6


def get_speed_status(speed_kmh, speed_limit_kmh):
    if speed_kmh > speed_limit_kmh:
        return "SPEEDING"
    return "OK"