import cv2

video = cv2.VideoCapture("test4.mp4")

width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = video.get(cv2.CAP_PROP_FPS)
frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

duration = frames / fps

print("Width:", int(width))
print("Height:", int(height))
print("FPS:", fps)
print("Frames:", int(frames))
print("Duration:", duration, "seconds")

video.release()