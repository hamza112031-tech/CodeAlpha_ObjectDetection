from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8m.pt")

input_path = "test_video3.mp4"
output_path = "output_detected.mp4"

video = cv2.VideoCapture(input_path)
fps_video = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

width, height = 960, 540

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))

frame_index = 0
start_time = time.time()

while True:
    success, frame = video.read()

    if not success:
        break

    frame = cv2.resize(frame, (width, height))

    results = model.track(frame, persist=True, verbose=False)
    annotated_frame = results[0].plot()
    num_objects = len(results[0].boxes)

    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 50), (20, 20, 20), -1)
    annotated_frame = cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0)

    cv2.putText(annotated_frame, f"Objects Detected: {num_objects}", (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

    cv2.putText(annotated_frame, "AI Object Detection & Tracking", (500, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    writer.write(annotated_frame)

    frame_index += 1
    print(f"Processing frame {frame_index}/{total_frames}", end="\r")

video.release()
writer.release()

elapsed = time.time() - start_time
print(f"\nDone! Processed {frame_index} frames in {elapsed:.1f} seconds.")
print(f"Output saved to: {output_path}")