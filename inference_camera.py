import cv2
from ultralytics import YOLO

print("Loading model...")
model = YOLO('simplist_model.pt') 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: impossible to load model.")
    exit()

print("Ready. Press 'q' to leave.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detection
    results = model(frame, conf=0.5, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow('Drone vision - YOLOv8', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()