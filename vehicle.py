from ultralytics import YOLO
import cv2

model = YOLO("vehicle1.pt")

cap = cv2.VideoCapture("video1.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("Vehicle Detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()