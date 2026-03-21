# test_ambulance_detector_high_confidence.py
import cv2
from ultralytics import YOLO

def test_model_on_video():
    """
    Loads a custom-trained YOLO model and filters for high-confidence
    'ambulance' detections.
    """
    model_path = 'ambulance.pt'
    model = YOLO(model_path)

    # Find the class index for 'ambulance'
    ambulance_class_index = -1
    for key, value in model.names.items():
        print(key,value)
        if value == 'Ambulance':
            ambulance_class_index = key
            break
            
    if ambulance_class_index == -1:
        print("❌ Error: 'ambulance' class not found in model.")
        return
        
    print(f"✅ Found 'ambulance' at index: {ambulance_class_index}. Filtering for high confidence.")

    video_path = 'video.mp4' 
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- KEY CHANGE: Increased confidence threshold from 0.5 to 0.75 ---
        # This makes the model "stricter" about what it calls an ambulance.
        results = model.predict(
            frame, 
            classes=[ambulance_class_index], 
            conf=0.85, # <-- INCREASED VALUE
            verbose=False
        )

        annotated_frame = results[0].plot()
        
        if len(results[0].boxes) > 0:
            print("High-Confidence Ambulance Detected!")

        cv2.imshow("Real-Time Ambulance Detection (High Confidence)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_model_on_video()