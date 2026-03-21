# test_vehicles.py
import cv2
from ultralytics import YOLO

# --- SETUP ---
# 1. Path to your custom-trained model weights
#    Update this to the path of your 'best.pt' file.
MODEL_PATH = '/Users/girivasanth/Documents/Project/rtdetr-l.pt'

# 2. Video source
#    Update this to the path of your test video.
#    Or, use 0 for your computer's webcam.
VIDEO_SOURCE = 'video.mp4' 
# --- END SETUP ---

def run_detector():
    """
    Loads a custom YOLOv8 or RT-DETR model and runs it on a video source.
    """
    # Load the trained model
    # Note: The YOLO() class can load both YOLOv8 and RT-DETR models.
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Open the video source
    print(f"Opening video source: {VIDEO_SOURCE}")
    try:
        cap = cv2.VideoCapture(VIDEO_SOURCE)
    except Exception as e:
        print(f"❌ Error opening video source: {e}")
        return
        
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source.")
        return

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run inference on the frame
            # The model returns a list of result objects
            results = model.predict(frame, verbose=False)

            # Visualize the results on the frame
            # The .plot() method automatically draws bounding boxes and labels
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("Vehicle Detection", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting...")
                break
        else:
            # Break the loop if the end of the video is reached
            print("🏁 End of video reached.")
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Video processing finished.")


if __name__ == "__main__":
    run_detector()