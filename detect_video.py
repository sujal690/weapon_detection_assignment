import cv2
import os
from ultralytics import YOLO

def detect_weapons_in_video(model_path: str, input_video_path: str, output_dir: str):
    print(f"Loading model from {model_path} ...")
    model = YOLO(model_path)
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare output video path
    output_video_path = os.path.join(output_dir, f"detected_{os.path.basename(input_video_path)}")
    
    # Define video codec and writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print("Starting video detection...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection on frame (pass frame directly)
        results = model(frame, verbose=False)
        
        # Annotate frame with detections
        annotated_frame = results[0].plot()
        
        # Write annotated frame to output video
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
    
    # Release resources
    cap.release()
    out.release()
    print(f"Detection finished. Output saved to: {output_video_path}")

if __name__ == "__main__":
    # Update these paths before running
    model_path = "./yolov8n_results2/weights/best.pt"  # Change this to your model path
    input_video_path = "./video_20250924_220118_edit.mp4"  # Change this to your uploaded video path
    output_dir = "./video_output"
    
    detect_weapons_in_video(model_path, input_video_path, output_dir)
