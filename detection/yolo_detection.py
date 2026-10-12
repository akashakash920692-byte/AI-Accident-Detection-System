from ultralytics import YOLO
import cv2
import os

# Load YOLO model
model = YOLO("yolo11n.pt")

# Input image
input_image = "sample.jpg"

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Run object detection
results = model(input_image)

# Get the first result
result = results[0]

# Draw detections on the image
annotated_image = result.plot()

# Save output
output_path = "outputs/yolo_detection.jpg"
cv2.imwrite(output_path, annotated_image)

print("✅ YOLO detection completed!")
print(f"✅ Detection result saved at: {output_path}")

# Display detected objects
print("\nDetected objects:")

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]

    print(
        f"• {class_name} - "
        f"{confidence * 100:.2f}% confidence"
    )