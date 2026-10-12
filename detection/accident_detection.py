from ultralytics import YOLO
import cv2
import os

# Load YOLO model
model = YOLO("yolo11n.pt")

# Input image
input_image = "sample.jpg"

# Run YOLO detection
results = model(input_image)

result = results[0]

# Store vehicle detections
vehicles = []

for box in result.boxes:

    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]

    # We only consider vehicles
    if class_name in ["car", "motorcycle", "bus", "truck"]:

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        vehicles.append({
            "class": class_name,
            "confidence": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })


print("\n🚗 VEHICLE ANALYSIS")
print("--------------------------")

print(f"Vehicles detected: {len(vehicles)}")

for vehicle in vehicles:

    print(
        f"{vehicle['class']} "
        f"- {vehicle['confidence'] * 100:.2f}% confidence"
    )


# Check whether two vehicle bounding boxes overlap
possible_collision = False

for i in range(len(vehicles)):

    for j in range(i + 1, len(vehicles)):

        a = vehicles[i]
        b = vehicles[j]

        # Calculate intersection
        x_left = max(a["x1"], b["x1"])
        y_top = max(a["y1"], b["y1"])
        x_right = min(a["x2"], b["x2"])
        y_bottom = min(a["y2"], b["y2"])

        if x_right > x_left and y_bottom > y_top:

            possible_collision = True

            print("\n⚠️ WARNING!")
            print(
                f"Possible collision between "
                f"{a['class']} and {b['class']}"
            )


print("\n==========================")

if possible_collision:

    print("🚨 POSSIBLE ACCIDENT DETECTED")

else:

    print("✅ NO POSSIBLE COLLISION DETECTED")