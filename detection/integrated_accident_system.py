import os
import sys
import cv2
import inspect

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# AI / COMPUTER VISION IMPORTS
# ============================================================

from ultralytics import YOLO


# ============================================================
# NLP IMPORT
# ============================================================

try:
    from nlp.emergency_report import generate_emergency_report
except ImportError:
    generate_emergency_report = None


# ============================================================
# MODEL PATHS
# ============================================================

OBJECT_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "yolo11n.pt"
)

ACCIDENT_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "runs",
    "classify",
    "models",
    "accident_classifier",
    "weights",
    "best.pt"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading AI models...")

object_model = YOLO(
    OBJECT_MODEL_PATH
)

accident_model = YOLO(
    ACCIDENT_MODEL_PATH
)

print("✅ YOLO object detection model loaded")
print("✅ Accident classification model loaded")


# ============================================================
# DIGITAL IMAGE PROCESSING
# ============================================================

def perform_dip(input_image):
    """
    Performs basic Digital Image Processing:
    1. Resize
    2. Grayscale
    3. Noise reduction
    4. Edge detection

    The processed image is saved for project demonstration.
    The original image is still used for ML classification
    and YOLO detection because the classifier was trained
    using normal RGB images.
    """

    image = cv2.imread(input_image)

    if image is None:
        raise ValueError(
            "Could not read the uploaded image."
        )

    # Resize
    resized = cv2.resize(
        image,
        (640, 480)
    )

    # Grayscale
    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    # Noise reduction
    denoised = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Edge detection
    edges = cv2.Canny(
        denoised,
        100,
        200
    )

    processed_path = os.path.join(
        OUTPUT_FOLDER,
        "processed_image.jpg"
    )

    cv2.imwrite(
        processed_path,
        edges
    )

    print("✅ DIP processing completed")
    print(
        f"Processed image: {processed_path}"
    )

    return processed_path


# ============================================================
# OBJECT DETECTION
# ============================================================

def perform_object_detection(input_image):

    results = object_model(
        input_image
    )

    result = results[0]

    # --------------------------------------------------------
    # Object counters
    # --------------------------------------------------------

    object_counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
        "person": 0
    }

    # --------------------------------------------------------
    # Detect objects
    # --------------------------------------------------------

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        class_name = object_model.names[
            class_id
        ]

        confidence = float(
            box.conf[0]
        )

        # Ignore very weak detections
        if confidence < 0.25:
            continue

        if class_name in object_counts:

            object_counts[
                class_name
            ] += 1

    # --------------------------------------------------------
    # Create annotated image
    # --------------------------------------------------------

    annotated_image = result.plot()

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "integrated_accident_analysis.jpg"
    )

    cv2.imwrite(
        output_path,
        annotated_image
    )

    print("✅ YOLO object detection completed")

    return object_counts, output_path


# ============================================================
# COLLISION HEURISTIC
# ============================================================

def detect_possible_collision(input_image):

    """
    Simple geometric collision heuristic.

    IMPORTANT:
    This is supporting evidence only.
    The trained accident classifier remains the primary
    accident decision.
    """

    results = object_model(
        input_image
    )

    result = results[0]

    vehicles = []

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        class_name = object_model.names[
            class_id
        ]

        confidence = float(
            box.conf[0]
        )

        if confidence < 0.25:
            continue

        if class_name not in [
            "car",
            "motorcycle",
            "bus",
            "truck"
        ]:
            continue

        coordinates = box.xyxy[0].tolist()

        vehicles.append({
            "class": class_name,
            "confidence": confidence,
            "x1": coordinates[0],
            "y1": coordinates[1],
            "x2": coordinates[2],
            "y2": coordinates[3]
        })

    collision_pairs = []

    # --------------------------------------------------------
    # Compare vehicle bounding boxes
    # --------------------------------------------------------

    for i in range(
        len(vehicles)
    ):

        for j in range(
            i + 1,
            len(vehicles)
        ):

            a = vehicles[i]
            b = vehicles[j]

            x_left = max(
                a["x1"],
                b["x1"]
            )

            y_top = max(
                a["y1"],
                b["y1"]
            )

            x_right = min(
                a["x2"],
                b["x2"]
            )

            y_bottom = min(
                a["y2"],
                b["y2"]
            )

            if (
                x_right > x_left
                and
                y_bottom > y_top
            ):

                collision_pairs.append(
                    f"{a['class']} + {b['class']}"
                )

    return collision_pairs


# ============================================================
# ACCIDENT CLASSIFICATION
# ============================================================

def classify_accident(input_image):

    results = accident_model(
        input_image
    )

    result = results[0]

    # --------------------------------------------------------
    # Get predicted class
    # --------------------------------------------------------

    top_class_index = int(
        result.probs.top1
    )

    predicted_class = result.names[
        top_class_index
    ]

    # --------------------------------------------------------
    # Get confidence
    # --------------------------------------------------------

    accident_confidence = float(
        result.probs.top1conf
    ) * 100

    print()
    print(
        "=========================================="
    )
    print(
        "AI ACCIDENT CLASSIFICATION"
    )
    print(
        "=========================================="
    )

    print(
        f"Predicted Class : {predicted_class}"
    )

    print(
        f"Confidence      : "
        f"{accident_confidence:.2f}%"
    )

    print(
        "=========================================="
    )
    print()

    return (
        predicted_class,
        accident_confidence
    )


# ============================================================
# SEVERITY ASSESSMENT
# ============================================================

def determine_severity(
    predicted_class,
    object_counts
):

    # --------------------------------------------------------
    # No accident
    # --------------------------------------------------------

    if predicted_class != "accident":

        return "NONE"

    # --------------------------------------------------------
    # Count vehicles
    # --------------------------------------------------------

    total_vehicles = (
        object_counts["car"]
        +
        object_counts["motorcycle"]
        +
        object_counts["bus"]
        +
        object_counts["truck"]
    )

    people = object_counts[
        "person"
    ]

    # --------------------------------------------------------
    # Severity rules
    # --------------------------------------------------------

    if (
        total_vehicles >= 5
        or
        people >= 5
    ):

        return "HIGH"

    elif (
        total_vehicles >= 2
        or
        people >= 2
    ):

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================
# NLP EMERGENCY REPORT
# ============================================================

def create_emergency_report(
    predicted_class,
    confidence,
    object_counts,
    severity
):

    # --------------------------------------------------------
    # Try existing NLP module
    # --------------------------------------------------------

    if generate_emergency_report:

        try:

            # Inspect the existing NLP function
            signature = inspect.signature(
                generate_emergency_report
            )

            parameters = signature.parameters

            # ------------------------------------------------
            # Build values based on common parameter names
            # ------------------------------------------------

            values = {}

            for name in parameters:

                lower_name = name.lower()

                if lower_name in [
                    "prediction",
                    "predicted_class",
                    "classification",
                    "incident_status",
                    "status"
                ]:

                    values[name] = predicted_class

                elif lower_name in [
                    "confidence",
                    "accident_confidence"
                ]:

                    values[name] = confidence

                elif lower_name in [
                    "object_counts",
                    "objects",
                    "detections"
                ]:

                    values[name] = object_counts

                elif lower_name == "cars":

                    values[name] = object_counts[
                        "car"
                    ]

                elif lower_name in [
                    "motorcycles",
                    "motorcycle"
                ]:

                    values[name] = object_counts[
                        "motorcycle"
                    ]

                elif lower_name == "buses":

                    values[name] = object_counts[
                        "bus"
                    ]

                elif lower_name == "trucks":

                    values[name] = object_counts[
                        "truck"
                    ]

                elif lower_name in [
                    "people",
                    "persons",
                    "person"
                ]:

                    values[name] = object_counts[
                        "person"
                    ]

                elif lower_name == "severity":

                    values[name] = severity

            # ------------------------------------------------
            # Call NLP generator
            # ------------------------------------------------

            if values:

                report = generate_emergency_report(
                    **values
                )

                if report:

                    return str(report)

        except Exception as error:

            print(
                "⚠️ NLP module fallback:"
                f" {error}"
            )

    # --------------------------------------------------------
    # Fallback NLP-style report
    # --------------------------------------------------------

    status = (
        "ACCIDENT DETECTED"
        if predicted_class == "accident"
        else
        "NO ACCIDENT DETECTED"
    )

    if predicted_class == "accident":

        action = (
            "Emergency response is recommended. "
            "The incident should be reviewed immediately "
            "by the responsible authority."
        )

    else:

        action = (
            "No immediate accident response is required "
            "based on the current AI classification."
        )

    report = f"""
========================================
AI EMERGENCY INCIDENT REPORT
========================================

INCIDENT STATUS
{status}

AI CONFIDENCE
{confidence:.2f}%

SCENE ANALYSIS
The AI system analyzed the uploaded road scene
using Digital Image Processing, Machine Learning,
Neural Network classification and Computer Vision.

The system detected:

Cars: {object_counts['car']}
Motorcycles: {object_counts['motorcycle']}
Buses: {object_counts['bus']}
Trucks: {object_counts['truck']}
People: {object_counts['person']}

TOTAL VEHICLES
{object_counts['car'] + object_counts['motorcycle'] + object_counts['bus'] + object_counts['truck']}

SEVERITY ASSESSMENT
{severity}

AI ASSESSMENT
The computer vision system classified the scene
as "{predicted_class.upper()}".

RECOMMENDED ACTION
{action}

========================================
Generated automatically by the
AI Accident Detection System
========================================
"""

    return report.strip()


# ============================================================
# MAIN AI ANALYSIS FUNCTION
# ============================================================

def analyze_accident(input_image):

    print()
    print(
        "##########################################"
    )
    print(
        "     AI ACCIDENT DETECTION SYSTEM"
    )
    print(
        "##########################################"
    )

    # ========================================================
    # STEP 1 — CHECK IMAGE
    # ========================================================

    if not os.path.exists(
        input_image
    ):

        raise FileNotFoundError(
            f"Input image not found: {input_image}"
        )

    print()
    print(
        "STEP 1: IMAGE RECEIVED"
    )

    print(
        f"Input: {input_image}"
    )


    # ========================================================
    # STEP 2 — DIP
    # ========================================================

    print()
    print(
        "STEP 2: DIGITAL IMAGE PROCESSING"
    )

    processed_image = perform_dip(
        input_image
    )


    # ========================================================
    # STEP 3 — ML / NN ACCIDENT CLASSIFICATION
    # ========================================================

    print()
    print(
        "STEP 3: ACCIDENT CLASSIFICATION"
    )

    (
        predicted_class,
        accident_confidence
    ) = classify_accident(
        input_image
    )


    # ========================================================
    # STEP 4 — YOLO OBJECT DETECTION
    # ========================================================

    print()
    print(
        "STEP 4: YOLO OBJECT DETECTION"
    )

    (
        object_counts,
        output_image
    ) = perform_object_detection(
        input_image
    )


    # ========================================================
    # STEP 5 — COLLISION ANALYSIS
    # ========================================================

    print()
    print(
        "STEP 5: VEHICLE INTERACTION ANALYSIS"
    )

    collision_pairs = detect_possible_collision(
        input_image
    )

    if collision_pairs:

        print(
            "Possible vehicle overlap detected:"
        )

        for pair in collision_pairs:

            print(
                f"⚠️ {pair}"
            )

    else:

        print(
            "No significant vehicle overlap detected."
        )


    # ========================================================
    # STEP 6 — SEVERITY
    # ========================================================

    print()
    print(
        "STEP 6: SEVERITY ASSESSMENT"
    )

    severity = determine_severity(
        predicted_class,
        object_counts
    )

    print(
        f"Severity: {severity}"
    )


    # ========================================================
    # STEP 7 — NLP REPORT
    # ========================================================

    print()
    print(
        "STEP 7: NLP EMERGENCY REPORT"
    )

    emergency_report = create_emergency_report(
        predicted_class,
        accident_confidence,
        object_counts,
        severity
    )

    print(
        "✅ Emergency report generated"
    )


    # ========================================================
    # TOTAL VEHICLES
    # ========================================================

    total_vehicles = (
        object_counts["car"]
        +
        object_counts["motorcycle"]
        +
        object_counts["bus"]
        +
        object_counts["truck"]
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    results = {

        # --------------------------------------------
        # AI CLASSIFICATION
        # --------------------------------------------

        "prediction":
            predicted_class,

        "predicted_class":
            predicted_class,

        "confidence":
            accident_confidence,

        "accident_confidence":
            accident_confidence,


        # --------------------------------------------
        # OBJECT DETECTION
        # --------------------------------------------

        "cars":
            object_counts["car"],

        "motorcycles":
            object_counts["motorcycle"],

        "buses":
            object_counts["bus"],

        "trucks":
            object_counts["truck"],

        "people":
            object_counts["person"],

        "total_vehicles":
            total_vehicles,


        # --------------------------------------------
        # ANALYSIS
        # --------------------------------------------

        "severity":
            severity,

        "possible_collision":
            len(collision_pairs) > 0,

        "collision_pairs":
            collision_pairs,


        # --------------------------------------------
        # IMAGES
        # --------------------------------------------

        "output_image":
            output_image,

        "processed_image":
            processed_image,


        # --------------------------------------------
        # NLP
        # --------------------------------------------

        "report":
            emergency_report,

        "emergency_report":
            emergency_report
    }


    # ========================================================
    # FINAL CONSOLE OUTPUT
    # ========================================================

    print()
    print(
        "##########################################"
    )

    if predicted_class == "accident":

        print(
            "🚨 ACCIDENT DETECTED"
        )

    else:

        print(
            "✅ NO ACCIDENT DETECTED"
        )

    print(
        "##########################################"
    )

    print(
        f"Prediction : "
        f"{predicted_class.upper()}"
    )

    print(
        f"Confidence : "
        f"{accident_confidence:.2f}%"
    )

    print(
        f"Cars       : "
        f"{object_counts['car']}"
    )

    print(
        f"Motorcycles: "
        f"{object_counts['motorcycle']}"
    )

    print(
        f"Buses      : "
        f"{object_counts['bus']}"
    )

    print(
        f"Trucks     : "
        f"{object_counts['truck']}"
    )

    print(
        f"People     : "
        f"{object_counts['person']}"
    )

    print(
        f"Total Vehicles: "
        f"{total_vehicles}"
    )

    print(
        f"Severity   : "
        f"{severity}"
    )

    print(
        f"Output     : "
        f"{output_image}"
    )

    print(
        "##########################################"
    )

    return results


# ============================================================
# TEST THE SYSTEM DIRECTLY
# ============================================================

if __name__ == "__main__":

    test_image = os.path.join(
        PROJECT_ROOT,
        "sample.jpg"
    )

    results = analyze_accident(
        test_image
    )

    print()
    print(
        "FINAL AI RESULT:"
    )

    print(
        results["prediction"]
    )

    print(
        f"{results['confidence']:.2f}%"
    )