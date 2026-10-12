import matplotlib.pyplot as plt
from pathlib import Path

from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT
    / "runs"
    / "classify"
    / "models"
    / "accident_classifier"
    / "weights"
    / "best.pt"
)

TEST_DIR = ROOT / "dataset" / "test"

# Load the trained model
model =YOLO(str(MODEL_PATH))

class_names = model.names

y_true = []
y_pred = []

valid_extensions = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

print("Evaluating test images...")

# Read each test image and its actual class
for class_folder in sorted(TEST_DIR.iterdir()):
    if not class_folder.is_dir():
        continue

    true_class = class_folder.name

    if true_class not in class_names.values():
        continue

    image_paths = [
        p for p in class_folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in valid_extensions
    ]

    for image_path in image_paths:
        result = model.predict(
            str(image_path),
            verbose=False
        )[0]

        predicted_index = int(result.probs.top1)
        predicted_class = class_names[predicted_index]

        y_true.append(true_class)
        y_pred.append(predicted_class)

# Report results
labels = ["accident", "no_accident"]

if not y_true:
    raise RuntimeError(
        f"No test images found under {TEST_DIR}"
    )

print("\n========== MODEL EVALUATION ==========")
print(f"Images evaluated: {len(y_true)}")
print(
    f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        digits=4
    )
)

print("Confusion Matrix")
print("Rows = actual; columns = predicted")
print("Class order:", labels)
print(confusion_matrix(y_true, y_pred, labels=labels))
print(confusion_matrix(y_true, y_pred, labels=labels))


# Generate and save the confusion matrix image
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 6))

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=labels,
    display_labels=labels,
    values_format="d",
    cmap="Blues",
    ax=ax
)

ax.set_title("AI Accident Detection - Confusion Matrix")
plt.tight_layout()

matrix_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(matrix_path, dpi=300, bbox_inches="tight")
plt.close()

# Save the evaluation metrics as a text report
report_path = OUTPUT_DIR / "model_evaluation_report.txt"

with open(report_path, "w", encoding="utf-8") as report_file:
    report_file.write("AI ACCIDENT DETECTION - MODEL EVALUATION\n")
    report_file.write("=" * 45 + "\n")
    report_file.write(f"Test images evaluated: {len(y_true)}\n")
    report_file.write(
        f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%\n\n"
    )
    report_file.write("CLASSIFICATION REPORT\n")
    report_file.write(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
            digits=4
        )
    )
    report_file.write("\n\nCONFUSION MATRIX\n")
    report_file.write("Rows = actual; columns = predicted\n")
    report_file.write(str(confusion_matrix(y_true, y_pred, labels=labels)))
    report_file.write("\n")

print(f"\nConfusion matrix saved to: {matrix_path}")
print(f"Evaluation report saved to: {report_path}")
