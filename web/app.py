import os
import sys
import shutil

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# FLASK IMPORTS
# ============================================================

from flask import (
    Flask,
    render_template,
    request,
    url_for,
    send_from_directory
)

from werkzeug.utils import secure_filename

from detection.integrated_accident_system import analyze_accident


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

UPLOAD_FOLDER = os.path.join(
    PROJECT_ROOT,
    "uploads"
)

RESULT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "web",
    "static",
    "results"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "outputs"
)


CONFUSION_MATRIX_FOLDER = os.path.join(
    PROJECT_ROOT,
    "web",
    "static",
    "results"
)



# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# HOME PAGE
# ============================================================


@app.route("/")
def home():
    return render_template(
        "index.html",
        model_accuracy=98.12,
        accident_precision=0.9796,
        accident_recall=0.9828,
        accident_f1=0.9812,
        confusion_matrix_url=url_for(
            "static",
            filename="results/confusion_matrix.png"
        ),
        evaluation_report_url=url_for(
            "static",
            filename="results/model_evaluation_report.txt"
        )
    )



# ============================================================
# RESULT IMAGE ROUTE
# ============================================================

@app.route("/results/<path:filename>")
def result_file(filename):

    return send_from_directory(
        RESULT_FOLDER,
        filename
    )


# ============================================================
# CONFUSION MATRIX ROUTE
# ============================================================

@app.route("/model/confusion-matrix")
def model_confusion_matrix():

    confusion_matrix_path = os.path.join(
        CONFUSION_MATRIX_FOLDER,
        "confusion_matrix.png"
    )

    # Check whether confusion matrix exists
    if not os.path.exists(
        confusion_matrix_path
    ):

        return (
            "Confusion matrix not found. "
            "Please train/evaluate the model first.",
            404
        )

    return send_from_directory(
        CONFUSION_MATRIX_FOLDER,
        "confusion_matrix.png"
    )


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

@app.route("/model/confusion-matrix-normalized")
def model_confusion_matrix_normalized():

    normalized_path = os.path.join(
        CONFUSION_MATRIX_FOLDER,
        "confusion_matrix_normalized.png"
    )

    if not os.path.exists(
        normalized_path
    ):

        return (
            "Normalized confusion matrix not found.",
            404
        )

    return send_from_directory(
        CONFUSION_MATRIX_FOLDER,
        "confusion_matrix_normalized.png"
    )


# ============================================================
# ANALYZE IMAGE
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # ========================================================
    # CHECK FILE
    # ========================================================

    if "image" not in request.files:

        return render_template(
            "index.html",
            error="Please select an image."
        )


    file = request.files["image"]


    if file.filename == "":

        return render_template(
            "index.html",
            error="No image selected."
        )


    # ========================================================
    # SECURE FILE NAME
    # ========================================================

    filename = secure_filename(
        file.filename
    )


    # Prevent empty filename
    if not filename:

        return render_template(
            "index.html",
            error="Invalid image filename."
        )


    # ========================================================
    # SAVE UPLOADED IMAGE
    # ========================================================

    input_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    try:

        file.save(
            input_path
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"Could not save uploaded image: {str(e)}"
        )


    # ========================================================
    # RUN COMPLETE AI PIPELINE
    # ========================================================

    try:

        results = analyze_accident(
            input_path
        )

    except Exception as e:

        print()
        print("==========================================")
        print("AI ANALYSIS ERROR")
        print("==========================================")
        print(str(e))
        print("==========================================")
        print()

        return render_template(
            "index.html",
            error=f"AI analysis failed: {str(e)}"
        )


    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if not results:

        return render_template(
            "index.html",
            error="AI analysis returned no results."
        )


    # ========================================================
    # GET OUTPUT IMAGE
    # ========================================================

    source_image = results.get(
        "output_image"
    )


    if not source_image:

        return render_template(
            "index.html",
            error=(
                "AI analysis completed, "
                "but no output image was generated."
            )
        )


    # ========================================================
    # CHECK OUTPUT IMAGE
    # ========================================================

    if not os.path.exists(
        source_image
    ):

        return render_template(
            "index.html",
            error=(
                "Output image not found: "
                f"{source_image}"
            )
        )


    # ========================================================
    # COPY OUTPUT IMAGE TO FLASK STATIC DIRECTORY
    # ========================================================

    result_filename = (
        "integrated_accident_analysis.jpg"
    )


    destination_image = os.path.join(
        RESULT_FOLDER,
        result_filename
    )


    try:

        shutil.copy2(
            source_image,
            destination_image
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=(
                "Could not prepare result image: "
                f"{str(e)}"
            )
        )


    # ========================================================
    # CREATE BROWSER IMAGE URL
    # ========================================================

    image_url = url_for(
        "result_file",
        filename=result_filename
    )


    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    print()
    print("==========================================")
    print("AI ACCIDENT DETECTION")
    print("==========================================")

    print(
        "Uploaded file:",
        filename
    )

    print(
        "Input image:",
        input_path
    )

    print(
        "AI output:",
        source_image
    )

    print(
        "Browser image URL:",
        image_url
    )

    print(
        "Result image:",
        destination_image
    )

    print("==========================================")
    print()


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    return render_template(
        "index.html",
        results=results,
        image_url=image_url,
        uploaded_filename=filename,

    model_accuracy=98.12,
    accident_precision=0.9796,
    accident_recall=0.9828,
    accident_f1=0.9812,

    confusion_matrix_url=url_for(
        "static",
        filename="results/confusion_matrix.png"
    ),
    evaluation_report_url=url_for(
        "static",
        filename="results/model_evaluation_report.txt"
    )
)


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print("🚨 AI ACCIDENT DETECTION SYSTEM")
    print("==========================================")
    print("Server starting...")
    print()
    print("Open:")
    print("http://127.0.0.1:5000")
    print("==========================================")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )