
# AI-Based Road Accident Detection & Emergency Report Generation System

An AI-powered computer vision prototype that analyzes road-scene images, classifies scenes as accident or no accident, detects vehicles and people, and generates an automated emergency incident report.

## Project Overview

Road accidents require timely identification and accurate reporting. This project combines machine learning, digital image processing, computer vision, and natural language processing to demonstrate an automated road-scene analysis workflow.

Users upload an image through a web dashboard and receive a classification result, detected-object information, an estimated severity level, and a generated incident report.

**Note:** This is an academic prototype. It does not automatically contact emergency services or provide verified injury assessments.

## Features

- Accident vs. no-accident image classification
- YOLO-based vehicle and person detection
- Digital image preprocessing
- Automated emergency incident report generation
- Estimated incident severity based on programmed heuristics
- Flask web dashboard for image uploads and analysis
- Model evaluation metrics and confusion matrix
- Downloadable text report and PDF export

## Technologies Used

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Flask
- HTML, CSS, and JavaScript

## Project Architecture

1. **Image Upload:** The user uploads a road-scene image through the web interface.
2. **Image Preprocessing:** Image-processing operations prepare and analyze the image.
3. **Accident Classification:** A trained image classifier predicts accident or no accident.
4. **Object Detection:** YOLO identifies objects such as cars, trucks, motorcycles, and people.
5. **Severity Estimation:** Programmed heuristics produce an estimated severity category.
6. **Report Generation:** An NLP-based reporting module generates an incident summary.
7. **Results Dashboard:** Flask displays the analysis, annotated image, evaluation metrics, and report.

## Model Evaluation

The current test evaluation reported:

| Metric | Result |
|---|---:|
| Test images | 1,858 |
| Top-1 accuracy | 98.12% |
| Correct predictions | 1,823 |
| Incorrect predictions | 35 |

The confusion matrix from the current evaluation should be included in the `outputs/` directory.

These results apply to the available test dataset and do not guarantee the same performance on unseen real-world road scenes.

## Project Structure

```text
AI_Accident_Detection/
├── dataset/
├── detection/
│   ├── accident_detection.py
│   ├── yolo_detection.py
│   └── integrated_accident_system.py
├── nlp/
│   └── emergency_report.py
├── preprocessing/
│   └── image_preprocessing.py
├── outputs/
│   ├── confusion_matrix.png
│   └── model_evaluation_report.txt
├── web/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
├── evaluate_model.py
└── README.md
```

Adjust the structure above if your local files differ.

## Requirements

- Python compatible with the installed project dependencies
- pip
- VS Code or another Python editor
- A Mac, Windows, or Linux computer

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI_Accident_Detection
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the environment

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

If a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

### 5. Prepare the trained model and dataset

Place the trained classification weights and required datasets in their configured paths. Check the project configuration before running the application.

### 6. Start the website

From the project root:

```bash
./venv/bin/python web/app.py
```

Open the local URL printed by Flask, commonly `http://127.0.0.1:5000`.

## Usage

1. Start the Flask application.
2. Open the dashboard in your browser.
3. Upload a road-scene image.
4. Run the analysis.
5. Review the classification and detected objects.
6. Read the generated incident report.
7. Download the report as a text file or export it to PDF.

## Outputs

The system can produce:

- An accident classification
- A classification confidence score
- Detected road objects
- An annotated image
- An estimated severity category
- An automated incident report
- Model evaluation metrics and a confusion matrix

## Limitations

- Performance depends on the quality and representativeness of the dataset.
- Model confidence is not a guarantee that a prediction is correct.
- Object counts do not establish whether anyone is injured.
- Severity estimation is heuristic-based and requires further validation.
- Image analysis alone may not determine the exact incident location or time.
- The prototype does not dispatch emergency services.

## Future Improvements

- Evaluate performance on more diverse real-world datasets.
- Improve accident detection under different lighting and weather conditions.
- Add video-based accident detection.
- Improve severity estimation using validated data.
- Add secure incident storage and authorized reporting workflows.
- Optimize inference speed and deployment.

## Academic Purpose

Developed as a Bachelor of Computer Applications project to demonstrate the integration of machine learning, computer vision, digital image processing, natural language processing, and web development.

## License

Choose an appropriate open-source license before distributing the project publicly.

## Trained Model

The trained accident classification model is hosted on Hugging Face.

- **Model repository:** [Accident Classifier](https://huggingface.co/Akashuuuu/accident-classifier)
- **Model file:** `best.pt`
- **Task:** Binary image classification — accident vs. no accident
- **Test accuracy:** 98.12% on 1,858 held-out test images

The system is an academic prototype. Predictions should be verified by a human and must not replace professional emergency assessment.
