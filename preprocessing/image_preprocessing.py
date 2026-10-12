import cv2
import os


def preprocess_image(input_path, output_path):

    # 1. Read the image
    image = cv2.imread(input_path)

    if image is None:
        print("❌ Error: Could not read the image.")
        return

    # 2. Resize the image
    resized = cv2.resize(image, (640, 480))

    # 3. Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 4. Reduce noise
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # 5. Detect edges
    edges = cv2.Canny(denoised, 100, 200)

    # 6. Create output folder
    os.makedirs("outputs", exist_ok=True)

    # 7. Save processed image
    cv2.imwrite(output_path, edges)

    print("✅ Image preprocessing completed!")
    print(f"✅ Processed image saved at: {output_path}")


if __name__ == "__main__":

    input_image = "sample.jpg"
    output_image = "outputs/processed_image.jpg"

    preprocess_image(input_image, output_image)