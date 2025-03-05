import os
import cv2
from ultralytics import YOLO
import numpy as np

# Define paths
model_path = r"C:\Users\muska\Documents\SEM 7\BTPF2\cod\runs\segment\train\weights\last.pt"
input_folder = r"C:\Users\muska\Documents\SEM 7\BTPF2\captures\vl"
output_folder = r"C:\Users\muska\Documents\SEM 7\BTPF2\captures\output"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load the pre-trained YOLO model
model = YOLO(model_path)

# Process all images in the input folder
for image_file in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_file)
    
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue

    # Initialize variables for each image
    dimensions = []  # Store width and contours for all detected objects
    calculated_length = None
    calculated_width = None

    # Load the image
    img = cv2.imread(image_path)
    H, W, _ = img.shape

    # Run the model on the image
    results = model(img)

    # First pass: Collect dimensions for all objects
    for result in results:
        for mask in result.masks.data:
            mask = mask.numpy() * 255
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            _, thresh = cv2.threshold(mask.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) > 200:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(cnt)
                    dimensions.append((w, h, cnt))  # Store width, height, and contour

    # Sort objects by width
    dimensions.sort(key=lambda x: x[0])  # Sort by width (ascending)

    # Assign the smallest width as the rectangle, largest as the drop
    if len(dimensions) >= 2:  # Ensure at least two objects exist
        wrect, hrect, rect_cnt = dimensions[0]
        wdrop, ldrop, drop_cnt = dimensions[-1]

        # Calculate drop's dimensions using rectangle width
        if wrect > 0:
            calculated_length = round((ldrop / wrect) * 2, 5)
            calculated_width = round((wdrop / wrect) * 2, 5)

        # Draw rectangle bounding box
        x, y, w, h = cv2.boundingRect(rect_cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, "Rectangle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Draw drop bounding box
        x, y, w, h = cv2.boundingRect(drop_cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Drop", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Annotate calculated results
        if calculated_length is not None and calculated_width is not None:
            cv2.putText(img, f"Length: {calculated_length} mm", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"Width: {calculated_width} mm", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save the final image with bounding boxes and labels
    output_image_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_image_path, img)

    # Print the results for the current image
    print(f"Processed {image_file}:")
    if len(dimensions) >= 2:
        print(f"  Width of rectangle: {wrect}px")
        print(f"  Width of drop: {wdrop}px")
        print(f"  Length of drop: {ldrop}px")
        print(f"  Calculated width: {calculated_width} mm")
        print(f"  Calculated length: {calculated_length} mm")
    else:
        print("  Not enough objects to classify (require at least 2).")
    print(f"  Output saved to: {output_image_path}")
