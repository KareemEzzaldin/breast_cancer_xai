import cv2
import os
import glob

def crop_and_apply_clahe(image_path, output_path):
    # Read the image in grayscale (Medical images are usually grayscale)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read {image_path}")
        return
    
    # 1. Cropping: Remove the useless black background
    _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (which represents the breast)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Crop the image using the bounding box
        cropped_img = img[y:y+h, x:x+w]
    else:
        # Fallback if no contours found
        cropped_img = img

    # 2. CLAHE Application
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed_img = clahe.apply(cropped_img)

    # Save the final image
    cv2.imwrite(output_path, processed_img)
    print(f"Successfully processed and saved: {os.path.basename(output_path)}")

def main():
    input_dir = os.path.join("data", "raw")
    output_dir = os.path.join("data", "processed")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Fetch all common image formats
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.dcm"]:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper()))) # Catch .JPG or .PNG
    
    if not image_paths:
        print("Warning: No images found in 'data/raw/'. Please add images first.")
        return

    print(f"Found {len(image_paths)} images. Starting preprocessing pipeline...")
    
    for img_path in image_paths:
        file_name = os.path.basename(img_path)
        # We will save everything as .png to keep high quality without compression
        out_name = os.path.splitext(file_name)[0] + ".png" 
        out_path = os.path.join(output_dir, out_name)
        crop_and_apply_clahe(img_path, out_path)
        
    print("Preprocessing completed successfully! Check the 'data/processed' folder.")

if __name__ == "__main__":
    main()