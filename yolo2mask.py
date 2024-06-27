import os
import cv2
import numpy as np

def generate_masks_for_experiment(exp_number):
    # Base directory where the prediction results are stored
    base_dir = './yolov5/runs/predict-seg'
    
    # Experiment directory
    if exp_number=='1':
        exp_dir = os.path.join(base_dir, 'exp')
        print(exp_dir)
        if not os.path.exists(exp_dir):
            raise FileNotFoundError(f"Experiment folder not found: {exp_dir}")
    else:
        exp_dir = os.path.join(base_dir, f'exp{exp_number}')
        if not os.path.exists(exp_dir):
            raise FileNotFoundError(f"Experiment folder not found: {exp_dir}")
    
    # Labels directory
    labels_dir = os.path.join(exp_dir, 'labels')
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    # Output directory for the masks
    masks_dir = os.path.join(exp_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    # Process each image file in the experiment folder
    for image_filename in os.listdir(exp_dir):
        if image_filename.endswith('.png'):  # Assuming the images are PNG
            image_path = os.path.join(exp_dir, image_filename)
            label_path = os.path.join(labels_dir, image_filename.replace('.png', '.txt'))
            
            if not os.path.exists(label_path):
                print(f"Label file not found for image: {image_filename}")
                continue
            
            # Load the original image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image at path: {image_path}")
                continue
            
            # Get the image dimensions
            image_height, image_width = image.shape[:2]

            # Read the text file
            with open(label_path, 'r') as file:
                data = file.readlines()

            # Parse the data
            segments = []
            for line in data:
                values = list(map(float, line.strip().split()))
                class_id = int(values[0])
                coords = np.array(values[1:]).reshape(-1, 2)  # Each pair of values represents a point (x, y)
                segments.append((class_id, coords))

            # Create an empty binary image
            mask = np.zeros((image_height, image_width), dtype=np.uint8)

            # Draw the polygons on the mask
            for class_id, coords in segments:
                # Convert coordinates to pixels (if normalized)
                coords[:, 0] *= image_width  # Scale x
                coords[:, 1] *= image_height  # Scale y
                coords = coords.astype(np.int32)  # Convert to integers

                # Draw the polygon
                cv2.fillPoly(mask, [coords], color=255)  # White color (255) for the binary mask

            # Save the binary mask
            mask_filename = f'mask_{image_filename}'
            mask_path = os.path.join(masks_dir, mask_filename)
            cv2.imwrite(mask_path, mask)
            print(f"Mask saved to {mask_path}")

# Example usage
exp_number = input("Enter the experiment number (exp): ")
generate_masks_for_experiment(exp_number)
