
# YOLOv5 Segmentation Inference Mask Generator

This script generates segmentation masks from inference results obtained using YOLOv5 for segmentation tasks. It processes images and their corresponding label files to create binary masks that represent segmented areas.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)

Ensure you have Python installed on your system. You can install the required packages using pip:

```bash
pip install opencv-python numpy
```

## Usage

### 1. Prepare Directory Structure

Ensure you have the following directory structure:

```
yolov5
├── runs
│   └── predict-seg
│       └── expX  # Replace X with experiment number
│           ├── labels
│           │   ├── img1.txt
│           │   ├── img2.txt
│           │   └── ...
│           ├── img1.png
│           ├── img2.png
│           └── ...
```

- **Note**: 
  - `expX` represents the experiment number (`exp`, `exp2`, etc.).
  - Each image (`img1.png`, `img2.png`, etc.) should have a corresponding label file (`img1.txt`, `img2.txt`, etc.) in the `labels` folder.

### 2. Run the Script

- Open a terminal or command prompt.
- Navigate to the directory where the script (`yolo2mask.py`) is saved.
- Run the script and provide the experiment number (`expX`) when prompted:

  ```bash
  python yolo2mask.py
  ```

- Enter the experiment number (`expX`) when prompted and press `Enter`.

### 3. Output

- The script will generate binary masks for each image in the specified experiment (`expX`) and save them in a `masks` folder inside the corresponding experiment folder (`expX`).

### 4. Generated Masks

- Each generated mask will be saved with a filename format `mask_imgX.png` in the `masks` folder.

## Example

Suppose you have YOLOv5 segmentation results for `exp`:

### Directory structure:

```
yolov5
├── runs
│   └── predict-seg
│       └── exp
│           ├── labels
│           │   ├── img1.txt
│           │   ├── img2.txt
│           │   └── ...
│           ├── img1.png
│           ├── img2.png
│           └── ...
```

### Run the script:

```bash
python generate_masks.py
```

- Enter `exp` when prompted.

- Masks will be generated and saved in:
  
  ```
  yolov5/runs/predict-seg/exp/masks/
  ```

---
