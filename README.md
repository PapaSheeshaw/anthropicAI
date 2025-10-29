# Office Items Classifier

A deep learning-based object detection and classification system for identifying common office items using YOLOv8. The model achieves **85.6% top-1 accuracy** and **99.3% top-5 accuracy** across six office item categories.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **High Accuracy**: 85.6% top-1 accuracy on office item classification
- **Real-time Inference**: 15-22 FPS on CPU, 80+ FPS on GPU
- **Modern GUI**: CustomTkinter-based interface with dark/light mode
- **Multiple Modes**: Webcam, single image upload, and batch processing
- **Pre-trained Model**: Ready-to-use trained weights included

## Detected Classes

The model can identify the following office items:

1. **Chair** - Office chairs and seating furniture
2. **Computer Keyboard** - Desktop and laptop keyboards
3. **Computer Mouse** - Wired and wireless mice
4. **Headphones** - Over-ear, on-ear, and in-ear devices
5. **Mobile Phone** - Smartphones and mobile devices
6. **Pen** - Writing instruments

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [GUI Application](#gui-application)
  - [Command Line](#command-line)
  - [Python API](#python-api)
- [Training Your Own Model](#training-your-own-model)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training/inference

### Step 1: Clone the Repository

```bash
git clone https://github.com/PapaSheeshaw/anthropicAI.git
cd anthropicAI
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with PyTorch installation, visit [pytorch.org](https://pytorch.org) to get the installation command for your specific system configuration.

### Step 4: Verify Installation

```bash
python -c "from ultralytics import YOLO; print('Installation successful!')"
```

## Quick Start

### Run the GUI Application

The easiest way to use the classifier:

```bash
python src/inference_gui.py
```

This launches a modern GUI with:
- **Webcam Mode**: Real-time classification from your camera
- **Upload Mode**: Classify individual images
- **Batch Mode**: Process entire folders of images

### Classify a Single Image (Command Line)

```python
from ultralytics import YOLO
from pathlib import Path

# Load the trained model
model = YOLO('models/yolo_training/weights/best.pt')

# Classify an image
results = model.predict('path/to/your/image.jpg', verbose=False)

# Get predictions
probs = results[0].probs
top1_class = model.names[probs.top1]
confidence = probs.top1conf.cpu().numpy()

print(f"Predicted: {top1_class} (Confidence: {confidence:.2%})")
```

## Usage

### GUI Application

#### Launching the GUI

```bash
python src/inference_gui.py
```

#### Features:

**1. Home Screen**
- Overview of model capabilities
- Model status indicator
- Quick navigation

**2. Webcam Classification**
- Click "Webcam" in sidebar
- Click "Start Webcam" to begin
- View real-time predictions with confidence scores
- Toggle top-5 predictions visibility
- Save frames with "Save Current Frame" button

**3. Upload Image**
- Click "Upload Image" in sidebar
- Select an image file (JPG, PNG, BMP, WEBP)
- View image with top-5 predictions
- Toggle predictions panel for cleaner view

**4. Batch Processing**
- Click "Batch Process" in sidebar
- Select a folder containing images
- Monitor real-time progress
- View summary statistics

**5. Appearance Modes**
- Use dropdown in sidebar to switch between:
  - Dark mode (default)
  - Light mode
  - System mode (follows OS preferences)

### Command Line

#### Single Image Classification

Create a Python script or use interactively:

```python
from ultralytics import YOLO

# Load model
model = YOLO('models/yolo_training/weights/best.pt')

# Predict
results = model.predict('image.jpg')

# Access results
probs = results[0].probs
print(f"Top-1: {model.names[probs.top1]} ({probs.top1conf:.2%})")
print(f"Top-5: {probs.top5}")
```

#### Batch Processing

```python
from ultralytics import YOLO
from pathlib import Path

model = YOLO('models/yolo_training/weights/best.pt')

# Process all images in a folder
image_folder = Path('path/to/images')
for img_path in image_folder.glob('*.jpg'):
    results = model.predict(str(img_path), verbose=False)
    probs = results[0].probs
    print(f"{img_path.name}: {model.names[probs.top1]} ({probs.top1conf:.2%})")
```

### Python API

#### Basic Usage

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('models/yolo_training/weights/best.pt')

# From file
results = model.predict('image.jpg')

# From numpy array (OpenCV)
img = cv2.imread('image.jpg')
results = model.predict(img)

# From PIL Image
from PIL import Image
img = Image.open('image.jpg')
results = model.predict(img)
```

#### Advanced: Custom Confidence Threshold

```python
from ultralytics import YOLO

model = YOLO('models/yolo_training/weights/best.pt')
results = model.predict('image.jpg', conf=0.7)  # Only predictions >70% confidence

probs = results[0].probs
if probs.top1conf > 0.7:
    print(f"High confidence: {model.names[probs.top1]}")
else:
    print("Low confidence prediction - review manually")
```

#### Webcam Integration

```python
import cv2
from ultralytics import YOLO

model = YOLO('models/yolo_training/weights/best.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False)
    probs = results[0].probs

    # Display prediction
    cv2.putText(frame,
                f"{model.names[probs.top1]}: {probs.top1conf:.2%}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Office Items Classifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Training Your Own Model

### Step 1: Download Dataset

The project uses Google's Open Images Dataset v7. To download:

```bash
python src/download_data.py
```

This downloads ~300 images per class for the 6 office item categories.

### Step 2: Prepare Data

Clean and organize the downloaded images:

```bash
python src/clean_data.py
python src/split_data.py
```

This creates train/validation splits in `data/processed/`.

### Step 3: Train Model

```bash
python src/train.py
```

**Training Configuration** (edit `src/train.py` if needed):
- **Epochs**: 20
- **Batch Size**: 32
- **Image Size**: 224×224
- **Model**: YOLOv8m-cls (medium variant)
- **Optimizer**: AdamW
- **Learning Rate**: 0.001 → 0.00001 (cosine decay)

**Training Time**:
- CPU: ~1-2 hours
- GPU (GTX 1660+): ~15-30 minutes

### Step 4: Evaluate Model

```bash
python src/evaluate_yolo.py
```

This generates:
- Per-class accuracy metrics
- Confusion matrix
- Evaluation report (JSON)

### Custom Dataset

To train on your own dataset:

1. Organize images in this structure:
   ```
   data/custom/
   ├── train/
   │   ├── class1/
   │   │   ├── image1.jpg
   │   │   └── image2.jpg
   │   └── class2/
   │       └── image1.jpg
   └── val/
       ├── class1/
       └── class2/
   ```

2. Update `DATA_DIR` in `src/train.py`:
   ```python
   DATA_DIR = PROJECT_ROOT / "data" / "custom"
   ```

3. Run training:
   ```bash
   python src/train.py
   ```

## Project Structure

```
PythonProject/
├── src/
│   ├── download_data.py      # Download Open Images dataset
│   ├── clean_data.py          # Clean and organize images
│   ├── split_data.py          # Train/val/test split
│   ├── train.py               # Train YOLOv8 model
│   ├── evaluate_yolo.py       # Evaluate model performance
│   └── inference_gui.py       # GUI application
├── data/
│   ├── raw/                   # Downloaded images
│   ├── cleaned/               # Cleaned images
│   └── processed/             # Train/val split
│       ├── train/
│       └── val/
├── models/
│   └── yolo_training/
│       └── weights/
│           └── best.pt        # Trained model weights
├── results/
│   ├── confusion_matrix.png
│   ├── results.csv
│   └── evaluation_results.json
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── GUI_README.md              # Detailed GUI documentation
└── MODEL_DOCUMENTATION.md     # Technical model details
```

## Model Performance

### Accuracy Metrics

| Metric | Value |
|--------|-------|
| **Top-1 Accuracy** | 85.6% |
| **Top-5 Accuracy** | 99.3% |
| **Training Loss** | 0.0297 |
| **Validation Loss** | 0.483 |

### Per-Class Performance

| Class | Accuracy |
|-------|----------|
| Mobile Phone | 91.3% |
| Pen | 89.8% |
| Computer Mouse | 88.5% |
| Chair | 84.4% |
| Headphones | 80.0% |
| Computer Keyboard | 69.6% |

**Note**: Computer Keyboard has lower accuracy due to visual similarity with Computer Mouse. See [90_PERCENT_ACTION_CHECKLIST.md](90_PERCENT_ACTION_CHECKLIST.md) for improvement strategies.

### Inference Speed

| Configuration | Latency | FPS | Use Case |
|---------------|---------|-----|----------|
| CPU (Intel i5/i7) | 45-65ms | 15-22 | Real-time webcam |
| GPU (GTX 1660+) | 8-12ms | 80-125 | High-throughput |
| Batch (CPU, n=8) | 28-38ms/img | 26-35 | Batch processing |

## Technical Details

### Model Architecture

- **Framework**: YOLOv8 (Ultralytics)
- **Variant**: YOLOv8m-cls (medium)
- **Parameters**: ~25.9 million
- **Input Size**: 224×224 pixels
- **Output**: 6-class probability distribution

### Training Details

- **Dataset Size**: ~1,508 images (83% train, 17% validation)
- **Transfer Learning**: Pretrained on ImageNet-1k
- **Optimizer**: AdamW (weight_decay=0.0008)
- **Learning Rate**: Cosine annealing (0.001 → 0.00001)
- **Data Augmentation**: Flip, rotation (±15°), scale (0.5-1.5×), HSV, random erasing

### Data Augmentation

The model uses extensive augmentation to improve generalization:
- Horizontal flip (50%)
- Rotation (±15°)
- Translation (±10%)
- Scaling (0.5-1.5×)
- HSV color jitter
- Random erasing (40%, 2-40% area)

For full technical documentation, see [MODEL_DOCUMENTATION.md](MODEL_DOCUMENTATION.md).

## Troubleshooting

### Model Not Found Error

**Error**: `Model not loaded` or file not found

**Solution**:
1. Ensure model exists at: `models/yolo_training/weights/best.pt`
2. If missing, train the model: `python src/train.py`
3. Or download pre-trained weights from releases

### Webcam Not Opening

**Error**: Webcam fails to open or shows black screen

**Solutions**:
1. Check if another application is using the webcam
2. Try different camera index (edit `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`)
3. Verify webcam permissions in system settings
4. On Windows: Check Camera privacy settings

### CUDA/GPU Errors

**Error**: CUDA out of memory or GPU not detected

**Solutions**:
1. Reduce batch size in `src/train.py`: `batch_size=16` or `batch_size=8`
2. Install CPU-only version: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
3. For GPU: Ensure CUDA drivers are installed correctly

### Slow Inference

**Issue**: Predictions are taking too long

**Solutions**:
1. Use GPU if available (80+ FPS vs 15-22 FPS on CPU)
2. Reduce image size (trade accuracy for speed)
3. Use smaller model: `yolov8n-cls.pt` (nano variant)
4. Convert to ONNX for 15-20% speedup

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'X'`

**Solution**:
```bash
# Ensure virtual environment is activated
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Low Accuracy on Custom Images

**Issue**: Model performs poorly on your images

**Reasons & Solutions**:
1. **Out of distribution**: Images differ significantly from training data
   - Solution: Add similar images to training set and retrain
2. **Poor lighting**: Underexposed or overexposed images
   - Solution: Improve image quality or add similar lighting conditions to training
3. **Occlusion**: Object is partially hidden
   - Solution: Model trained with random erasing should handle this, but extreme occlusion (>60%) degrades performance
4. **Unusual angles**: Object photographed from uncommon viewpoint
   - Solution: Add more viewpoint diversity to training data

## Documentation

- **[GUI_README.md](GUI_README.md)**: Detailed GUI usage guide
- **[MODEL_DOCUMENTATION.md](MODEL_DOCUMENTATION.md)**: Complete technical documentation


## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request


## Acknowledgments

- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Dataset**: [Google Open Images v7](https://storage.googleapis.com/openimages/web/index.html)
- **GUI Framework**: [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- **Computer Vision**: [OpenCV](https://opencv.org/)

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{office_items_classifier,
  author = {Keith Muhlanga, Paul Kamani, Nkem O Iwuagwu},
  title = {Office Items Classifier: YOLOv8-based Object Detection},
  year = {2025},
  url = {https://github.com/PapaSheeshaw/anthropicAI}
}
```

## Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Email: keithmuhlanga@gmail.com

---

