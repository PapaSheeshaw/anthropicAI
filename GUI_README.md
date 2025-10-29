# Office Items Classifier - GUI Application

A modern, user-friendly GUI for the YOLO Office Items Classifier built with CustomTkinter.

## Features

> **NEW**: Toggle top-5 predictions visibility! Click "Hide Top-5" button to save screen space. See [TOGGLE_TOP5_FEATURE.md](TOGGLE_TOP5_FEATURE.md) for details.

> **NEW**: Automatic bounding box detection for the top-1 detected class! See [BOUNDING_BOX_FEATURE.md](BOUNDING_BOX_FEATURE.md) for details.

### 1. Home Screen
- Welcome screen with overview of features
- Model status indicator
- Quick navigation to all features

### 2. Webcam Classification
- Real-time classification using your webcam
- Live video preview with FPS counter
- **Bounding box visualization** for detected objects (detection models)
- Top-5 predictions display with **toggle button** to show/hide
- Confidence scores
- Save current frame capability
- Start/Stop controls

### 3. Upload Image
- Select and classify single images
- Support for multiple image formats (JPG, PNG, BMP, WEBP)
- **Bounding box visualization** for detected objects (detection models)
- Visual display of uploaded image
- Top-5 predictions with confidence scores and **toggle button**
- Easy file selection dialog

### 4. Batch Process
- Process entire folders of images
- Real-time progress tracking
- Results displayed in scrollable list
- Summary statistics (average confidence)
- Supports all common image formats

### 5. Appearance Modes
- Dark mode (default)
- Light mode
- System mode (follows system preferences)

## Installation

### Prerequisites
Make sure you have Python 3.7+ installed.

### Install Dependencies

```bash
pip install customtkinter pillow ultralytics opencv-python numpy
```

## Usage

### Running the GUI

```bash
python src/inference_gui.py
```

### Using the Interface

1. **Home Screen**
   - The home screen displays when you first launch the application
   - Shows the model status and available features

2. **Webcam Classification**
   - Click "Webcam" in the sidebar
   - Click "Start Webcam" to begin real-time classification
   - View predictions in the right panel
   - Click "Save Current Frame" to save a snapshot
   - Click "Stop Webcam" when done

3. **Upload Image**
   - Click "Upload Image" in the sidebar
   - Click "Select Image" to choose a file
   - View the image and prediction results
   - See top-5 predictions with confidence scores

4. **Batch Process**
   - Click "Batch Process" in the sidebar
   - Click "Select Folder" and choose a directory with images
   - Watch the progress as images are processed
   - Scroll through results to see all classifications

5. **Change Appearance**
   - Use the "Appearance" dropdown in the sidebar
   - Choose between Dark, Light, or System mode

## Keyboard Shortcuts

In Webcam mode (when webcam window is active):
- The webcam is controlled through the GUI buttons

## File Outputs

### Saved Webcam Frames
When you save a frame from the webcam, it's saved as:
```
webcam_capture_YYYYMMDD_HHMMSS.jpg
```
in the current directory.

## Features in Detail

### Real-time Performance
- The webcam mode uses a separate thread for video capture
- Predictions are made on each frame
- FPS counter helps monitor performance
- Queue-based architecture prevents UI blocking

### Image Display
- Automatic image resizing for optimal display
- Maintains aspect ratio
- Supports high-resolution images

### Batch Processing
- Processes images in a background thread
- Real-time progress updates
- Non-blocking UI during processing
- Results are displayed as they're computed

## Troubleshooting

### Model Not Found
If you see "Model not loaded" error:
1. Ensure the model exists at: `models/yolo_training/weights/best.pt`
2. Train the model first: `python src/train.py`
3. Check the model path in the code if you moved it

### Webcam Not Opening
If webcam fails to open:
1. Check if another application is using the webcam
2. Try changing the camera index in the code (line with `cv2.VideoCapture(0)`)
3. Ensure you have webcam permissions enabled

### Performance Issues
If the webcam is laggy:
1. Close other applications
2. The FPS counter shows current performance
3. Consider using a lower resolution webcam

## Technical Details

### Architecture
- **GUI Framework**: CustomTkinter (modern tkinter wrapper)
- **ML Framework**: Ultralytics YOLO
- **Image Processing**: OpenCV, PIL
- **Threading**: Python threading for non-blocking operations
- **Queue**: Thread-safe queue for video frames

### Model Loading
The application automatically loads the model from:
```
models/yolo_training/weights/best.pt
```

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- WebP (.webp)

## Customization

### Changing Colors
Edit the color theme in `inference_gui.py`:
```python
ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"
```

### Changing Window Size
Modify the geometry in the `__init__` method:
```python
self.window.geometry("1200x800")  # Width x Height
```

### Changing Model Path
Update the model path in the `__init__` method:
```python
self.model_path = Path("your/custom/path/to/model.pt")
```

## Comparison with CLI Version

### GUI Advantages
- Visual feedback for all operations
- No need to type commands
- Live webcam preview
- Better image display
- Progress tracking for batch operations
- More intuitive for non-technical users

### CLI Advantages
- Faster startup
- Lower memory usage
- Better for automation/scripting
- More detailed console output

## Requirements

```
customtkinter>=5.0.0
pillow>=9.0.0
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.20.0
```

## License

Same as the main project.

## Credits

Built with:
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) - Modern UI framework
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection framework
- [OpenCV](https://opencv.org/) - Computer vision library
