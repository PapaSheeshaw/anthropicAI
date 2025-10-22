"""
Interactive inference script for the trained YOLO model.

Features:
1. Real-time webcam classification
2. Upload and classify single image
3. Batch classification on folder
4. Test set validation examples
"""

from ultralytics import YOLO
from pathlib import Path
import sys
import cv2
import numpy as np
from tkinter import Tk, filedialog
import time


def predict_single_image(model, image_path):
    """
    Predict class for a single image.

    Args:
        model: Trained YOLO model
        image_path: Path to image file or numpy array

    Returns:
        Tuple of (class_name, confidence, top5_results)
    """
    # Run prediction
    results = model.predict(image_path, verbose=False)

    # Get probabilities
    probs = results[0].probs

    # Top-1 prediction
    top1_class = probs.top1
    top1_conf = probs.top1conf.cpu().numpy()
    top1_name = results[0].names[top1_class]

    # Top-5 predictions
    top5_indices = probs.top5
    top5_confs = probs.top5conf.cpu().numpy()
    top5_results = [
        (results[0].names[idx], conf)
        for idx, conf in zip(top5_indices, top5_confs)
    ]

    return top1_name, top1_conf, top5_results


def webcam_inference(model):
    """
    Real-time classification using webcam.

    Args:
        model: Trained YOLO model
    """
    show_top3 = True  # Flag to control top3 predictions visibility
    print("\n" + "="*60)
    print("WEBCAM INFERENCE")
    print("="*60)
    print("\nStarting webcam...")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press SPACE to pause/resume")
    print()

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("✗ Error: Could not open webcam")
        print("\nTroubleshooting:")
        print("  1. Check if webcam is connected")
        print("  2. Close other applications using the webcam")
        print("  3. Try changing camera index: cv2.VideoCapture(1)")
        return

    print("✓ Webcam opened successfully")
    print("\nPress any key in the webcam window to start...\n")

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    paused = False
    frame_count = 0
    fps_start_time = time.time()
    fps = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("✗ Error: Failed to grab frame")
                break

            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_start_time)
                fps_start_time = time.time()

            # Predict on frame
            pred_class, confidence, top5 = predict_single_image(model, frame)

            # Create display frame
            display_frame = frame.copy()

            # Add semi-transparent overlay for text background
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 180), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

            # Add prediction text
            y_offset = 30
            cv2.putText(display_frame, f"Prediction: {pred_class}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (0, 255, 0), 2)

            y_offset += 30
            cv2.putText(display_frame, f"Confidence: {confidence:.2%}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 2)

            # Add top-3 predictions if enabled
            if show_top3:
                y_offset += 35
                cv2.putText(display_frame, "Top-3 Predictions:",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1)

                for i, (cls, conf) in enumerate(top5[:3], 1):
                    y_offset += 20
                    text = f"{i}. {cls}: {conf:.1%}"
                    cv2.putText(display_frame, text,
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (150, 150, 255), 1)

            # Add FPS counter
            cv2.putText(display_frame, f"FPS: {fps:.1f}",
                       (540, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 0), 2)

            # Add controls text
            cv2.putText(display_frame, "Q: Quit | S: Save | SPACE: Pause | T  : Toggle Top-3 Predictions",
                       (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       (200, 200, 200), 1)

        else:
            # Show paused frame
            cv2.putText(display_frame, "PAUSED",
                       (270, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                       (0, 0, 255), 3)

        # Display frame
        cv2.imshow('Office Items Classifier - Webcam', display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n✓ Webcam inference stopped")
            break
        elif key == ord('s'):
            # Save current frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"webcam_capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Saved frame as: {filename}")
        elif key == ord('t'):
            # Toggle top 3 predictions visibility
            show_top3 = not show_top3
            print("✓ Top-3 predictions " + ("shown" if show_top3 else "hidden"))


        elif key == ord('p'):
            # Toggle pause
            paused = not paused
            print("⏸ Paused" if paused else "▶ Resumed")

    cap.release()
    cv2.destroyAllWindows()
    print()


def upload_image_inference(model):
    """
    Upload and classify a single image using file dialog.

    Args:
        model: Trained YOLO model
    """
    print("\n" + "="*60)
    print("IMAGE UPLOAD INFERENCE")
    print("="*60)
    print("\nOpening file dialog...")

    # Hide tkinter root window
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not file_path:
        print("✗ No file selected\n")
        return

    file_path = Path(file_path)

    if not file_path.exists():
        print(f"✗ File not found: {file_path}\n")
        return

    print(f"\n✓ Selected: {file_path.name}")
    print("="*60)

    # Predict
    pred_class, confidence, top5 = predict_single_image(model, str(file_path))

    # Display results
    print(f"\nImage: {file_path.name}")
    print(f"  Predicted class: {pred_class}")
    print(f"  Confidence:      {confidence:.2%}")

    print(f"\n  Top-5 predictions:")
    for i, (cls, conf) in enumerate(top5, 1):
        print(f"    {i}. {cls:<20} {conf:.2%}")

    # Show image with prediction
    img = cv2.imread(str(file_path))
    if img is not None:
        # Resize if too large
        height, width = img.shape[:2]
        max_display = 800
        if max(height, width) > max_display:
            scale = max_display / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        # Add prediction text overlay
        overlay = img.copy()
        h, w = img.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        cv2.putText(img, f"Prediction: {pred_class}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 255, 0), 2)
        cv2.putText(img, f"Confidence: {confidence:.2%}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 255), 2)

        # Display image
        print("\nDisplaying image... (press any key to close)")
        cv2.imshow(f'Classification Result - {file_path.name}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print()


def batch_folder_inference(model):
    """
    Classify all images in a selected folder.

    Args:
        model: Trained YOLO model
    """
    print("\n" + "="*60)
    print("BATCH FOLDER INFERENCE")
    print("="*60)
    print("\nOpening folder dialog...")

    # Hide tkinter root window
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    # Open folder dialog
    folder_path = filedialog.askdirectory(
        title="Select a folder with images"
    )

    root.destroy()

    if not folder_path:
        print("✗ No folder selected\n")
        return

    folder_path = Path(folder_path)

    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder_path.glob(ext))
        image_files.extend(folder_path.glob(ext.upper()))

    if len(image_files) == 0:
        print(f"✗ No images found in: {folder_path}\n")
        return

    print(f"\n✓ Found {len(image_files)} images in: {folder_path.name}")
    print("="*60 + "\n")

    # Process all images
    results_list = []

    print("Processing images...\n")
    for img_path in image_files:
        pred_class, confidence, _ = predict_single_image(model, str(img_path))
        results_list.append((img_path.name, pred_class, confidence))
        print(f"✓ {img_path.name:<40} → {pred_class:<20} ({confidence:.2%})")

    # Summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"\nTotal images processed: {len(results_list)}")
    print(f"Average confidence: {np.mean([r[2] for r in results_list]):.2%}")

    # Count predictions per class
    class_counts = {}
    for _, pred_class, _ in results_list:
        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1

    print(f"\nPredictions by class:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls:<20}: {count:3d} ({count/len(results_list):.1%})")

    print()


def test_set_examples(model, project_root):
    """
    Show predictions on test set examples.

    Args:
        model: Trained YOLO model
        project_root: Project root directory
    """
    print("\n" + "="*60)
    print("TEST SET EXAMPLES")
    print("="*60 + "\n")

    test_dir = project_root / "data" / "processed" / "test"

    if not test_dir.exists():
        print("⚠ Test directory not found. Please run data preparation first.\n")
        return

    # Get class folders
    class_folders = sorted([d for d in test_dir.iterdir() if d.is_dir()])

    if not class_folders:
        print("⚠ No class folders found in test directory.\n")
        return

    print(f"Found {len(class_folders)} classes in test set")
    print("Showing 1 example from each class:\n")

    correct = 0
    total = 0

    for class_folder in class_folders:
        images = list(class_folder.glob("*.jpg"))[:1]

        for img_path in images:
            # Predict
            pred_class, confidence, top5 = predict_single_image(model, str(img_path))

            is_correct = pred_class == class_folder.name
            if is_correct:
                correct += 1
            total += 1

            status = "✓" if is_correct else "✗"

            print(f"{status} Image: {img_path.name}")
            print(f"  True class:      {class_folder.name}")
            print(f"  Predicted class: {pred_class}")
            print(f"  Confidence:      {confidence:.2%}")

            if not is_correct:
                print(f"\n  Top-5 predictions:")
                for i, (cls, conf) in enumerate(top5, 1):
                    marker = "←" if cls == class_folder.name else ""
                    print(f"    {i}. {cls:<20} {conf:.2%} {marker}")

            print()

    if total > 0:
        accuracy = correct / total
        print("="*60)
        print(f"Test samples accuracy: {accuracy:.2%} ({correct}/{total})")
        print("="*60 + "\n")


def show_menu():
    """Display interactive menu."""
    print("\n" + "="*60)
    print("OFFICE ITEMS CLASSIFIER - INFERENCE MENU")
    print("="*60)
    print("\nChoose an option:")
    print("\n  1. Webcam Real-time Classification")
    print("  2. Upload and Classify Single Image")
    print("  3. Batch Classify Folder")
    print("  4. Show Test Set Examples")
    print("  5. Exit")
    print("\n" + "="*60)


def main():
    """Main execution function with interactive menu."""
    PROJECT_ROOT = Path(__file__).parent.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "yolo_training" / "weights" / "best.pt"

    # Check if model exists
    if not MODEL_PATH.exists():
        print("\n" + "="*60)
        print("ERROR: MODEL NOT FOUND")
        print("="*60)
        print(f"\n✗ Model not found at: {MODEL_PATH}")
        print("\nPlease train the model first:")
        print("  python src/train.py")
        print("\nOr download a pretrained model and place it in:")
        print(f"  {MODEL_PATH.parent}/")
        print()
        return 1

    # Load model
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    print(f"\nModel path: {MODEL_PATH}")
    print("Loading...")

    try:
        model = YOLO(str(MODEL_PATH))
        print(f"\n✓ Model loaded successfully")
        print(f"✓ Classes: {', '.join(model.names.values())}")
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        return 1

    # Interactive menu loop
    while True:
        show_menu()

        try:
            choice = input("\nEnter your choice (1-5): ").strip()

            if choice == '1':
                webcam_inference(model)

            elif choice == '2':
                upload_image_inference(model)

            elif choice == '3':
                batch_folder_inference(model)

            elif choice == '4':
                test_set_examples(model, PROJECT_ROOT)

            elif choice == '5':
                print("\n" + "="*60)
                print("Thank you for using Office Items Classifier!")
                print("="*60 + "\n")
                break

            else:
                print("\n⚠ Invalid choice. Please enter 1-5.")

        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user.")
            print("="*60 + "\n")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()

    return 0


if __name__ == "__main__":
    exit(main())
