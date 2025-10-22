"""
Office Item Classifier - Webcam Testing Script
===============================================
Real-time classification using webcam feed with trained YOLO model.

Features:
- Live webcam feed with predictions
- Confidence scores and class names
- FPS counter
- Visual feedback (bounding box around frame)
- Keyboard controls for interaction

Controls:
- 'q' or 'ESC': Quit
- 's': Save screenshot
- 'p': Pause/Resume
- 'c': Toggle confidence display
"""

import cv2
import numpy as np
from pathlib import Path
import time
from ultralytics import YOLO
import argparse


# =============================================================================
# CONFIGURATION
# =============================================================================

# Resolve project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# WEBCAM TESTING CLASS
# =============================================================================

class WebcamClassifier:
    """Real-time classification using webcam."""

    def __init__(self, model_path, camera_id=0, confidence_threshold=0.5):
        """
        Initialize webcam classifier.

        Args:
            model_path: Path to trained YOLO model (.pt file)
            camera_id: Camera device ID (0 for default webcam)
            confidence_threshold: Minimum confidence to display prediction
        """
        self.model_path = Path(model_path)
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold

        # Load model
        print(f"Loading model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.class_names = list(self.model.names.values())
        print(f"Model loaded with {len(self.class_names)} classes")
        print(f"Classes: {', '.join(self.class_names)}")

        # Initialize webcam
        print(f"\nInitializing webcam (camera {self.camera_id})...")
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        # Get camera properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        print(f"Camera initialized: {self.frame_width}x{self.frame_height} @ {self.fps}fps")

        # State variables
        self.paused = False
        self.show_confidence = True
        self.screenshot_count = 0

        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0

    def predict(self, frame):
        """
        Run prediction on frame.

        Args:
            frame: OpenCV frame (BGR)

        Returns:
            dict: Prediction results with class_name, confidence, class_id
        """
        # Run YOLO prediction
        results = self.model.predict(frame, verbose=False, conf=self.confidence_threshold)

        if len(results) == 0:
            return None

        # Get prediction
        probs = results[0].probs

        if probs is None:
            return None

        # Get top prediction
        top1_class = probs.top1
        top1_conf = float(probs.top1conf)
        class_name = self.class_names[top1_class]

        # Get top 3 predictions
        top3_indices = probs.top5[:3]  # Get top 3 from top5
        top3_confs = [float(probs.data[i]) for i in top3_indices]
        top3_classes = [self.class_names[i] for i in top3_indices]

        return {
            'class_id': top1_class,
            'class_name': class_name,
            'confidence': top1_conf,
            'top3_classes': top3_classes,
            'top3_confidences': top3_confs
        }

    def draw_prediction(self, frame, prediction):
        """
        Draw prediction on frame.

        Args:
            frame: OpenCV frame (BGR)
            prediction: Prediction dictionary

        Returns:
            frame: Frame with drawn prediction
        """
        height, width = frame.shape[:2]

        if prediction is None:
            # No prediction - draw red border
            cv2.rectangle(frame, (10, 10), (width-10, height-10), (0, 0, 255), 3)
            cv2.putText(frame, "No object detected", (30, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return frame

        # Draw green border for successful prediction
        cv2.rectangle(frame, (10, 10), (width-10, height-10), (0, 255, 0), 3)

        # Prepare text
        class_name = prediction['class_name']
        confidence = prediction['confidence']

        # Main prediction (large text)
        text = f"{class_name}"
        if self.show_confidence:
            text += f" ({confidence:.2%})"

        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3

        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Draw semi-transparent background for main text
        padding = 10
        bg_x1 = 20
        bg_y1 = 40
        bg_x2 = bg_x1 + text_width + padding * 2
        bg_y2 = bg_y1 + text_height + padding * 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw main prediction text
        cv2.putText(frame, text, (30, 70),
                   font, font_scale, (0, 255, 0), thickness)

        # Draw top 3 predictions (smaller text)
        if self.show_confidence:
            y_offset = 120
            for i, (cls, conf) in enumerate(zip(
                prediction['top3_classes'][:3],
                prediction['top3_confidences'][:3]
            )):
                text = f"{i+1}. {cls}: {conf:.1%}"
                color = (0, 255, 0) if i == 0 else (200, 200, 200)
                cv2.putText(frame, text, (30, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 35

        return frame

    def draw_info(self, frame):
        """
        Draw FPS and controls info.

        Args:
            frame: OpenCV frame (BGR)

        Returns:
            frame: Frame with info drawn
        """
        height, width = frame.shape[:2]

        # FPS counter (top right)
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (width - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Controls (bottom)
        controls = [
            "Q/ESC: Quit",
            "S: Screenshot",
            "P: Pause",
            "C: Toggle confidence"
        ]

        y_offset = height - 120
        for control in controls:
            cv2.putText(frame, control, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

        # Paused indicator
        if self.paused:
            cv2.putText(frame, "PAUSED", (width // 2 - 80, height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        return frame

    def save_screenshot(self, frame, prediction):
        """
        Save screenshot with prediction.

        Args:
            frame: OpenCV frame (BGR)
            prediction: Prediction dictionary
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if prediction:
            filename = f"screenshot_{timestamp}_{prediction['class_name']}.jpg"
        else:
            filename = f"screenshot_{timestamp}_no_detection.jpg"

        filepath = SCREENSHOTS_DIR / filename
        cv2.imwrite(str(filepath), frame)

        self.screenshot_count += 1
        print(f"Screenshot saved: {filepath}")

    def update_fps(self):
        """Calculate and update FPS."""
        self.fps_frame_count += 1

        # Update FPS every second
        if self.fps_frame_count >= 30:
            elapsed = time.time() - self.fps_start_time
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = time.time()

    def run(self):
        """
        Main loop for webcam testing.
        """
        print("\n" + "=" * 70)
        print("WEBCAM TESTING - YOLO CLASSIFICATION")
        print("=" * 70)
        print("\nControls:")
        print("  Q or ESC : Quit")
        print("  S        : Save screenshot")
        print("  P        : Pause/Resume")
        print("  C        : Toggle confidence display")
        print("\nStarting webcam feed...\n")

        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()

                if not ret:
                    print("Failed to read frame from webcam")
                    break

                # Process frame if not paused
                if not self.paused:
                    # Run prediction
                    prediction = self.predict(frame)

                    # Draw prediction
                    frame = self.draw_prediction(frame, prediction)
                else:
                    prediction = None

                # Draw info
                frame = self.draw_info(frame)

                # Update FPS
                self.update_fps()

                # Display frame
                cv2.imshow('Office Item Classifier - Webcam', frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    print("\nQuitting...")
                    break

                elif key == ord('s'):  # Save screenshot
                    self.save_screenshot(frame, prediction)

                elif key == ord('p'):  # Pause/Resume
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "RESUMED"
                    print(f"\n{status}")

                elif key == ord('c'):  # Toggle confidence
                    self.show_confidence = not self.show_confidence
                    status = "ON" if self.show_confidence else "OFF"
                    print(f"\nConfidence display: {status}")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            # Cleanup
            print("\nCleaning up...")
            self.cap.release()
            cv2.destroyAllWindows()

            print(f"\nSession summary:")
            print(f"  Screenshots saved: {self.screenshot_count}")
            print(f"  Location: {SCREENSHOTS_DIR}")
            print("\n[OK] Done!")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


# =============================================================================
# MODEL SELECTION
# =============================================================================

def find_best_model():
    """
    Find the best trained model.

    Returns:
        Path: Path to best model, or None if not found
    """
    # Check for YOLO training directory
    yolo_dir = MODEL_DIR / "yolo_training" / "weights"

    if yolo_dir.exists():
        best_model = yolo_dir / "best.pt"
        if best_model.exists():
            return best_model

    # Check for any .pt files in models directory
    pt_files = list(MODEL_DIR.glob("**/*.pt"))
    if pt_files:
        # Return the most recently modified
        return max(pt_files, key=lambda x: x.stat().st_mtime)

    return None


def select_model_interactive():
    """
    Let user select a model interactively.

    Returns:
        Path: Selected model path, or None if cancelled
    """
    # Find all .pt files
    pt_files = list(MODEL_DIR.glob("**/*.pt"))

    if not pt_files:
        print(f"[X] No model files found in {MODEL_DIR}")
        return None

    print("\n" + "=" * 70)
    print("AVAILABLE MODELS")
    print("=" * 70)
    print(f"\nFound {len(pt_files)} model(s):\n")

    for i, model_file in enumerate(pt_files, 1):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        mod_time = time.strftime(
            '%Y-%m-%d %H:%M',
            time.localtime(model_file.stat().st_mtime)
        )

        # Highlight best model
        marker = " [BEST]" if model_file.name == "best.pt" else ""

        print(f"{i}. {model_file.name}{marker}")
        print(f"   Path: {model_file.parent}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Modified: {mod_time}")
        print()

    while True:
        try:
            choice = input(f"Select model (1-{len(pt_files)}) or 'q' to quit: ").strip()

            if choice.lower() == 'q':
                return None

            idx = int(choice) - 1

            if 0 <= idx < len(pt_files):
                return pt_files[idx]
            else:
                print(f"   [!] Invalid choice. Enter 1-{len(pt_files)}")

        except ValueError:
            print("   [!] Please enter a valid number or 'q'")
        except KeyboardInterrupt:
            print("\n")
            return None


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Test YOLO classification model using webcam'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model file (.pt)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.3,
        help='Confidence threshold (default: 0.3)'
    )

    args = parser.parse_args()

    # Find model
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"[X] Model not found: {model_path}")
            return
    else:
        # Try to find best model automatically
        model_path = find_best_model()

        if model_path:
            print(f"Found model: {model_path}")
            use_default = input("Use this model? (y/n): ").strip().lower()

            if use_default != 'y':
                model_path = select_model_interactive()
        else:
            print("No model found automatically.")
            model_path = select_model_interactive()

    if not model_path:
        print("[X] No model selected. Exiting.")
        return

    # Create and run classifier
    try:
        classifier = WebcamClassifier(
            model_path=model_path,
            camera_id=args.camera,
            confidence_threshold=args.confidence
        )

        classifier.run()

    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()
