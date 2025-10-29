"""
Modern GUI for YOLO Office Items Classifier using CustomTkinter.

Features:
- Webcam real-time classification with live preview
- Single image upload and classification
- Batch folder processing
- Clean modern interface with dark/light mode
"""

import customtkinter as ctk
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from tkinter import filedialog
import queue


class OfficeItemsClassifierGUI:
    def __init__(self):
        # Initialize window
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.window = ctk.CTk()
        self.window.title("Office Items Classifier")
        self.window.geometry("1200x800")

        # Model variables
        self.model = None
        self.model_path = Path(__file__).parent.parent / "models" / "yolo_training" / "weights" / "best.pt"

        # Webcam variables
        self.webcam_active = False
        self.cap = None
        self.webcam_thread = None
        self.frame_queue = queue.Queue(maxsize=2)

        # UI state variables
        self.show_top5 = True  # Toggle for top-5 predictions visibility

        # UI Setup
        self.setup_ui()

        # Load model
        self.load_model()

    def setup_ui(self):
        """Setup the user interface."""
        # Configure grid
        self.window.grid_columnconfigure(0, weight=0)  # Sidebar
        self.window.grid_columnconfigure(1, weight=1)  # Main content
        self.window.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.setup_sidebar()

        # Main content area
        self.main_frame = ctk.CTkFrame(self.window)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # Title in main area
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Welcome to Office Items Classifier",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=20)

        # Content frame (will be replaced based on selection)
        self.content_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)

        self.show_home_screen()

    def setup_sidebar(self):
        """Setup the sidebar with navigation buttons."""
        self.sidebar = ctk.CTkFrame(self.window, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_rowconfigure(7, weight=1)

        # Logo/Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="🏢 Classifier",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 30))

        # Navigation buttons
        self.home_btn = ctk.CTkButton(
            self.sidebar,
            text="Home",
            command=self.show_home_screen,
            font=ctk.CTkFont(size=14)
        )
        self.home_btn.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.webcam_btn = ctk.CTkButton(
            self.sidebar,
            text="Webcam",
            command=self.show_webcam_screen,
            font=ctk.CTkFont(size=14)
        )
        self.webcam_btn.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.upload_btn = ctk.CTkButton(
            self.sidebar,
            text="Upload Image",
            command=self.show_upload_screen,
            font=ctk.CTkFont(size=14)
        )
        self.upload_btn.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        self.batch_btn = ctk.CTkButton(
            self.sidebar,
            text="Batch Process",
            command=self.show_batch_screen,
            font=ctk.CTkFont(size=14)
        )
        self.batch_btn.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

        # Appearance mode switch
        self.appearance_label = ctk.CTkLabel(
            self.sidebar,
            text="Appearance:",
            font=ctk.CTkFont(size=12)
        )
        self.appearance_label.grid(row=8, column=0, padx=20, pady=(20, 0))

        self.appearance_mode = ctk.CTkOptionMenu(
            self.sidebar,
            values=["Dark", "Light", "System"],
            command=self.change_appearance_mode,
            font=ctk.CTkFont(size=12)
        )
        self.appearance_mode.grid(row=9, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.appearance_mode.set("Dark")

    def change_appearance_mode(self, new_appearance_mode):
        """Change the appearance mode."""
        ctk.set_appearance_mode(new_appearance_mode.lower())

    def clear_content_frame(self):
        """Clear the content frame."""
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def show_home_screen(self):
        """Show the home/welcome screen."""
        self.clear_content_frame()
        self.title_label.configure(text="Welcome to Office Items Classifier")

        # Info frame
        info_frame = ctk.CTkFrame(self.content_frame)
        info_frame.grid(row=0, column=0, padx=40, pady=40, sticky="nsew")
        info_frame.grid_columnconfigure(0, weight=1)

        welcome_text = ctk.CTkLabel(
            info_frame,
            text="Choose an option from the sidebar to get started:",
            font=ctk.CTkFont(size=16),
            wraplength=600
        )
        welcome_text.pack(pady=20)

        # Feature cards
        features = [
            ("📹 Webcam", "Real-time classification using your webcam"),
            ("📁 Upload Image", "Classify a single image file"),
            ("📂 Batch Process", "Process multiple images in a folder"),
        ]

        for icon_title, description in features:
            feature_frame = ctk.CTkFrame(info_frame)
            feature_frame.pack(pady=10, padx=20, fill="x")

            title_label = ctk.CTkLabel(
                feature_frame,
                text=icon_title,
                font=ctk.CTkFont(size=18, weight="bold")
            )
            title_label.pack(pady=(10, 5))

            desc_label = ctk.CTkLabel(
                feature_frame,
                text=description,
                font=ctk.CTkFont(size=14),
                text_color="gray"
            )
            desc_label.pack(pady=(0, 10))

        # Model status
        if self.model:
            status_text = f"✓ Model loaded successfully\nClasses: {', '.join(self.model.names.values())}"
            status_color = "green"
        else:
            status_text = "✗ Model not loaded"
            status_color = "red"

        status_label = ctk.CTkLabel(
            info_frame,
            text=status_text,
            font=ctk.CTkFont(size=14),
            text_color=status_color
        )
        status_label.pack(pady=20)

    def show_webcam_screen(self):
        """Show the webcam inference screen."""
        self.clear_content_frame()
        self.title_label.configure(text="Webcam Classification")

        if not self.model:
            self.show_error("Model not loaded. Please check model path.")
            return

        # Configure grid
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(1, weight=0)
        self.content_frame.grid_rowconfigure(0, weight=1)

        # Video display
        self.video_label = ctk.CTkLabel(self.content_frame, text="")
        self.video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Right panel for results
        right_panel = ctk.CTkFrame(self.content_frame, width=300)
        right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        right_panel.grid_propagate(False)

        # Results display
        results_title = ctk.CTkLabel(
            right_panel,
            text="Prediction Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        results_title.pack(pady=20)

        self.pred_class_label = ctk.CTkLabel(
            right_panel,
            text="Class: -",
            font=ctk.CTkFont(size=16)
        )
        self.pred_class_label.pack(pady=10)

        self.confidence_label = ctk.CTkLabel(
            right_panel,
            text="Confidence: -",
            font=ctk.CTkFont(size=14)
        )
        self.confidence_label.pack(pady=10)

        # Top-5 predictions
        top5_title = ctk.CTkLabel(
            right_panel,
            text="Top-5 Predictions:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        top5_title.pack(pady=(20, 10))

        self.top5_frame = ctk.CTkFrame(right_panel)
        self.top5_frame.pack(pady=10, padx=20, fill="x")

        self.top5_labels = []
        for i in range(5):
            label = ctk.CTkLabel(
                self.top5_frame,
                text=f"{i+1}. -",
                font=ctk.CTkFont(size=12)
            )
            label.pack(pady=2, anchor="w")
            self.top5_labels.append(label)

        # FPS counter
        self.fps_label = ctk.CTkLabel(
            right_panel,
            text="FPS: 0",
            font=ctk.CTkFont(size=12)
        )
        self.fps_label.pack(pady=20)

        # Control buttons
        button_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        button_frame.pack(pady=20, fill="x", padx=20)

        self.start_webcam_btn = ctk.CTkButton(
            button_frame,
            text="Start Webcam",
            command=self.toggle_webcam,
            font=ctk.CTkFont(size=14),
            fg_color="green"
        )
        self.start_webcam_btn.pack(pady=5, fill="x")

        self.save_frame_btn = ctk.CTkButton(
            button_frame,
            text="Save Current Frame",
            command=self.save_webcam_frame,
            font=ctk.CTkFont(size=14),
            state="disabled"
        )
        self.save_frame_btn.pack(pady=5, fill="x")

        self.toggle_top5_btn = ctk.CTkButton(
            button_frame,
            text="Hide Top-5",
            command=self.toggle_top5_predictions,
            font=ctk.CTkFont(size=14),
            fg_color="gray"
        )
        self.toggle_top5_btn.pack(pady=5, fill="x")

    def show_upload_screen(self):
        """Show the single image upload screen."""
        self.clear_content_frame()
        self.title_label.configure(text="Upload and Classify Image")

        if not self.model:
            self.show_error("Model not loaded. Please check model path.")
            return

        # Configure grid
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(1, weight=0)
        self.content_frame.grid_rowconfigure(0, weight=1)

        # Image display
        self.upload_image_label = ctk.CTkLabel(
            self.content_frame,
            text="No image loaded",
            font=ctk.CTkFont(size=16)
        )
        self.upload_image_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Right panel for results and controls
        right_panel = ctk.CTkFrame(self.content_frame, width=300)
        right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        right_panel.grid_propagate(False)

        # Upload button
        upload_btn = ctk.CTkButton(
            right_panel,
            text="Select Image",
            command=self.upload_and_classify,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50
        )
        upload_btn.pack(pady=20, padx=20, fill="x")

        # Results
        results_title = ctk.CTkLabel(
            right_panel,
            text="Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        results_title.pack(pady=(20, 10))

        self.upload_pred_label = ctk.CTkLabel(
            right_panel,
            text="Class: -",
            font=ctk.CTkFont(size=16)
        )
        self.upload_pred_label.pack(pady=10)

        self.upload_conf_label = ctk.CTkLabel(
            right_panel,
            text="Confidence: -",
            font=ctk.CTkFont(size=14)
        )
        self.upload_conf_label.pack(pady=10)

        # Top-5 predictions
        top5_title = ctk.CTkLabel(
            right_panel,
            text="Top-5 Predictions:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        top5_title.pack(pady=(20, 10))

        self.upload_top5_frame = ctk.CTkFrame(right_panel)
        self.upload_top5_frame.pack(pady=10, padx=20, fill="x")

        self.upload_top5_labels = []
        for i in range(5):
            label = ctk.CTkLabel(
                self.upload_top5_frame,
                text=f"{i+1}. -",
                font=ctk.CTkFont(size=12)
            )
            label.pack(pady=2, anchor="w")
            self.upload_top5_labels.append(label)

        # Toggle button for top-5 predictions
        self.upload_toggle_top5_btn = ctk.CTkButton(
            right_panel,
            text="Hide Top-5",
            command=self.toggle_top5_predictions,
            font=ctk.CTkFont(size=14),
            fg_color="gray"
        )
        self.upload_toggle_top5_btn.pack(pady=10, padx=20, fill="x")

    def show_batch_screen(self):
        """Show the batch processing screen."""
        self.clear_content_frame()
        self.title_label.configure(text="Batch Process Folder")

        if not self.model:
            self.show_error("Model not loaded. Please check model path.")
            return

        # Configure grid
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(1, weight=1)

        # Controls frame
        controls_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        controls_frame.grid(row=0, column=0, pady=20, sticky="ew")
        controls_frame.grid_columnconfigure(1, weight=1)

        select_folder_btn = ctk.CTkButton(
            controls_frame,
            text="Select Folder",
            command=self.batch_process_folder,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=40
        )
        select_folder_btn.grid(row=0, column=0, padx=20)

        self.batch_status_label = ctk.CTkLabel(
            controls_frame,
            text="No folder selected",
            font=ctk.CTkFont(size=14)
        )
        self.batch_status_label.grid(row=0, column=1, padx=20, sticky="w")

        # Results frame with scrollbar
        results_frame = ctk.CTkScrollableFrame(self.content_frame)
        results_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        results_frame.grid_columnconfigure(0, weight=1)

        self.batch_results_frame = results_frame

    def load_model(self):
        """Load the YOLO model."""
        if not self.model_path.exists():
            print(f"Model not found at: {self.model_path}")
            return

        try:
            self.model = YOLO(str(self.model_path))
            print(f"Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict_image(self, image_source):
        """Predict class for an image with detection boxes."""
        if not self.model:
            return None, None, None, None

        # Check if model is detection or classification
        results = self.model.predict(image_source, verbose=False)

        # Try to get classification probabilities (for classification models)
        if hasattr(results[0], 'probs') and results[0].probs is not None:
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

            # No bounding boxes for classification
            return top1_name, top1_conf, top5_results, None

        # Handle detection models
        elif hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            # Get the detection with highest confidence
            if len(boxes) > 0:
                # Sort by confidence
                confidences = boxes.conf.cpu().numpy()
                best_idx = np.argmax(confidences)

                # Top-1 detection
                top1_conf = confidences[best_idx]
                top1_class_id = int(boxes.cls[best_idx].cpu().numpy())
                top1_name = results[0].names[top1_class_id]
                top1_box = boxes.xyxy[best_idx].cpu().numpy()

                # Get top-5 detections or all if less than 5
                sorted_indices = np.argsort(confidences)[::-1][:5]
                top5_results = [
                    (results[0].names[int(boxes.cls[i].cpu().numpy())], confidences[i])
                    for i in sorted_indices
                ]

                return top1_name, top1_conf, top5_results, top1_box
            else:
                return None, None, [], None
        else:
            # No detections
            return None, None, [], None

    def toggle_webcam(self):
        """Start or stop the webcam."""
        if not self.webcam_active:
            self.start_webcam()
        else:
            self.stop_webcam()

    def start_webcam(self):
        """Start webcam capture."""
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.show_error("Could not open webcam")
            return

        self.webcam_active = True
        self.start_webcam_btn.configure(text="Stop Webcam", fg_color="red")
        self.save_frame_btn.configure(state="normal")

        # Start webcam thread
        self.webcam_thread = threading.Thread(target=self.webcam_loop, daemon=True)
        self.webcam_thread.start()

        # Start UI update
        self.update_webcam_display()

    def stop_webcam(self):
        """Stop webcam capture."""
        self.webcam_active = False

        if self.cap:
            self.cap.release()

        self.start_webcam_btn.configure(text="Start Webcam", fg_color="green")
        self.save_frame_btn.configure(state="disabled")

    def webcam_loop(self):
        """Webcam capture loop running in separate thread."""
        fps_start = time.time()
        frame_count = 0

        while self.webcam_active:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_start)
                fps_start = time.time()

                # Update FPS display
                self.window.after(0, lambda f=fps: self.fps_label.configure(text=f"FPS: {f:.1f}"))

            # Predict
            pred_class, confidence, top5, bbox = self.predict_image(frame)

            # Convert frame for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Draw bounding box if available
            if bbox is not None:
                x1, y1, x2, y2 = bbox.astype(int)
                # Draw on RGB frame
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Add label background
                label = f"{pred_class}: {confidence:.2%}"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )

                # Draw label background
                cv2.rectangle(
                    frame_rgb,
                    (x1, y1 - label_height - 10),
                    (x1 + label_width + 10, y1),
                    (0, 255, 0),
                    -1
                )

                # Draw label text
                cv2.putText(
                    frame_rgb,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )

            # Add to queue (non-blocking)
            if not self.frame_queue.full():
                try:
                    self.frame_queue.put_nowait((frame_rgb, frame, pred_class, confidence, top5))
                except queue.Full:
                    pass

            time.sleep(0.01)  # Small delay to prevent CPU overload

    def update_webcam_display(self):
        """Update the webcam display (runs in main thread)."""
        if not self.webcam_active:
            return

        try:
            frame_rgb, original_frame, pred_class, confidence, top5 = self.frame_queue.get_nowait()

            # Store current frame for saving
            self.current_webcam_frame = original_frame

            # Resize frame for display
            height, width = frame_rgb.shape[:2]
            max_width = 800
            max_height = 600

            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)

            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))

            # Convert to PhotoImage
            img = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=img)

            # Update display
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo

            # Update predictions
            if pred_class and confidence:
                self.pred_class_label.configure(text=f"Class: {pred_class}")
                self.confidence_label.configure(text=f"Confidence: {confidence:.2%}")

                # Update top-5
                if top5:
                    for i, (cls, conf) in enumerate(top5):
                        self.top5_labels[i].configure(text=f"{i+1}. {cls}: {conf:.1%}")

        except queue.Empty:
            pass

        # Schedule next update
        self.window.after(30, self.update_webcam_display)

    def save_webcam_frame(self):
        """Save the current webcam frame."""
        if hasattr(self, 'current_webcam_frame'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"webcam_capture_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_webcam_frame)
            self.show_info(f"Frame saved as: {filename}")

    def toggle_top5_predictions(self):
        """Toggle visibility of top-5 predictions."""
        self.show_top5 = not self.show_top5

        # Update visibility of top-5 frames
        if hasattr(self, 'top5_frame'):
            if self.show_top5:
                self.top5_frame.pack(pady=10, padx=20, fill="x")
            else:
                self.top5_frame.pack_forget()

        if hasattr(self, 'upload_top5_frame'):
            if self.show_top5:
                self.upload_top5_frame.pack(pady=10, padx=20, fill="x")
            else:
                self.upload_top5_frame.pack_forget()

        # Update button text for all toggle buttons
        button_text = "Show Top-5" if not self.show_top5 else "Hide Top-5"

        if hasattr(self, 'toggle_top5_btn'):
            self.toggle_top5_btn.configure(text=button_text)

        if hasattr(self, 'upload_toggle_top5_btn'):
            self.upload_toggle_top5_btn.configure(text=button_text)

    def upload_and_classify(self):
        """Upload and classify a single image."""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        # Load original image with OpenCV for drawing
        img_cv = cv2.imread(file_path)
        img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        # Classify and get bounding box
        pred_class, confidence, top5, bbox = self.predict_image(file_path)

        # Draw bounding box if available
        if bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            # Draw on RGB image
            cv2.rectangle(img_cv_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Add label background
            label = f"{pred_class}: {confidence:.2%}"

            # Calculate label size with proper font scale
            font_scale = max(0.7, min(img_cv_rgb.shape[1] / 1000, 1.5))
            thickness = max(2, int(img_cv_rgb.shape[1] / 500))

            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Draw label background
            cv2.rectangle(
                img_cv_rgb,
                (x1, y1 - label_height - 10),
                (x1 + label_width + 10, y1),
                (0, 255, 0),
                -1
            )

            # Draw label text
            cv2.putText(
                img_cv_rgb,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness
            )

        # Convert to PIL for display
        img_pil = Image.fromarray(img_cv_rgb)

        # Resize for display
        display_size = (700, 500)
        img_pil.thumbnail(display_size, Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(img_pil)
        self.upload_image_label.configure(image=photo, text="")
        self.upload_image_label.image = photo

        # Update results
        if pred_class and confidence:
            self.upload_pred_label.configure(text=f"Class: {pred_class}")
            self.upload_conf_label.configure(text=f"Confidence: {confidence:.2%}")

            # Update top-5
            for i, (cls, conf) in enumerate(top5):
                self.upload_top5_labels[i].configure(text=f"{i+1}. {cls}: {conf:.1%}")
        else:
            self.upload_pred_label.configure(text="No detection found")
            self.upload_conf_label.configure(text="Confidence: -")

    def batch_process_folder(self):
        """Process all images in a folder."""
        folder_path = filedialog.askdirectory(title="Select folder with images")

        if not folder_path:
            return

        folder_path = Path(folder_path)

        # Clear previous results
        for widget in self.batch_results_frame.winfo_children():
            widget.destroy()

        # Find images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder_path.glob(ext))
            image_files.extend(folder_path.glob(ext.upper()))

        if not image_files:
            self.show_error("No images found in folder")
            return

        self.batch_status_label.configure(text=f"Processing {len(image_files)} images...")

        # Process in separate thread
        thread = threading.Thread(
            target=self.process_batch_thread,
            args=(image_files,),
            daemon=True
        )
        thread.start()

    def process_batch_thread(self, image_files):
        """Process batch of images in separate thread."""
        results_list = []

        for i, img_path in enumerate(image_files):
            pred_class, confidence, _, _ = self.predict_image(str(img_path))

            # Only add to results if detection was made
            if pred_class and confidence:
                results_list.append((img_path.name, pred_class, confidence))

                # Add result to UI
                self.window.after(
                    0,
                    lambda name=img_path.name, cls=pred_class, conf=confidence:
                    self.add_batch_result(name, cls, conf)
                )
            else:
                # Add no detection result
                self.window.after(
                    0,
                    lambda name=img_path.name:
                    self.add_batch_result(name, "No detection", 0.0)
                )

            # Update progress
            progress_text = f"Processed {i+1}/{len(image_files)} images"
            self.window.after(0, lambda t=progress_text: self.batch_status_label.configure(text=t))

        # Show summary
        if results_list:
            avg_conf = np.mean([r[2] for r in results_list])
            summary_text = f"Completed! Processed {len(results_list)} images. Avg confidence: {avg_conf:.2%}"
        else:
            summary_text = f"Completed! Processed {len(image_files)} images. No detections found."
        self.window.after(0, lambda t=summary_text: self.batch_status_label.configure(text=t))

    def add_batch_result(self, filename, pred_class, confidence):
        """Add a result to the batch results display."""
        result_frame = ctk.CTkFrame(self.batch_results_frame)
        result_frame.pack(pady=5, padx=10, fill="x")
        result_frame.grid_columnconfigure(1, weight=1)

        filename_label = ctk.CTkLabel(
            result_frame,
            text=filename,
            font=ctk.CTkFont(size=12),
            width=300,
            anchor="w"
        )
        filename_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        class_label = ctk.CTkLabel(
            result_frame,
            text=pred_class,
            font=ctk.CTkFont(size=12, weight="bold"),
            width=150,
            anchor="w"
        )
        class_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        conf_label = ctk.CTkLabel(
            result_frame,
            text=f"{confidence:.1%}",
            font=ctk.CTkFont(size=12),
            width=80,
            anchor="e"
        )
        conf_label.grid(row=0, column=2, padx=10, pady=5, sticky="e")

    def show_error(self, message):
        """Show an error message."""
        error_label = ctk.CTkLabel(
            self.content_frame,
            text=f"Error: {message}",
            font=ctk.CTkFont(size=16),
            text_color="red"
        )
        error_label.pack(pady=20)

    def show_info(self, message):
        """Show an info message."""
        # Create temporary toplevel window
        info_window = ctk.CTkToplevel(self.window)
        info_window.title("Info")
        info_window.geometry("400x150")

        label = ctk.CTkLabel(
            info_window,
            text=message,
            font=ctk.CTkFont(size=14)
        )
        label.pack(pady=20)

        ok_btn = ctk.CTkButton(
            info_window,
            text="OK",
            command=info_window.destroy,
            width=100
        )
        ok_btn.pack(pady=10)

    def run(self):
        """Run the application."""
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def on_closing(self):
        """Handle window closing."""
        self.webcam_active = False
        if self.cap:
            self.cap.release()
        self.window.destroy()


def main():
    """Main entry point."""
    app = OfficeItemsClassifierGUI()
    app.run()


if __name__ == "__main__":
    main()
