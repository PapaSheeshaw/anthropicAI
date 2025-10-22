"""
Train YOLOv8 classification model on office items dataset.

This script:
1. Loads preprocessed data
2. Configures YOLOv8 model
3. Trains with augmentation and optimization
4. Validates performance
5. Saves best model
"""

import torch
from pathlib import Path
import time
from ultralytics import YOLO
import json

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Training configuration
CONFIG = {
    # Model
    'model_name': 'yolov8m-cls.pt',  # Options: n, s, m, l, x (nano to extra-large)
    'epochs': 30,
    'batch_size': 32,
    'imgsz': 224,
    'device': 0 if torch.cuda.is_available() else 'cpu',

    # Optimizer
    'optimizer': 'AdamW',
    'lr0': 0.001,          # Initial learning rate
    'lrf': 0.01,           # Final learning rate (lr0 * lrf)
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'cos_lr': True,        # Cosine learning rate scheduler

    # Augmentation
    'hsv_h': 0.015,        # HSV-Hue augmentation
    'hsv_s': 0.7,          # HSV-Saturation augmentation
    'hsv_v': 0.4,          # HSV-Value augmentation
    'degrees': 15.0,       # Rotation augmentation (degrees)
    'translate': 0.1,      # Translation augmentation (fraction)
    'scale': 0.5,          # Scale augmentation (gain)
    'fliplr': 0.5,         # Horizontal flip probability

    # Training settings
    'patience': 10,        # Early stopping patience
    'workers': 4,          # Data loader workers
    'cache': False,        # Cache images to RAM (True for faster training if enough RAM)
    'pretrained': True,    # Use pretrained weights
}


def check_data_directory():
    """
    Verify that data directory exists and has correct structure.

    Returns:
        Tuple of (is_valid: bool, message: str, class_info: dict)
    """
    if not DATA_DIR.exists():
        return False, f"Data directory not found: {DATA_DIR}", {}

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "validation"

    if not train_dir.exists():
        return False, f"Training directory not found: {train_dir}", {}

    if not val_dir.exists():
        return False, f"Validation directory not found: {val_dir}", {}

    # Get class information
    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    val_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])

    if not train_classes:
        return False, "No class folders found in training directory", {}

    if train_classes != val_classes:
        return False, "Class mismatch between train and validation", {}

    # Count images per class
    class_info = {}
    for class_name in train_classes:
        train_count = len(list((train_dir / class_name).glob("*.jpg"))) + \
                     len(list((train_dir / class_name).glob("*.png")))
        val_count = len(list((val_dir / class_name).glob("*.jpg"))) + \
                   len(list((val_dir / class_name).glob("*.png")))

        class_info[class_name] = {
            'train': train_count,
            'validation': val_count
        }

    return True, "Data directory valid", class_info


def print_training_info(config, class_info):
    """Print training configuration and dataset information."""
    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {config['model_name']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Image size: {config['imgsz']}×{config['imgsz']}")
    print(f"Device: {config['device']}")
    print(f"Optimizer: {config['optimizer']}")
    print(f"Learning rate: {config['lr0']} → {config['lr0'] * config['lrf']}")
    print(f"Pretrained: {config['pretrained']}")
    print(f"{'='*60}\n")

    print(f"{'='*60}")
    print("DATASET INFORMATION")
    print(f"{'='*60}")
    print(f"Number of classes: {len(class_info)}")
    print(f"Classes: {', '.join(class_info.keys())}")
    print(f"\n{'Class':<20} {'Train':<8} {'Val':<8} {'Total':<8}")
    print("-" * 60)

    total_train = 0
    total_val = 0

    for class_name, counts in sorted(class_info.items()):
        train_count = counts['train']
        val_count = counts['validation']
        total = train_count + val_count

        print(f"{class_name:<20} {train_count:<8} {val_count:<8} {total:<8}")

        total_train += train_count
        total_val += val_count

    print("-" * 60)
    print(f"{'TOTAL':<20} {total_train:<8} {total_val:<8} {total_train + total_val:<8}")
    print(f"{'='*60}\n")


def train_yolo_model(model_name, data_dir, config):
    """
    Train YOLO classification model.

    IMPORTANT: Pass data_dir directly, not a YAML file!
    YOLO automatically looks for:
    - data_dir/train/
    - data_dir/validation/ (or data_dir/val/)

    Args:
        model_name: Name of YOLO model (e.g., 'yolov8n-cls.pt')
        data_dir: Path to data directory containing train/ and validation/
        config: Training configuration dictionary

    Returns:
        Tuple of (model, results)
    """
    print(f"{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Load model
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    print(f"✓ Model loaded successfully\n")

    # Train
    print("Training started...")
    print("="*60 + "\n")

    results = model.train(
        data=str(data_dir),  # ← Pass directory path directly!
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch_size'],
        device=config['device'],

        # Optimizer settings
        optimizer=config['optimizer'],
        lr0=config['lr0'],
        lrf=config['lrf'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        cos_lr=config['cos_lr'],

        # Augmentation
        hsv_h=config['hsv_h'],
        hsv_s=config['hsv_s'],
        hsv_v=config['hsv_v'],
        degrees=config['degrees'],
        translate=config['translate'],
        scale=config['scale'],
        fliplr=config['fliplr'],

        # Training settings
        patience=config['patience'],
        workers=config['workers'],
        cache=config['cache'],
        pretrained=config['pretrained'],

        # Save settings
        project=str(MODEL_DIR),
        name='yolo_training',
        exist_ok=True,
        verbose=True,
    )

    elapsed_time = time.time() - start_time

    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Total training time: {elapsed_time/60:.2f} minutes")
    print(f"{'='*60}\n")

    return model, results


def validate_model(model, data_dir):
    """
    Validate trained model.

    Args:
        model: Trained YOLO model
        data_dir: Path to data directory

    Returns:
        Validation results
    """
    print(f"{'='*60}")
    print("VALIDATING MODEL")
    print(f"{'='*60}\n")

    val_results = model.val(data=str(data_dir))

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Top-1 Accuracy: {val_results.top1:.4f} ({val_results.top1*100:.2f}%)")
    print(f"Top-5 Accuracy: {val_results.top5:.4f} ({val_results.top5*100:.2f}%)")
    print(f"{'='*60}\n")

    return val_results


def save_training_info(config, class_info, val_results):
    """Save training configuration and results to JSON file."""
    training_info = {
        'config': config,
        'dataset': {
            'num_classes': len(class_info),
            'classes': list(class_info.keys()),
            'class_counts': class_info
        },
        'results': {
            'top1_accuracy': float(val_results.top1),
            'top5_accuracy': float(val_results.top5)
        }
    }

    info_path = MODEL_DIR / "yolo_training" / "training_info.json"
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)

    print(f"✓ Training info saved to: {info_path}\n")


def main():
    """Main execution function."""
    try:
        # Check data directory
        is_valid, message, class_info = check_data_directory()

        if not is_valid:
            print(f"\n✗ Error: {message}")
            print("\nPlease run the data preparation scripts first:")
            print("  1. python src/download_data.py")
            print("  2. python src/clean_data.py")
            print("  3. python src/split_data.py\n")
            return 1

        # Print training info
        print_training_info(CONFIG, class_info)

        # Check for GPU
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
        else:
            print("⚠ No GPU available. Training will use CPU (slower).\n")

        # Confirm training
        print("="*60)
        print("Ready to start training!")
        print("="*60)
        print("\nThis will take approximately:")
        print("  - 15-30 minutes on GPU")
        print("  - 2-3 hours on CPU")
        print("\nPress Ctrl+C to cancel\n")

        # Small delay to allow user to cancel
        time.sleep(3)

        # Train model
        model, results = train_yolo_model(
            model_name=CONFIG['model_name'],
            data_dir=DATA_DIR,
            config=CONFIG
        )

        # Validate model
        val_results = validate_model(model, DATA_DIR)

        # Save training info
        save_training_info(CONFIG, class_info, val_results)

        # Print final information
        print("="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print("\nModel saved to:")
        print(f"  Best model: {MODEL_DIR / 'yolo_training' / 'weights' / 'best.pt'}")
        print(f"  Last model: {MODEL_DIR / 'yolo_training' / 'weights' / 'last.pt'}")
        print("\nResults saved to:")
        print(f"  {MODEL_DIR / 'yolo_training'}")
        print("\nNext steps:")
        print("  1. Review training curves: models/yolo_training/results.png")
        print("  2. Evaluate on test set: python src/evaluate_yolo.py")
        print("  3. Use model for inference:")
        print("     from ultralytics import YOLO")
        print("     model = YOLO('models/yolo_training/weights/best.pt')")
        print("     results = model.predict('image.jpg')")
        print("="*60 + "\n")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user.\n")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
