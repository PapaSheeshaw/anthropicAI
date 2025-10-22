"""
Download office item images from Google's Open Images v7 dataset using FiftyOne.

This script downloads images for 7 office-related classes:
- Chair
- Computer keyboard
- Computer mouse
- Headphones
- Microphone
- Mobile phone
- Pen

Target: 300 images per class
"""

import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path
import shutil
import os
from tqdm import tqdm

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

# Download configuration
CLASSES = [
    "Chair",
    "Computer keyboard",
    "Computer mouse",
    "Headphones",
    "Microphone",
    "Mobile phone",
    "Pen"
]

TARGET_IMAGES_PER_CLASS = 300
MAX_TOTAL_SAMPLES = 10000
SPLIT = "train"


def download_office_items(selected_classes=None):
    """
    Download images using FiftyOne Zoo API.

    Args:
        selected_classes: List of class names to download. If None, downloads all CLASSES.

    Returns:
        Path to output directory
    """
    classes_to_download = selected_classes or CLASSES

    print(f"\n{'='*60}")
    print("DOWNLOADING OFFICE ITEMS DATASET")
    print(f"{'='*60}")
    print(f"Classes: {', '.join(classes_to_download)}")
    print(f"Target images per class: {TARGET_IMAGES_PER_CLASS}")
    print(f"Max samples to process: {MAX_TOTAL_SAMPLES}")
    print(f"{'='*60}\n")

    # Create output directories
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    for class_name in classes_to_download:
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
        print(f"✓ Created directory: {class_dir}")

    print("\n" + "="*60)
    print("LOADING OPEN IMAGES DATASET")
    print("="*60)
    print("This may take several minutes on first run...")
    print("FiftyOne will download and cache the dataset metadata.\n")

    # Load dataset from FiftyOne
    try:
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=SPLIT,
            label_types=["detections"],
            classes=classes_to_download,
            max_samples=MAX_TOTAL_SAMPLES,
            dataset_name="office_items_dataset",
            only_matching=True,
        )

        print(f"\n✓ Loaded {len(dataset)} samples from Open Images v7")

    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        return None

    # Export images by class
    print("\n" + "="*60)
    print("EXPORTING IMAGES BY CLASS")
    print("="*60 + "\n")

    exported_counts = {cls: 0 for cls in classes_to_download}
    skipped_no_file = 0
    skipped_class_full = 0

    for sample in tqdm(dataset, desc="Processing samples", unit="img"):
        # Check if sample has ground truth detections
        if not hasattr(sample, 'ground_truth') or not sample.ground_truth:
            continue

        # Get all target labels in this image
        target_labels = [
            det.label for det in sample.ground_truth.detections
            if det.label in classes_to_download
        ]

        if not target_labels:
            continue

        # Use the first matching label as primary class
        primary_label = target_labels[0]

        # Check if we've reached the target for this class
        if exported_counts[primary_label] >= TARGET_IMAGES_PER_CLASS:
            skipped_class_full += 1
            continue

        # Copy image to class folder
        if sample.filepath and os.path.exists(sample.filepath):
            src_path = Path(sample.filepath)
            dst_filename = f"{primary_label}_{exported_counts[primary_label]:04d}{src_path.suffix}"
            dst_path = output_path / primary_label / dst_filename

            try:
                shutil.copy2(src_path, dst_path)
                exported_counts[primary_label] += 1
            except Exception as e:
                print(f"\n✗ Error copying {src_path}: {e}")
        else:
            skipped_no_file += 1

    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)

    total_exported = sum(exported_counts.values())

    print(f"\nTotal images exported: {total_exported}")
    print(f"Skipped (class full): {skipped_class_full}")
    print(f"Skipped (file not found): {skipped_no_file}")

    print("\nImages per class:")
    for class_name in sorted(classes_to_download):
        count = exported_counts[class_name]
        status = "✓" if count >= TARGET_IMAGES_PER_CLASS * 0.8 else "⚠"
        print(f"  {status} {class_name:20s}: {count:4d} / {TARGET_IMAGES_PER_CLASS}")

    print(f"\n✓ Images saved to: {output_path}")
    print("="*60 + "\n")

    # Clean up FiftyOne dataset to save disk space
    try:
        dataset.delete()
        print("✓ Cleaned up FiftyOne dataset cache\n")
    except:
        pass

    return output_path


def main():
    """Main execution function."""
    try:
        output_path = download_office_items()

        if output_path:
            print("\n" + "="*60)
            print("NEXT STEPS")
            print("="*60)
            print("\n1. Review downloaded images in:")
            print(f"   {output_path}")
            print("\n2. Run data cleaning:")
            print("   python src/clean_data.py")
            print("\n" + "="*60 + "\n")
            return 0
        else:
            print("\n✗ Download failed. Please check the error messages above.\n")
            return 1

    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user.")
        print("You can safely re-run this script to resume downloading.\n")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
