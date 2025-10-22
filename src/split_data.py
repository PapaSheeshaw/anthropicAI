"""
Split cleaned data into train/validation/test sets.

This script performs stratified splitting to ensure each class
is represented proportionally in all splits.

Split ratios:
- Train: 70%
- Validation: 15%
- Test: 15%
"""

import random
from pathlib import Path
import shutil
from collections import defaultdict

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent
CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Split configuration
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def split_dataset():
    """
    Perform stratified split of cleaned dataset.

    For each class:
    1. Shuffle images with fixed seed
    2. Split by ratios (70/15/15)
    3. Copy to train/validation/test directories

    Returns:
        Dictionary with split statistics
    """
    print(f"\n{'='*60}")
    print("SPLITTING DATASET")
    print(f"{'='*60}")
    print(f"Input directory: {CLEANED_DATA_DIR}")
    print(f"Output directory: {PROCESSED_DATA_DIR}")
    print(f"Split ratios: Train={TRAIN_RATIO:.0%}, Val={VALIDATION_RATIO:.0%}, Test={TEST_RATIO:.0%}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"{'='*60}\n")

    # Check if cleaned data exists
    if not CLEANED_DATA_DIR.exists():
        print(f"✗ Error: Cleaned data directory not found at {CLEANED_DATA_DIR}")
        print("\nPlease run clean_data.py first!")
        return None

    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)

    # Remove existing processed data
    if PROCESSED_DATA_DIR.exists():
        print("⚠ Removing existing processed data...")
        shutil.rmtree(PROCESSED_DATA_DIR)

    # Create split directories
    train_dir = PROCESSED_DATA_DIR / "train"
    val_dir = PROCESSED_DATA_DIR / "validation"
    test_dir = PROCESSED_DATA_DIR / "test"

    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)

    print("✓ Created split directories\n")

    # Initialize statistics
    stats = {
        'train': {},
        'validation': {},
        'test': {},
        'total': {}
    }

    # Process each class
    class_folders = sorted([d for d in CLEANED_DATA_DIR.iterdir() if d.is_dir()])

    if not class_folders:
        print("✗ No class folders found in cleaned data directory!")
        return None

    print(f"Found {len(class_folders)} class folders\n")

    for class_folder in class_folders:
        class_name = class_folder.name
        print(f"Processing: {class_name}")

        # Get all images
        # Use set to automatically handle duplicates from case-insensitive filesystem
        image_files = set()
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            # Glob returns Path objects
            for img in class_folder.glob(f"*{ext}"):
                image_files.add(img)
            # Also check uppercase extensions
            for img in class_folder.glob(f"*{ext.upper()}"):
                image_files.add(img)

        # Convert set to list for shuffling
        image_files = list(image_files)

        # Shuffle with fixed seed
        random.shuffle(image_files)

        total = len(image_files)
        stats['total'][class_name] = total

        if total == 0:
            print(f"  ⚠ No images found for {class_name}")
            continue

        # Calculate split sizes
        train_size = int(total * TRAIN_RATIO)
        val_size = int(total * VALIDATION_RATIO)
        # Test gets the remainder to ensure all images are used
        test_size = total - train_size - val_size

        # Split images
        train_images = image_files[:train_size]
        val_images = image_files[train_size:train_size + val_size]
        test_images = image_files[train_size + val_size:]

        # Create class folders in each split
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)

        # Copy files to train
        for img in train_images:
            dst = train_dir / class_name / img.name
            shutil.copy2(img, dst)

        # Copy files to validation
        for img in val_images:
            dst = val_dir / class_name / img.name
            shutil.copy2(img, dst)

        # Copy files to test
        for img in test_images:
            dst = test_dir / class_name / img.name
            shutil.copy2(img, dst)

        # Update statistics
        stats['train'][class_name] = len(train_images)
        stats['validation'][class_name] = len(val_images)
        stats['test'][class_name] = len(test_images)

        print(f"  Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

    # Print summary
    print(f"\n{'='*60}")
    print("SPLIT SUMMARY")
    print(f"{'='*60}\n")

    print(f"{'Class':<20} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
    print("-" * 60)

    total_all = 0
    total_train = 0
    total_val = 0
    total_test = 0

    for class_name in sorted(stats['total'].keys()):
        total = stats['total'][class_name]
        train = stats['train'].get(class_name, 0)
        val = stats['validation'].get(class_name, 0)
        test = stats['test'].get(class_name, 0)

        print(f"{class_name:<20} {total:<8} {train:<8} {val:<8} {test:<8}")

        total_all += total
        total_train += train
        total_val += val
        total_test += test

    print("-" * 60)
    print(f"{'TOTAL':<20} {total_all:<8} {total_train:<8} {total_val:<8} {total_test:<8}")

    print(f"\nSplit percentages:")
    print(f"  Train:      {total_train/total_all*100:.1f}%")
    print(f"  Validation: {total_val/total_all*100:.1f}%")
    print(f"  Test:       {total_test/total_all*100:.1f}%")

    print(f"\n✓ Processed data saved to: {PROCESSED_DATA_DIR}")
    print(f"{'='*60}\n")

    return stats


def verify_split():
    """
    Verify the split was successful.

    Checks:
    - All directories exist
    - All classes are present in each split
    - No file overlaps between splits
    """
    print("Verifying split...")

    train_dir = PROCESSED_DATA_DIR / "train"
    val_dir = PROCESSED_DATA_DIR / "validation"
    test_dir = PROCESSED_DATA_DIR / "test"

    # Check directories exist
    if not all([train_dir.exists(), val_dir.exists(), test_dir.exists()]):
        print("✗ Split directories not found!")
        return False

    # Get class names from train
    train_classes = set([d.name for d in train_dir.iterdir() if d.is_dir()])
    val_classes = set([d.name for d in val_dir.iterdir() if d.is_dir()])
    test_classes = set([d.name for d in test_dir.iterdir() if d.is_dir()])

    # Check all classes present in all splits
    if not (train_classes == val_classes == test_classes):
        print("✗ Class mismatch between splits!")
        print(f"  Train: {train_classes}")
        print(f"  Val: {val_classes}")
        print(f"  Test: {test_classes}")
        return False

    # Check for file overlaps (just filenames, not full paths)
    train_files = set()
    val_files = set()
    test_files = set()

    for class_name in train_classes:
        train_files.update([f.name for f in (train_dir / class_name).glob("*.*")])
        val_files.update([f.name for f in (val_dir / class_name).glob("*.*")])
        test_files.update([f.name for f in (test_dir / class_name).glob("*.*")])

    overlap_train_val = train_files & val_files
    overlap_train_test = train_files & test_files
    overlap_val_test = val_files & test_files

    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("✗ File overlaps detected between splits!")
        return False

    print("✓ Split verification passed")
    return True


def main():
    """Main execution function."""
    try:
        stats = split_dataset()

        if stats:
            # Verify split
            if verify_split():
                print("\n" + "="*60)
                print("NEXT STEPS")
                print("="*60)
                print("\n1. Review split data in:")
                print(f"   {PROCESSED_DATA_DIR}")
                print("\n2. Train YOLO model:")
                print("   python src/train.py")
                print("\n" + "="*60 + "\n")
                return 0
            else:
                print("\n✗ Split verification failed!\n")
                return 1
        else:
            print("\n✗ Splitting failed. Please check the error messages above.\n")
            return 1

    except KeyboardInterrupt:
        print("\n\n⚠ Splitting interrupted by user.\n")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
