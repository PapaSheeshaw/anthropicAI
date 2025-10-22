"""
Clean and preprocess raw images for training.

This script:
1. Validates images (format, dimensions, aspect ratio)
2. Detects and removes duplicates
3. Applies preprocessing (CLAHE, denoising, sharpening)
4. Resizes to standard size (224x224)
5. Saves cleaned images
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import hashlib
from collections import defaultdict
from pathlib import Path
import shutil
from tqdm import tqdm

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"

# Validation settings
MIN_WIDTH = 100
MIN_HEIGHT = 100
MAX_ASPECT_RATIO = 5.0
VALID_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Preprocessing settings
APPLY_CLAHE = True
APPLY_SHARPENING = True
APPLY_NOISE_REDUCTION = True
RESIZE_TO_STANDARD = True

STANDARD_WIDTH = 224
STANDARD_HEIGHT = 224

# Quality thresholds
MIN_BRIGHTNESS = 20
MAX_BRIGHTNESS = 235
MIN_VARIANCE = 100


def is_valid_image(image_path):
    """
    Validate image file.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (is_valid: bool, reason: str)
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size

            # Check file format
            if Path(image_path).suffix.lower() not in VALID_FORMATS:
                return False, "Invalid format"

            # Check dimensions
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return False, f"Too small: {width}×{height}"

            # Check aspect ratio
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > MAX_ASPECT_RATIO:
                return False, f"Bad aspect ratio: {aspect_ratio:.1f}:1"

            # Test RGB conversion
            img.convert('RGB')

            return True, "Valid"

    except Exception as e:
        return False, f"Corrupted: {str(e)[:50]}"


def preprocess_image(image_path):
    """
    Preprocess image with quality enhancement.

    Pipeline:
    1. Quality assessment (brightness, variance)
    2. CLAHE (contrast enhancement in LAB space)
    3. Noise reduction (non-local means denoising)
    4. Sharpening (unsharp mask)
    5. Resize to 224×224 (LANCZOS)

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (processed_image: PIL.Image, success: bool, message: str)
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Quality assessment
        gray = np.array(image.convert('L'))
        mean_brightness = np.mean(gray)
        variance = np.var(gray)

        # Check brightness
        if mean_brightness < MIN_BRIGHTNESS or mean_brightness > MAX_BRIGHTNESS:
            return None, False, f"Bad brightness: {mean_brightness:.1f}"

        # Check detail/sharpness
        if variance < MIN_VARIANCE:
            return None, False, f"Low detail/blurry: var={variance:.1f}"

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if APPLY_CLAHE:
            img_array = np.array(image)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            lab = cv2.merge([l, a, b])
            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            image = Image.fromarray(img_array)

        # Noise reduction
        if APPLY_NOISE_REDUCTION:
            img_array = np.array(image)
            denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 6, 6, 7, 21)
            image = Image.fromarray(denoised)

        # Sharpening
        if APPLY_SHARPENING:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)

        # Resize to standard size
        if RESIZE_TO_STANDARD:
            image = image.resize(
                (STANDARD_WIDTH, STANDARD_HEIGHT),
                Image.Resampling.LANCZOS
            )

        return image, True, "Preprocessed"

    except Exception as e:
        return None, False, f"Processing error: {str(e)[:50]}"


def clean_dataset():
    """
    Clean and preprocess entire dataset.

    Returns:
        Dictionary with cleaning statistics per class
    """
    print(f"\n{'='*60}")
    print("CLEANING DATASET")
    print(f"{'='*60}")
    print(f"Input directory: {RAW_DATA_DIR}")
    print(f"Output directory: {CLEANED_DATA_DIR}")
    print(f"Target size: {STANDARD_WIDTH}×{STANDARD_HEIGHT}")
    print(f"{'='*60}\n")

    # Check if raw data exists
    if not RAW_DATA_DIR.exists():
        print(f"✗ Error: Raw data directory not found at {RAW_DATA_DIR}")
        print("\nPlease run download_data.py first!")
        return None

    # Remove existing cleaned data
    if CLEANED_DATA_DIR.exists():
        print("⚠ Removing existing cleaned data...")
        shutil.rmtree(CLEANED_DATA_DIR)

    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize tracking
    seen_hashes = set()
    stats = defaultdict(lambda: {
        'found': 0,
        'valid': 0,
        'invalid': 0,
        'duplicates': 0,
        'low_quality': 0
    })

    # Process each class
    class_folders = sorted([d for d in RAW_DATA_DIR.iterdir() if d.is_dir()])

    if not class_folders:
        print("✗ No class folders found in raw data directory!")
        return None

    print(f"Found {len(class_folders)} class folders\n")

    for class_folder in class_folders:
        class_name = class_folder.name
        print(f"\n{'='*60}")
        print(f"Processing: {class_name}")
        print(f"{'='*60}")

        # Create cleaned class folder
        cleaned_class_folder = CLEANED_DATA_DIR / class_name
        cleaned_class_folder.mkdir(exist_ok=True)

        # Get all image files
        # Use set to automatically handle duplicates from case-insensitive filesystem
        image_files = set()
        for ext in VALID_FORMATS:
            # Glob returns Path objects
            for img in class_folder.glob(f"*{ext}"):
                image_files.add(img)
            # Also check uppercase extensions
            for img in class_folder.glob(f"*{ext.upper()}"):
                image_files.add(img)

        # Convert set to list for processing
        image_files = list(image_files)

        stats[class_name]['found'] = len(image_files)
        print(f"Found {len(image_files)} images")

        if len(image_files) == 0:
            print("⚠ No images found in this class!")
            continue

        valid_count = 0

        # Process each image
        for img_path in tqdm(image_files, desc=f"  Cleaning {class_name}", unit="img"):
            # Validate
            is_valid, reason = is_valid_image(img_path)
            if not is_valid:
                stats[class_name]['invalid'] += 1
                continue

            # Check for duplicates using MD5 hash
            try:
                with open(img_path, 'rb') as f:
                    img_hash = hashlib.md5(f.read()).hexdigest()

                if img_hash in seen_hashes:
                    stats[class_name]['duplicates'] += 1
                    continue

                seen_hashes.add(img_hash)
            except Exception as e:
                stats[class_name]['invalid'] += 1
                continue

            # Preprocess
            processed_img, ok, reason = preprocess_image(img_path)
            if not ok:
                stats[class_name]['low_quality'] += 1
                continue

            # Save
            new_name = f"{class_name}_{valid_count:04d}.jpg"
            new_path = cleaned_class_folder / new_name

            try:
                processed_img.save(new_path, 'JPEG', quality=95)
                valid_count += 1
                stats[class_name]['valid'] = valid_count
            except Exception as e:
                print(f"\n✗ Error saving {new_path}: {e}")
                continue

        # Print class summary
        print(f"\n  Results:")
        print(f"    ✓ Valid:       {stats[class_name]['valid']}")
        print(f"    ✗ Invalid:     {stats[class_name]['invalid']}")
        print(f"    ✗ Duplicates:  {stats[class_name]['duplicates']}")
        print(f"    ✗ Low quality: {stats[class_name]['low_quality']}")

    # Print overall summary
    print(f"\n{'='*60}")
    print("CLEANING SUMMARY")
    print(f"{'='*60}\n")

    print(f"{'Class':<20} {'Found':<8} {'Valid':<8} {'Invalid':<8} {'Dup':<8} {'LowQ':<8}")
    print("-" * 60)

    total_found = 0
    total_valid = 0
    total_invalid = 0
    total_dup = 0
    total_low_q = 0

    for class_name in sorted(stats.keys()):
        s = stats[class_name]
        print(f"{class_name:<20} {s['found']:<8} {s['valid']:<8} "
              f"{s['invalid']:<8} {s['duplicates']:<8} {s['low_quality']:<8}")

        total_found += s['found']
        total_valid += s['valid']
        total_invalid += s['invalid']
        total_dup += s['duplicates']
        total_low_q += s['low_quality']

    print("-" * 60)
    print(f"{'TOTAL':<20} {total_found:<8} {total_valid:<8} "
          f"{total_invalid:<8} {total_dup:<8} {total_low_q:<8}")

    print(f"\n✓ Cleaned images saved to: {CLEANED_DATA_DIR}")
    print(f"✓ Total valid images: {total_valid}")
    print(f"✓ Retention rate: {total_valid/total_found*100:.1f}%")

    print(f"\n{'='*60}\n")

    return stats


def main():
    """Main execution function."""
    try:
        stats = clean_dataset()

        if stats:
            print("="*60)
            print("NEXT STEPS")
            print("="*60)
            print("\n1. Review cleaned images in:")
            print(f"   {CLEANED_DATA_DIR}")
            print("\n2. Run data splitting:")
            print("   python src/split_data.py")
            print("\n" + "="*60 + "\n")
            return 0
        else:
            print("\n✗ Cleaning failed. Please check the error messages above.\n")
            return 1

    except KeyboardInterrupt:
        print("\n\n⚠ Cleaning interrupted by user.\n")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
