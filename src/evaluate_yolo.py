"""
Evaluate trained YOLO model on test set.

This script:
1. Loads trained model
2. Runs inference on test set
3. Computes comprehensive metrics
4. Generates visualization plots
"""

from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
from pathlib import Path
import json
from tqdm import tqdm

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_model(model_path):
    """
    Load trained YOLO model.

    Args:
        model_path: Path to model weights file

    Returns:
        YOLO model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    print(f"✓ Model loaded successfully\n")

    return model


def evaluate_yolo_model(model, test_dir, class_names):
    """
    Evaluate YOLO model on test set.

    Args:
        model: Trained YOLO model
        test_dir: Path to test directory
        class_names: List of class names

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"{'='*60}")
    print("EVALUATING MODEL ON TEST SET")
    print(f"{'='*60}\n")

    all_preds = []
    all_labels = []
    all_probs = []
    all_confidences = []

    # Process each class
    for class_idx, class_name in enumerate(class_names):
        class_dir = test_dir / class_name

        if not class_dir.exists():
            print(f"⚠ Warning: Class directory not found: {class_dir}")
            continue

        # Get all image files
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

        if len(image_files) == 0:
            print(f"⚠ Warning: No images found for class {class_name}")
            continue

        print(f"Processing {class_name}: {len(image_files)} images")

        # Predict on all images
        for img_path in tqdm(image_files, desc=f"  {class_name}", leave=False):
            try:
                # Predict
                results = model.predict(str(img_path), verbose=False)

                # Get prediction
                probs = results[0].probs
                pred_class = probs.top1
                pred_conf = probs.top1conf.cpu().numpy()
                pred_probs = probs.data.cpu().numpy()

                all_preds.append(pred_class)
                all_labels.append(class_idx)
                all_probs.append(pred_probs)
                all_confidences.append(pred_conf)

            except Exception as e:
                print(f"\n✗ Error processing {img_path}: {e}")
                continue

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_confidences = np.array(all_confidences)

    print(f"\n✓ Processed {len(all_labels)} images\n")

    # Compute metrics
    print("Computing metrics...")

    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # Top-5 accuracy
    top5_correct = 0
    for i in range(len(all_labels)):
        top5_preds = np.argsort(all_probs[i])[-5:]
        if all_labels[i] in top5_preds:
            top5_correct += 1
    top5_accuracy = top5_correct / len(all_labels)

    # Mean confidence
    mean_confidence = np.mean(all_confidences)

    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm,
        'mean_confidence': mean_confidence,
        'predictions': all_preds,
        'labels': all_labels,
        'confidences': all_confidences
    }


def print_results(results, class_names):
    """
    Print evaluation results.

    Args:
        results: Dictionary with evaluation metrics
        class_names: List of class names
    """
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}\n")

    # Overall metrics
    print("Overall Metrics:")
    print(f"  Top-1 Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Top-5 Accuracy: {results['top5_accuracy']:.4f} ({results['top5_accuracy']*100:.2f}%)")
    print(f"  Mean Confidence: {results['mean_confidence']:.4f}")

    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Samples':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print("-" * 80)

    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} "
              f"{results['support'][i]:<8} "
              f"{results['per_class_accuracy'][i]:.4f}  "
              f"{results['precision'][i]:.4f}  "
              f"{results['recall'][i]:.4f}  "
              f"{results['f1'][i]:.4f}")

    # Average metrics
    print("-" * 80)
    avg_precision = np.mean(results['precision'])
    avg_recall = np.mean(results['recall'])
    avg_f1 = np.mean(results['f1'])
    print(f"{'AVERAGE':<20} "
          f"{np.sum(results['support']):<8} "
          f"{results['accuracy']:.4f}  "
          f"{avg_precision:.4f}  "
          f"{avg_recall:.4f}  "
          f"{avg_f1:.4f}")

    print(f"\n{'='*60}\n")


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot normalized confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 10))

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Accuracy'},
        vmin=0,
        vmax=1
    )

    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Confusion matrix saved to: {save_path}")


def plot_per_class_metrics(results, class_names, save_path):
    """
    Plot per-class accuracy, precision, recall, and F1 score.

    Args:
        results: Dictionary with evaluation metrics
        class_names: List of class names
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(class_names))
    width = 0.2

    # Plot bars
    ax.bar(x - 1.5*width, results['per_class_accuracy'], width, label='Accuracy', color='steelblue')
    ax.bar(x - 0.5*width, results['precision'], width, label='Precision', color='seagreen')
    ax.bar(x + 0.5*width, results['recall'], width, label='Recall', color='coral')
    ax.bar(x + 1.5*width, results['f1'], width, label='F1 Score', color='mediumpurple')

    # Customize plot
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Per-class metrics plot saved to: {save_path}")


def plot_confidence_distribution(confidences, save_path):
    """
    Plot confidence score distribution.

    Args:
        confidences: Array of confidence scores
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(confidences, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(confidences), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')

    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Prediction Confidence Distribution', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Confidence distribution plot saved to: {save_path}")


def save_results_json(results, class_names, save_path):
    """
    Save evaluation results to JSON file.

    Args:
        results: Dictionary with evaluation metrics
        class_names: List of class names
        save_path: Path to save JSON file
    """
    results_dict = {
        'overall': {
            'top1_accuracy': float(results['accuracy']),
            'top5_accuracy': float(results['top5_accuracy']),
            'mean_confidence': float(results['mean_confidence'])
        },
        'per_class': {}
    }

    for i, class_name in enumerate(class_names):
        results_dict['per_class'][class_name] = {
            'samples': int(results['support'][i]),
            'accuracy': float(results['per_class_accuracy'][i]),
            'precision': float(results['precision'][i]),
            'recall': float(results['recall'][i]),
            'f1_score': float(results['f1'][i])
        }

    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"✓ Results saved to: {save_path}")


def main():
    """Main execution function."""
    try:
        # Check if model exists
        model_path = MODEL_DIR / "yolo_training" / "weights" / "best.pt"

        if not model_path.exists():
            print(f"\n✗ Error: Model not found at {model_path}")
            print("\nPlease train the model first:")
            print("  python src/train.py\n")
            return 1

        # Check if test data exists
        test_dir = DATA_DIR / "test"

        if not test_dir.exists():
            print(f"\n✗ Error: Test directory not found at {test_dir}")
            print("\nPlease run data preparation first:")
            print("  python src/split_data.py\n")
            return 1

        # Load model
        model = load_model(model_path)

        # Get class names from model
        class_names = list(model.names.values())
        print(f"Classes: {', '.join(class_names)}\n")

        # Evaluate model
        results = evaluate_yolo_model(model, test_dir, class_names)

        # Print results
        print_results(results, class_names)

        # Generate plots
        print("Generating visualizations...")

        # Confusion matrix
        cm_path = RESULTS_DIR / "confusion_matrix.png"
        plot_confusion_matrix(results['confusion_matrix'], class_names, cm_path)

        # Per-class metrics
        metrics_path = RESULTS_DIR / "per_class_metrics.png"
        plot_per_class_metrics(results, class_names, metrics_path)

        # Confidence distribution
        conf_path = RESULTS_DIR / "confidence_distribution.png"
        plot_confidence_distribution(results['confidences'], conf_path)

        # Save results to JSON
        json_path = RESULTS_DIR / "evaluation_results.json"
        save_results_json(results, class_names, json_path)

        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*60}")
        print("\nResults saved to:")
        print(f"  {RESULTS_DIR}")
        print("\nGenerated files:")
        print(f"  - confusion_matrix.png")
        print(f"  - per_class_metrics.png")
        print(f"  - confidence_distribution.png")
        print(f"  - evaluation_results.json")
        print(f"\n{'='*60}\n")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user.\n")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

