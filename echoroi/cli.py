"""Command-line interface for EchoROI."""

import argparse
import os
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="EchoROI – U-Net echocardiographic ROI segmentation & de-identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  echoroi train --image-dir data/images --mask-dir data/masks --model-path models/my_model.keras

  # Make predictions on images
  echoroi predict --model-path models/echoroi.keras --input data/test_image.png --output results/

  # Evaluate model performance
  echoroi evaluate --model-path models/echoroi.keras --image-dir data/val_images --mask-dir data/val_masks

  # Create sample data for testing
  echoroi create-data --output-dir sample_data --num-samples 10
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new U-Net model')
    train_parser.add_argument('--image-dir', required=True, help='Directory containing training images')
    train_parser.add_argument('--mask-dir', required=True, help='Directory containing training masks')
    train_parser.add_argument('--model-path', default='models/unet_model.keras', help='Path to save trained model')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split fraction')
    train_parser.add_argument('--img-size', type=int, nargs=2, default=[256, 256], help='Input image size (height width)')
    train_parser.add_argument('--results-dir', default='training_results', help='Directory for training artifacts')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions on images')
    predict_parser.add_argument('--model-path', required=True, help='Path to trained model')
    predict_parser.add_argument('--input', required=True, help='Input image path or directory')
    predict_parser.add_argument('--output', required=True, help='Output directory for results')
    predict_parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    predict_parser.add_argument('--visualize', action='store_true', help='Create visualization plots')
    predict_parser.add_argument('--extract-roi', action='store_true', help='Extract and save ROI crops')
    predict_parser.add_argument('--deidentify', action='store_true', help='Create deidentified images')

    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    evaluate_parser.add_argument('--model-path', required=True, help='Path to trained model')
    evaluate_parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    evaluate_parser.add_argument('--mask-dir', required=True, help='Directory containing test masks')
    evaluate_parser.add_argument('--output', help='Output directory for evaluation results')
    evaluate_parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')

    # Create data command
    create_data_parser = subparsers.add_parser('create-data', help='Create sample synthetic data')
    create_data_parser.add_argument('--output-dir', default='sample_data', help='Output directory for sample data')
    create_data_parser.add_argument('--num-samples', type=int, default=10, help='Number of sample images to create')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark model inference speed')
    benchmark_parser.add_argument('--model-path', required=True, help='Path to trained model')
    benchmark_parser.add_argument('--image-path', required=True, help='Test image path')
    benchmark_parser.add_argument('--num-runs', type=int, default=10, help='Number of inference runs')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Execute the appropriate command
    try:
        if args.command == 'train':
            train_cli(args)
        elif args.command == 'predict':
            predict_cli(args)
        elif args.command == 'evaluate':
            evaluate_cli(args)
        elif args.command == 'create-data':
            create_data_cli(args)
        elif args.command == 'benchmark':
            benchmark_cli(args)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def train_cli(args):
    """CLI for training command."""
    from .training import UNetTrainer

    print("="*60)
    print("EchoROI - TRAINING")
    print("="*60)

    results_dir = args.results_dir

    # Initialize trainer
    trainer = UNetTrainer(
        img_size=tuple(args.img_size),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split
    )

    # Train model
    trainer.train(
        args.image_dir, args.mask_dir, args.model_path,
        results_dir=results_dir
    )

    # Save all training artifacts
    metrics = trainer.save_results(results_dir)

    print("\nTraining completed successfully!")
    print(f"Model saved to: {args.model_path}")
    print(f"Training artifacts saved to: {results_dir}/")
    print(f"  Final Dice: {metrics['dice']:.4f}  |  IoU: {metrics['iou']:.4f}")


def predict_cli(args):
    """CLI for prediction command."""
    from glob import glob

    import cv2

    from .inference import UNetPredictor, save_prediction_results

    print("="*60)
    print("EchoROI - PREDICTION")
    print("="*60)

    # Initialize predictor
    predictor = UNetPredictor(args.model_path)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Get input files
    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        image_paths = glob(os.path.join(args.input, "*.png")) + glob(os.path.join(args.input, "*.jpg"))
        if not image_paths:
            raise ValueError(f"No image files found in {args.input}")
    else:
        raise ValueError(f"Input path does not exist: {args.input}")

    print(f"Processing {len(image_paths)} images...")

    # Process each image
    for i, image_path in enumerate(image_paths, 1):
        print(f"Processing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")

        try:
            if args.visualize:
                # Create comprehensive visualization
                vis_path = os.path.join(args.output, f"{os.path.splitext(os.path.basename(image_path))[0]}_visualization.png")
                results = predictor.process_image_with_visualization(image_path, args.threshold, vis_path)

                # Save individual results if requested
                if args.extract_roi:
                    roi_path = os.path.join(args.output, f"{os.path.splitext(os.path.basename(image_path))[0]}_roi.png")
                    cv2.imwrite(roi_path, results['roi'])

                if args.deidentify:
                    deident_path = os.path.join(args.output, f"{os.path.splitext(os.path.basename(image_path))[0]}_deidentified.png")
                    cv2.imwrite(deident_path, results['deidentified'])

            else:
                # Simple prediction
                mask = predictor.predict_single_image(image_path, args.threshold)
                save_prediction_results(image_path, mask, args.output)

                # Additional processing if requested
                if args.extract_roi or args.deidentify:
                    original = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                    original_resized = predictor.preprocessor.resize_with_padding(original)

                    if args.extract_roi:
                        roi = predictor.extract_roi(original_resized, mask)
                        roi_path = os.path.join(args.output, f"{os.path.splitext(os.path.basename(image_path))[0]}_roi.png")
                        cv2.imwrite(roi_path, roi)

                    if args.deidentify:
                        deidentified = predictor.apply_mask_for_deidentification(original_resized, mask)
                        deident_path = os.path.join(args.output, f"{os.path.splitext(os.path.basename(image_path))[0]}_deidentified.png")
                        cv2.imwrite(deident_path, deidentified)

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    print(f"\\nPrediction completed! Results saved to: {args.output}")


def evaluate_cli(args):
    """CLI for evaluation command."""
    import json

    from .inference import UNetPredictor
    from .preprocessing import UltrasoundPreprocessor
    from .training import evaluate_model

    print("="*60)
    print("EchoROI - EVALUATION")
    print("="*60)

    # Initialize components
    predictor = UNetPredictor(args.model_path)
    preprocessor = UltrasoundPreprocessor()

    # Load test data
    print("Loading test data...")
    X_test, Y_test = preprocessor.load_dataset(args.image_dir, args.mask_dir)

    # Evaluate model
    metrics = evaluate_model(predictor.model, X_test, Y_test)

    # Save results if output directory specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

        # Save metrics to JSON
        metrics_path = os.path.join(args.output, "evaluation_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\\nEvaluation results saved to: {metrics_path}")

    return metrics


def create_data_cli(args):
    """CLI for creating sample data."""
    from .preprocessing import create_sample_data

    print("="*60)
    print("EchoROI - CREATE SAMPLE DATA")
    print("="*60)

    create_sample_data(args.output_dir, args.num_samples)
    print(f"\\nSample data created successfully in: {args.output_dir}")


def benchmark_cli(args):
    """CLI for benchmarking."""
    from .inference import UNetPredictor

    print("="*60)
    print("EchoROI - BENCHMARK")
    print("="*60)

    # Initialize predictor
    predictor = UNetPredictor(args.model_path)

    # Run benchmark
    stats = predictor.benchmark_inference_speed(args.image_path, args.num_runs)

    return stats


# Individual CLI functions for setup.py entry points
def train_cli_entry():
    """Entry point for training CLI."""
    import sys
    sys.argv = ['echoroi', 'train'] + sys.argv[1:]
    main()


def predict_cli_entry():
    """Entry point for prediction CLI."""
    import sys
    sys.argv = ['echoroi', 'predict'] + sys.argv[1:]
    main()


def evaluate_cli_entry():
    """Entry point for evaluation CLI."""
    import sys
    sys.argv = ['echoroi', 'evaluate'] + sys.argv[1:]
    main()


if __name__ == "__main__":
    main()
