"""
Main entry point for training the LSTM Portfolio Risk Assistant.
Runs the full training pipeline: data download, feature engineering, model training, evaluation.
"""

import os
import argparse
from datetime import datetime

from config import (
    ALL_TICKERS,
    MODELS_DIR,
    MODEL_PATH,
    YEARS_OF_DATA
)
from data import download_all_tickers, get_data_summary
from features import prepare_all_data
from model import (
    build_lstm_model,
    train_model,
    evaluate_model,
    build_baseline_models,
    compare_models,
    save_model,
    plot_training_history,
    plot_confusion_matrix
)


def main(force_download: bool = False, skip_training: bool = False):
    """
    Run the full training pipeline.
    
    Args:
        force_download: If True, re-download data even if local files exist
        skip_training: If True, skip model training (for testing data pipeline)
    """
    print("=" * 60)
    print("LSTM Portfolio Risk Assistant - Training Pipeline")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # =========================================================================
    # Step 1: Download Data
    # =========================================================================
    print("Step 1: Downloading data...")
    print("-" * 40)
    
    data = download_all_tickers(force=force_download)
    
    if not data:
        print("ERROR: No data downloaded. Exiting.")
        return
    
    print("\nData Summary:")
    print(get_data_summary(data).to_string(index=False))
    print()
    
    # =========================================================================
    # Step 2: Prepare Features and Labels
    # =========================================================================
    print("Step 2: Preparing features and labels...")
    print("-" * 40)
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = prepare_all_data()
    
    if len(X_train) == 0:
        print("ERROR: No training data prepared. Exiting.")
        return
    
    print(f"\nFeature dimensions: {X_train.shape[2]}")
    print(f"Number of features: {len(feature_names)}")
    print()
    
    if skip_training:
        print("Skipping training (skip_training=True)")
        return
    
    # =========================================================================
    # Step 3: Build and Train LSTM Model
    # =========================================================================
    print("Step 3: Building and training LSTM model...")
    print("-" * 40)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape=input_shape)
    
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Train model
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    history = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        model_path=MODEL_PATH
    )
    
    # Plot training history
    plot_training_history(history, save_path=os.path.join(MODELS_DIR, 'training_history.png'))
    print()
    
    # =========================================================================
    # Step 4: Train Baseline Models
    # =========================================================================
    print("Step 4: Training baseline models...")
    print("-" * 40)
    
    baselines = build_baseline_models()
    
    for name, baseline in baselines.items():
        print(f"Training {name}...")
        baseline.fit(X_train, y_train)
    
    print()
    
    # =========================================================================
    # Step 5: Evaluate and Compare Models
    # =========================================================================
    print("Step 5: Evaluating models...")
    print("-" * 40)
    
    # Compare all models
    comparison_df = compare_models(model, baselines, X_test, y_test)
    
    # Save comparison results
    comparison_df.to_csv(os.path.join(MODELS_DIR, 'model_comparison.csv'))
    print(f"\nComparison saved to {os.path.join(MODELS_DIR, 'model_comparison.csv')}")
    
    # Plot confusion matrix for LSTM
    import torch
    import numpy as np
    
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test)
        outputs = model(X_test_t)
        y_pred = torch.argmax(outputs, dim=1).numpy()
    
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=os.path.join(MODELS_DIR, 'confusion_matrix.png')
    )
    
    # =========================================================================
    # Step 6: Save Final Model
    # =========================================================================
    print("\nStep 6: Saving final model...")
    print("-" * 40)
    
    save_model(model, MODEL_PATH)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Scaler saved to: models/scaler.pkl")
    print(f"\nBest Test Accuracy: {comparison_df.loc['LSTM', 'accuracy']:.4f}")
    print(f"Best Test F1 (macro): {comparison_df.loc['LSTM', 'f1_macro']:.4f}")
    
    # Check if LSTM beats baselines
    baseline_accuracies = comparison_df.loc[comparison_df.index != 'LSTM', 'accuracy']
    lstm_accuracy = comparison_df.loc['LSTM', 'accuracy']
    
    if lstm_accuracy > baseline_accuracies.max():
        print("\n✓ LSTM beats all baselines!")
    else:
        print(f"\n⚠ LSTM does not beat all baselines. Best baseline: {baseline_accuracies.max():.4f}")
    
    print("\nTo run the dashboard, use: streamlit run dashboard.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM Portfolio Risk Assistant')
    parser.add_argument('--force-download', action='store_true',
                        help='Force re-download data even if local files exist')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training (for testing data pipeline)')
    
    args = parser.parse_args()
    
    main(force_download=args.force_download, skip_training=args.skip_training)
