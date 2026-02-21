"""
LSTM model architecture, training, and evaluation module.
Includes baseline models for comparison.
Uses PyTorch for the LSTM implementation.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    MODELS_DIR,
    LSTM_UNITS,
    DROPOUT_RATE,
    DENSE_UNITS,
    N_CLASSES,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    PATIENCE,
    MODEL_PATH,
    REGIME_CLASSES
)


# =============================================================================
# LSTM MODEL ARCHITECTURE (PyTorch)
# =============================================================================

class LSTMClassifier(nn.Module):
    """
    LSTM model for regime classification.
    
    Architecture:
    - Input: (batch, sequence_length, n_features)
    - LSTM layers with dropout
    - Dense output with softmax
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = None, n_classes: int = 3, 
                 dropout: float = 0.2, dense_units: int = 16):
        super(LSTMClassifier, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = LSTM_UNITS
        
        self.hidden_dims = hidden_dims
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        # Dense layers
        self.dense1 = nn.Linear(hidden_dims[1], dense_units)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        
        # Output layer
        self.output = nn.Linear(dense_units, n_classes)
        
    def forward(self, x):
        # First LSTM layer
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        # Second LSTM layer - use last output
        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # Take the last time step
        out = self.dropout2(out)
        
        # Dense layers
        out = self.dense1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        
        # Output
        out = self.output(out)
        return out


def build_lstm_model(input_shape: tuple, n_classes: int = 3) -> LSTMClassifier:
    """
    Build LSTM model for regime classification.
    
    Args:
        input_shape: Tuple of (sequence_length, n_features)
        n_classes: Number of output classes
    
    Returns:
        PyTorch LSTM model
    """
    input_dim = input_shape[1]
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dims=LSTM_UNITS,
        n_classes=n_classes,
        dropout=DROPOUT_RATE,
        dense_units=DENSE_UNITS
    )
    return model


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_model(
    model: LSTMClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_path: str = None,
    epochs: int = None,
    batch_size: int = None,
    class_weights: dict = None,
    device: str = None
) -> dict:
    """
    Train the LSTM model.
    
    Args:
        model: PyTorch LSTM model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_path: Path to save best model
        epochs: Number of epochs (default from config)
        batch_size: Batch size (default from config)
        class_weights: Optional class weights for imbalanced data
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        Training history dictionary
    """
    if epochs is None:
        epochs = EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Training on device: {device}")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    # Create data loader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Compute class weights if not provided
    if class_weights is None:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        print(f"Class weights: {class_weights}")
    
    # Convert class weights to tensor
    weight_list = [class_weights.get(i, 1.0) for i in range(N_CLASSES)]
    weight_tensor = torch.FloatTensor(weight_list).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=PATIENCE // 2
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_acc = (val_predicted == y_val_t).sum().item() / len(y_val_t)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            if model_path:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
    
    return history


def plot_training_history(history: dict, save_path: str = None):
    """
    Plot training and validation metrics.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_accuracy'], label='Train Accuracy')
    axes[1].plot(history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


# =============================================================================
# BASELINE MODELS
# =============================================================================

class MajorityClassBaseline:
    """Baseline that always predicts the majority class."""
    
    def __init__(self):
        self.majority_class = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit by finding the majority class."""
        unique, counts = np.unique(y, return_counts=True)
        self.majority_class = unique[np.argmax(counts)]
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict majority class for all samples."""
        return np.full(len(X), self.majority_class)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability distribution."""
        proba = np.zeros((len(X), N_CLASSES))
        proba[:, self.majority_class] = 1.0
        return proba


class RuleBasedBaseline:
    """
    Baseline that uses recent volatility to predict regime.
    Uses the last volatility value in the sequence to make prediction.
    """
    
    def __init__(self, vol_feature_idx: int = -1):
        """
        Args:
            vol_feature_idx: Index of volatility feature in feature array
        """
        self.vol_feature_idx = vol_feature_idx
        self.low_threshold = None
        self.high_threshold = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit by computing volatility thresholds from training data.
        
        Args:
            X: Sequences (n_samples, seq_length, n_features)
            y: Labels
        """
        # Get the last volatility value from each sequence
        last_vol = X[:, -1, self.vol_feature_idx]
        
        # Compute thresholds based on training data distribution
        self.low_threshold = np.percentile(last_vol, 33)
        self.high_threshold = np.percentile(last_vol, 67)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict based on recent volatility.
        
        Args:
            X: Sequences (n_samples, seq_length, n_features)
        
        Returns:
            Predicted labels
        """
        last_vol = X[:, -1, self.vol_feature_idx]
        
        predictions = np.ones(len(X), dtype=int)  # Default: medium
        predictions[last_vol <= self.low_threshold] = 0  # low
        predictions[last_vol > self.high_threshold] = 2  # high
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return pseudo-probabilities based on distance to thresholds."""
        last_vol = X[:, -1, self.vol_feature_idx]
        
        proba = np.zeros((len(X), N_CLASSES))
        
        for i, vol in enumerate(last_vol):
            if vol <= self.low_threshold:
                proba[i, 0] = 0.7
                proba[i, 1] = 0.2
                proba[i, 2] = 0.1
            elif vol > self.high_threshold:
                proba[i, 0] = 0.1
                proba[i, 1] = 0.2
                proba[i, 2] = 0.7
            else:
                proba[i, 0] = 0.2
                proba[i, 1] = 0.6
                proba[i, 2] = 0.2
        
        return proba


class RandomForestBaseline:
    """Baseline using Random Forest on flattened sequences."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit Random Forest on flattened sequences.
        
        Args:
            X: Sequences (n_samples, seq_length, n_features)
            y: Labels
        """
        # Flatten sequences
        X_flat = X.reshape(X.shape[0], -1)
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_flat, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using flattened sequences."""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities."""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict_proba(X_flat)


def build_baseline_models() -> dict:
    """
    Build all baseline models.
    
    Returns:
        Dictionary of baseline models
    """
    return {
        'Majority Class': MajorityClassBaseline(),
        'Rule-Based': RuleBasedBaseline(vol_feature_idx=2),  # Volatility_5d is typically index 2
        'Random Forest': RandomForestBaseline()
    }


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                   model_name: str = "Model", device: str = None) -> dict:
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained model (PyTorch or sklearn-like)
        X_test: Test features
        y_test: Test labels
        model_name: Name for display
        device: Device for PyTorch models
    
    Returns:
        Dictionary of metrics
    """
    # Get predictions
    if isinstance(model, nn.Module):
        # PyTorch model
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            outputs = model(X_test_t)
            y_proba = torch.softmax(outputs, dim=1).cpu().numpy()
        
        y_pred = np.argmax(y_proba, axis=1)
    elif hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_proba, axis=1)
    else:
        y_proba = model.predict(X_test)
        y_pred = np.argmax(y_proba, axis=1)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0)
    }
    
    # Per-class metrics
    for i, class_name in enumerate(REGIME_CLASSES):
        metrics[f'f1_{class_name}'] = f1_score(y_test, y_pred, labels=[i], average='micro')
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          class_names: list = None, save_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class
        save_path: Optional path to save figure
    """
    if class_names is None:
        class_names = REGIME_CLASSES
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def compare_models(lstm_model: LSTMClassifier, baselines: dict, 
                   X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Compare LSTM model against baselines.
    
    Args:
        lstm_model: Trained LSTM model
        baselines: Dictionary of baseline models
        X_test: Test features
        y_test: Test labels
    
    Returns:
        DataFrame with comparison metrics
    """
    
    results = {}
    
    # Evaluate LSTM
    results['LSTM'] = evaluate_model(lstm_model, X_test, y_test, 'LSTM')
    
    # Evaluate baselines
    for name, model in baselines.items():
        results[name] = evaluate_model(model, X_test, y_test, name)
    
    # Create comparison DataFrame
    df = pd.DataFrame(results).T
    
    # Sort by accuracy
    df = df.sort_values('accuracy', ascending=False)
    
    print("\n" + "=" * 50)
    print("Model Comparison:")
    print("=" * 50)
    print(df[['accuracy', 'f1_macro', 'f1_weighted']].to_string())
    
    return df


# =============================================================================
# MODEL SAVE/LOAD
# =============================================================================

def save_model(model: LSTMClassifier, path: str = None):
    """
    Save trained model.
    
    Args:
        model: Trained PyTorch model
        path: Save path (default from config)
    """
    if path is None:
        path = MODEL_PATH
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_trained_model(path: str = None, input_shape: tuple = None) -> LSTMClassifier:
    """
    Load trained model.
    
    Args:
        path: Model path (default from config)
        input_shape: Required input shape for model initialization
    
    Returns:
        Loaded PyTorch model
    """
    if path is None:
        path = MODEL_PATH
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    
    if input_shape is None:
        raise ValueError("input_shape is required to load PyTorch model")
    
    model = build_lstm_model(input_shape=input_shape)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    
    print(f"Model loaded from {path}")
    return model


if __name__ == "__main__":
    # Test model building
    print("=" * 50)
    print("Testing model architecture...")
    print("=" * 50)
    
    # Create dummy data
    seq_length = 30
    n_features = 25
    n_samples = 100
    
    X_dummy = np.random.randn(n_samples, seq_length, n_features)
    y_dummy = np.random.randint(0, 3, n_samples)
    
    # Build model
    model = build_lstm_model(input_shape=(seq_length, n_features))
    print(model)
    
    # Test forward pass
    X_tensor = torch.FloatTensor(X_dummy[:5])
    output = model(X_tensor)
    print(f"\nOutput shape: {output.shape}")
    
    # Test baselines
    print("\n" + "=" * 50)
    print("Testing baseline models...")
    print("=" * 50)
    
    baselines = build_baseline_models()
    for name, baseline in baselines.items():
        baseline.fit(X_dummy, y_dummy)
        print(f"{name}: fitted successfully")
