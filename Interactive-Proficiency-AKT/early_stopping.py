"""
Early stopping mechanism for training optimization.

This module provides an early stopping class that monitors validation metrics
and stops training when no improvement is observed for a specified number of epochs.
"""
from typing import Optional


class EarlyStopping:
    """
    Early stopping callback to stop training when validation metric stops improving.
    
    Monitors a validation metric (typically AUC or loss) and stops training
    if no improvement is observed for a specified number of consecutive epochs.
    This helps prevent overfitting and saves computational resources.
    
    Args:
        patience (int, optional): Number of epochs to wait before stopping after
            no improvement. Defaults to 10.
        min_delta (float, optional): Minimum change to qualify as an improvement.
            Defaults to 0.0 (any improvement counts).
        mode (str, optional): 'max' for metrics to maximize (e.g., AUC),
            'min' for metrics to minimize (e.g., loss). Defaults to 'max'.
        enabled (bool, optional): Whether early stopping is enabled.
            Defaults to True.
    
    Attributes:
        patience (int): Number of epochs to wait.
        min_delta (float): Minimum change threshold.
        mode (str): Optimization mode ('max' or 'min').
        enabled (bool): Whether early stopping is active.
        counter (int): Number of epochs without improvement.
        best_score (Optional[float]): Best metric value observed so far.
        stopped_epoch (int): Epoch at which training was stopped.
    
    Example:
        >>> early_stopping = EarlyStopping(patience=10, mode='max')
        >>> for epoch in range(max_epochs):
        ...     train_model()
        ...     val_auc = validate_model()
        ...     if early_stopping(val_auc):
        ...         print(f"Early stopping at epoch {epoch}")
        ...         break
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'max', enabled: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.enabled = enabled
        self.counter = 0
        self.best_score: Optional[float] = None
        self.stopped_epoch = 0
        
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop based on current metric score.
        
        Args:
            score (float): Current validation metric score to evaluate.
        
        Returns:
            bool: True if training should stop, False otherwise.
        
        Note:
            - For 'max' mode: improvement means score > best_score + min_delta
            - For 'min' mode: improvement means score < best_score - min_delta
            - Resets counter when improvement is detected
            - Increments counter when no improvement
        """
        if not self.enabled:
            return False
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            is_better = score > self.best_score + self.min_delta
        else:  # mode == 'min'
            is_better = score < self.best_score - self.min_delta
        
        if is_better:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = self.counter
                return True
            return False
    
    def reset(self) -> None:
        """
        Reset early stopping state.
        
        Clears the counter and best score, allowing the early stopping
        mechanism to be reused for a new training run.
        """
        self.counter = 0
        self.best_score = None
        self.stopped_epoch = 0
