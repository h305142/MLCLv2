import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple


class MetricsCalculator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all stored metrics"""
        self.predictions = []
        self.targets = []
        self.losses = []
        
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
        """Update metrics with new batch results"""
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.losses.append(loss)
    
    def compute_accuracy(self) -> float:
        """Compute overall accuracy"""
        if len(self.predictions) == 0:
            return 0.0
        return accuracy_score(self.targets, self.predictions) * 100.0
    
    def compute_precision_recall_f1(self) -> Tuple[float, float, float]:
        """Compute macro-averaged precision, recall, and F1 score"""
        if len(self.predictions) == 0:
            return 0.0, 0.0, 0.0
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.targets, self.predictions, average='macro', zero_division=0
        )
        return precision, recall, f1
    
    def compute_per_class_accuracy(self, num_classes: int) -> Dict[int, float]:
        """Compute accuracy for each class"""
        if len(self.predictions) == 0:
            return {i: 0.0 for i in range(num_classes)}
        
        cm = confusion_matrix(self.targets, self.predictions, labels=range(num_classes))
        per_class_acc = {}
        
        for i in range(num_classes):
            if cm[i].sum() > 0:
                per_class_acc[i] = cm[i, i] / cm[i].sum() * 100.0
            else:
                per_class_acc[i] = 0.0
        
        return per_class_acc
    
    def compute_worst_group_accuracy(self, group_labels: List[int]) -> float:
        """Compute worst-case group accuracy"""
        if len(self.predictions) == 0:
            return 0.0
        
        unique_groups = set(group_labels)
        group_accuracies = []
        
        for group in unique_groups:
            group_mask = np.array(group_labels) == group
            if group_mask.sum() > 0:
                group_preds = np.array(self.predictions)[group_mask]
                group_targets = np.array(self.targets)[group_mask]
                group_acc = accuracy_score(group_targets, group_preds) * 100.0
                group_accuracies.append(group_acc)
        
        return min(group_accuracies) if group_accuracies else 0.0
    
    def compute_average_loss(self) -> float:
        """Compute average loss"""
        if len(self.losses) == 0:
            return 0.0
        return np.mean(self.losses)
    
    def get_all_metrics(self, num_classes: int = None, group_labels: List[int] = None) -> Dict[str, float]:
        """Get all computed metrics"""
        metrics = {
            'accuracy': self.compute_accuracy(),
            'average_loss': self.compute_average_loss()
        }
        
        precision, recall, f1 = self.compute_precision_recall_f1()
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        if num_classes is not None:
            per_class_acc = self.compute_per_class_accuracy(num_classes)
            for class_id, acc in per_class_acc.items():
                metrics[f'class_{class_id}_accuracy'] = acc
        
        if group_labels is not None:
            metrics['worst_group_accuracy'] = self.compute_worst_group_accuracy(group_labels)
        
        return metrics


class RobustnessMetrics:
    @staticmethod
    def compute_distribution_shift_gap(clean_acc: float, shifted_acc: float) -> float:
        """Compute accuracy gap between clean and shifted distributions"""
        return clean_acc - shifted_acc
    
    @staticmethod
    def compute_average_worst_group_gap(group_accuracies: List[float]) -> float:
        """Compute gap between best and worst group accuracies"""
        if len(group_accuracies) < 2:
            return 0.0
        return max(group_accuracies) - min(group_accuracies)
    
    @staticmethod
    def compute_fairness_metrics(group_accuracies: Dict[int, float]) -> Dict[str, float]:
        """Compute fairness metrics across groups"""
        accuracies = list(group_accuracies.values())
        
        return {
            'min_group_accuracy': min(accuracies),
            'max_group_accuracy': max(accuracies),
            'accuracy_gap': max(accuracies) - min(accuracies),
            'accuracy_std': np.std(accuracies)
        }


class UncertaintyMetrics:
    @staticmethod
    def compute_entropy(probabilities: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distributions"""
        log_probs = torch.log(probabilities + 1e-8)
        entropy = -(probabilities * log_probs).sum(dim=1)
        return entropy
    
    @staticmethod
    def compute_confidence(probabilities: torch.Tensor) -> torch.Tensor:
        """Compute confidence (maximum probability)"""
        max_probs, _ = torch.max(probabilities, dim=1)
        return max_probs
    
    @staticmethod
    def compute_uncertainty_metrics(logits: torch.Tensor) -> Dict[str, float]:
        """Compute uncertainty-related metrics"""
        probs = torch.softmax(logits, dim=1)
        
        entropy = UncertaintyMetrics.compute_entropy(probs)
        confidence = UncertaintyMetrics.compute_confidence(probs)
        
        return {
            'average_entropy': entropy.mean().item(),
            'average_confidence': confidence.mean().item(),
            'entropy_std': entropy.std().item(),
            'confidence_std': confidence.std().item()
        }