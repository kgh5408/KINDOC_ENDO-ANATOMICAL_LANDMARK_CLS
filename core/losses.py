import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance
    
    Focal Loss는 Lin et al. (2017)에서 제안된 손실 함수로,
    hard negative examples에 집중하여 class imbalance 문제를 해결합니다.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    where:
        p_t: predicted probability for the true class
        alpha_t: balancing factor (class weight)
        gamma: focusing parameter (default: 2.0)
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Balancing factor for each class (Tensor of size [num_classes])
                   If None, all classes have equal weight
            gamma: Focusing parameter. Higher gamma => more focus on hard examples
                   Recommended: 2.0 for most cases
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch, num_classes] logits (before softmax)
            targets: [batch] class indices
        
        Returns:
            loss: scalar loss value
        """
        # Convert logits to probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get probability for the true class
        batch_size = inputs.size(0)
        probs_for_true_class = probs[range(batch_size), targets]
        
        # Calculate focal term: (1 - p_t)^gamma
        focal_weight = (1 - probs_for_true_class) ** self.gamma
        
        # Calculate cross entropy: -log(p_t)
        ce_loss = -torch.log(probs_for_true_class + 1e-8)
        
        # Apply alpha balancing if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross Entropy with Label Smoothing
    
    Label smoothing은 one-hot encoding 대신 smoothed labels을 사용하여
    모델의 overconfidence를 방지하고 generalization을 향상시킵니다.
    
    Instead of [1, 0] or [0, 1], uses [(1-eps)+eps/K, eps/K] where K is num_classes
    """
    
    def __init__(self, smoothing=0.1):
        """
        Args:
            smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch, num_classes] logits
            targets: [batch] class indices
        
        Returns:
            loss: scalar loss value
        """
        log_probs = F.log_softmax(inputs, dim=1)
        num_classes = inputs.size(1)
        
        # Create smoothed labels
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        
        # Calculate loss
        loss = -(targets_smooth * log_probs).sum(dim=1).mean()
        
        return loss


def get_loss_function(loss_type='crossentropy', class_weights=None, **kwargs):
    """Loss function factory
    
    Args:
        loss_type: 'crossentropy', 'focal', 'label_smoothing'
        class_weights: Tensor of shape [num_classes] for class balancing
        **kwargs: Additional parameters for specific loss functions
            - gamma: for focal loss (default: 2.0)
            - smoothing: for label smoothing (default: 0.1)
    
    Returns:
        loss_fn: PyTorch loss function
    """
    if loss_type == 'crossentropy':
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_type == 'focal':
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=class_weights, gamma=gamma)
    
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        if class_weights is not None:
            print("Warning: Label smoothing does not support class weights directly")
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

