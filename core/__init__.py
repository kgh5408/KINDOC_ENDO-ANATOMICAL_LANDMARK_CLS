"""Core module for colonoscopy anatomical location classification

모듈 구조:
- models: 모델 정의 (BinaryClassifier, create_model)
- dataset: 데이터셋 및 DataLoader (BinaryClassificationDataset, create_dataloaders)
- trainer: 학습 (Trainer) - with Focal Loss, Mixed Precision, Gradient Clipping
- evaluator: 평가 (Evaluator, load_and_evaluate)
- visualizer: 시각화 (plot_training_history, print_training_summary)
- losses: Loss functions (FocalLoss, LabelSmoothingCrossEntropy, get_loss_function)

주요 특징:
- 모델 출력: [batch, num_classes] logits (softmax 적용 전)
- 학습: FocalLoss (default) or CrossEntropyLoss, Mixed Precision Training
- 평가: softmax 적용 후 확률값 기반 평가, optimal threshold 찾기
- 향상된 성능: Stronger augmentation, GroupNorm, Gradient clipping
"""

from .models.binary_classifier import (
    BinaryClassifier, 
    create_model, 
    list_available_models,
    print_available_models
)
from .dataset import BinaryClassificationDataset, create_dataloaders
from .trainer import Trainer
from .evaluator import Evaluator, load_and_evaluate
from .visualizer import plot_training_history, print_training_summary
from .losses import FocalLoss, LabelSmoothingCrossEntropy, get_loss_function

__all__ = [
    'BinaryClassifier',
    'create_model',
    'list_available_models',
    'print_available_models',
    'BinaryClassificationDataset',
    'create_dataloaders',
    'Trainer',
    'Evaluator',
    'load_and_evaluate',
    'plot_training_history',
    'print_training_summary',
    'FocalLoss',
    'LabelSmoothingCrossEntropy',
    'get_loss_function'
]
