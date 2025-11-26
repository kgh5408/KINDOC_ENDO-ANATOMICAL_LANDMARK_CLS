import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


class Evaluator:
    """일반화된 평가 클래스
    
    모델 출력(logits)에 softmax를 적용하여 확률값으로 변환하고 평가
    Binary classification: [batch, 2] -> softmax -> [P(class 0), P(class 1)]
    """
    
    def __init__(self, model, test_loader, device='cuda', save_dir=None):
        """
        Args:
            model: PyTorch 모델 (출력: [batch, num_classes])
            test_loader: 테스트 데이터로더
            device: 'cuda' or 'cpu'
            save_dir: 결과 저장 디렉토리
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    def predict(self, threshold=None):
        """예측 수행
        
        모델 출력(logits)에 softmax를 적용하여 확률값으로 변환
        Binary classification: outputs [batch, 2] -> probs [batch, 2] where sum(probs, dim=1) = 1
        
        Args:
            threshold: 분류 threshold. None이면 argmax 사용
        """
        self.model.eval()
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Evaluating'):
                images = images.to(self.device)
                
                # 모델 출력 (logits) -> softmax -> 확률값
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                if threshold is not None:
                    # threshold 기반 예측
                    predicted = (probs[:, 1] >= threshold).long()
                else:
                    # argmax 기반 예측
                    _, predicted = outputs.max(1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def find_optimal_threshold(self, y_true, y_probs):
        """Sensitivity와 Specificity의 차이가 최소인 threshold 찾기
        
        Args:
            y_true: 실제 레이블
            y_probs: 예측 확률 (positive class)
            
        Returns:
            optimal_threshold, sensitivity, specificity
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        
        # Sensitivity (TPR) = Recall
        # Specificity = 1 - FPR
        sensitivity = tpr
        specificity = 1 - fpr
        
        # Sensitivity와 Specificity의 차이 계산
        diff = np.abs(sensitivity - specificity)
        
        # 차이가 최소인 지점 찾기
        optimal_idx = np.argmin(diff)
        optimal_threshold = thresholds[optimal_idx]
        optimal_sensitivity = sensitivity[optimal_idx]
        optimal_specificity = specificity[optimal_idx]
        
        return optimal_threshold, optimal_sensitivity, optimal_specificity
    
    def evaluate(self, class_names=None, use_optimal_threshold=True):
        """전체 평가 수행
        
        Args:
            class_names: 클래스 이름 리스트
            use_optimal_threshold: optimal threshold 사용 여부
        """
        if class_names is None:
            class_names = ['Negative', 'Positive']
        
        # 먼저 확률 얻기 (threshold 없이)
        y_true, _, y_probs = self.predict(threshold=None)
        
        # Optimal threshold 사용 시에만 계산 및 출력
        if use_optimal_threshold:
            # Optimal threshold 찾기
            optimal_threshold, optimal_sens, optimal_spec = self.find_optimal_threshold(
                y_true, y_probs[:, 1]
            )
            
            print(f"\n{'='*50}")
            print("Optimal Threshold Analysis")
            print(f"{'='*50}")
            print(f"Optimal Threshold: {optimal_threshold:.4f}")
            print(f"Sensitivity (Recall): {optimal_sens:.4f}")
            print(f"Specificity: {optimal_spec:.4f}")
            print(f"Difference: {abs(optimal_sens - optimal_spec):.4f}")
            print(f"{'='*50}\n")
            
            # Optimal threshold로 재예측
            y_true, y_pred, y_probs = self.predict(threshold=optimal_threshold)
        else:
            # argmax 사용 (두 클래스 중 확률이 더 큰 클래스 선택)
            y_true, y_pred, y_probs = self.predict(threshold=None)
            optimal_threshold = 0.5  # metrics 저장용 명목상 값 (실제로는 argmax 사용)
        
        # Specificity 계산 (TN / (TN + FP))
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # 메트릭 계산
        metrics = {
            'optimal_threshold': float(optimal_threshold),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
            'auc': roc_auc_score(y_true, y_probs[:, 1])
        }
        
        # 결과 출력
        print(f"\n{'='*50}")
        if use_optimal_threshold:
            print("Evaluation Results (with Optimal Threshold)")
        else:
            print("Evaluation Results (with Default Threshold)")
        print(f"{'='*50}")
        print(f"Threshold:   {metrics['optimal_threshold']:.4f}")
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"F1-Score:    {metrics['f1']:.4f}")
        print(f"AUC-ROC:     {metrics['auc']:.4f}")
        print(f"{'='*50}\n")
        
        # Classification Report
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        print(cm)
        print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        
        # 저장
        if self.save_dir:
            self.save_results(metrics, y_true, y_pred, y_probs, cm, class_names, use_optimal_threshold)
        
        return metrics, y_true, y_pred, y_probs
    
    def save_results(self, metrics, y_true, y_pred, y_probs, cm, class_names, use_optimal_threshold=True):
        """결과 저장"""
        # 메트릭 저장
        metrics_path = os.path.join(self.save_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Threshold 정보 별도 저장 (optimal threshold 사용 시에만)
        if use_optimal_threshold:
            threshold_info = {
                'optimal_threshold': metrics['optimal_threshold'],
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                'sens_spec_difference': abs(metrics['sensitivity'] - metrics['specificity']),
                'note': 'Threshold where |Sensitivity - Specificity| is minimized'
            }
            threshold_path = os.path.join(self.save_dir, 'optimal_threshold.json')
            with open(threshold_path, 'w') as f:
                json.dump(threshold_info, f, indent=4)
        
        # Confusion Matrix 시각화
        self.plot_confusion_matrix(cm, class_names)
        
        # ROC Curve 시각화 (optimal threshold 사용 시에만 표시)
        if use_optimal_threshold:
            self.plot_roc_curve(y_true, y_probs[:, 1], optimal_threshold=metrics['optimal_threshold'])
            # Sensitivity-Specificity Curve 시각화
            self.plot_sensitivity_specificity_curve(y_true, y_probs[:, 1], optimal_threshold=metrics['optimal_threshold'])
        else:
            self.plot_roc_curve(y_true, y_probs[:, 1], optimal_threshold=None)
        
        print(f"\nResults saved to: {self.save_dir}")
    
    def plot_confusion_matrix(self, cm, class_names):
        """Confusion Matrix 시각화"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, y_true, y_scores, optimal_threshold=None):
        """ROC Curve 시각화"""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        # Optimal threshold 지점 표시
        if optimal_threshold is not None:
            # optimal threshold에서의 FPR, TPR 찾기
            idx = np.argmin(np.abs(thresholds - optimal_threshold))
            plt.plot(fpr[idx], tpr[idx], 'ro', markersize=10, 
                    label=f'Optimal Threshold ({optimal_threshold:.3f})')
            plt.annotate(f'Threshold: {optimal_threshold:.3f}\nSens: {tpr[idx]:.3f}\nSpec: {1-fpr[idx]:.3f}',
                        xy=(fpr[idx], tpr[idx]), xytext=(fpr[idx]+0.15, tpr[idx]-0.15),
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, 'roc_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_sensitivity_specificity_curve(self, y_true, y_scores, optimal_threshold=None):
        """Sensitivity와 Specificity 곡선 시각화"""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        sensitivity = tpr
        specificity = 1 - fpr
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, sensitivity, 'b-', label='Sensitivity (Recall)', linewidth=2)
        plt.plot(thresholds, specificity, 'r-', label='Specificity', linewidth=2)
        
        # Optimal threshold 표시
        if optimal_threshold is not None:
            idx = np.argmin(np.abs(thresholds - optimal_threshold))
            plt.axvline(x=optimal_threshold, color='g', linestyle='--', linewidth=2,
                       label=f'Optimal Threshold ({optimal_threshold:.3f})')
            plt.plot(optimal_threshold, sensitivity[idx], 'bo', markersize=10)
            plt.plot(optimal_threshold, specificity[idx], 'ro', markersize=10)
            
            # 텍스트 주석
            plt.text(optimal_threshold + 0.05, sensitivity[idx], 
                    f'Sens: {sensitivity[idx]:.3f}', fontsize=10, color='blue')
            plt.text(optimal_threshold + 0.05, specificity[idx], 
                    f'Spec: {specificity[idx]:.3f}', fontsize=10, color='red')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Sensitivity and Specificity vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        save_path = os.path.join(self.save_dir, 'sensitivity_specificity_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def load_and_evaluate(checkpoint_path, test_loader, device='cuda', save_dir=None, 
                      model_name='resnet50', num_classes=2, use_multi_gpu=True):
    """체크포인트 로드 후 평가
    
    Args:
        checkpoint_path: 체크포인트 경로
        test_loader: 테스트 데이터로더
        device: 'cuda' or 'cpu'
        save_dir: 결과 저장 디렉토리
        model_name: 모델 이름
        num_classes: 출력 클래스 수
        use_multi_gpu: Multi-GPU 사용 여부
    """
    from core.models.binary_classifier import BinaryClassifier
    import torch.nn as nn
    
    # 모델 생성 및 로드
    model = BinaryClassifier(model_name=model_name, pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Multi-GPU 설정
    if use_multi_gpu and device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    # 평가
    evaluator = Evaluator(model, test_loader, device, save_dir)
    metrics, y_true, y_pred, y_probs = evaluator.evaluate()
    
    return metrics, y_true, y_pred, y_probs



