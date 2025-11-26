import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
try:
    # PyTorch >= 2.0
    from torch.amp import autocast, GradScaler
except ImportError:
    # PyTorch < 2.0 (backward compatibility)
    from torch.cuda.amp import autocast, GradScaler
from .losses import get_loss_function


class Trainer:
    """일반화된 학습 클래스
    
    CrossEntropyLoss 사용 (내부적으로 softmax 포함)
    평가 메트릭: Loss, AUC-ROC
    모델 출력: [batch, num_classes] logits
    """
    
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                 save_dir='outputs', experiment_name='experiment', use_amp=True):
        """
        Args:
            model: PyTorch 모델 (출력: [batch, num_classes] logits)
            train_loader: 학습 데이터로더
            val_loader: 검증 데이터로더
            device: 'cuda' or 'cpu'
            save_dir: 모델 저장 디렉토리
            experiment_name: 실험 이름
            use_amp: Automatic Mixed Precision 사용 여부 (CUDA에서만 동작)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.use_amp = use_amp and (device == 'cuda')
        
        # Mixed precision scaler
        if self.use_amp:
            try:
                # PyTorch >= 2.0
                self.scaler = GradScaler('cuda')
            except TypeError:
                # PyTorch < 2.0 (backward compatibility)
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 학습 히스토리
        self.history = {
            'train_loss': [],
            'train_auc': [],
            'train_f1': [],
            'val_loss': [],
            'val_auc': [],
            'val_f1': []
        }
        
        # 학습 파라미터 (train 메서드에서 설정됨)
        self.training_params = {}
        
    def train_epoch(self, criterion, optimizer, max_grad_norm=1.0):
        """1 epoch 학습 (with Mixed Precision and Gradient Clipping)
        
        Loss: CrossEntropyLoss or FocalLoss (logits 입력, 내부적으로 softmax 적용)
        Metrics: AUC-ROC, F1-Score (target class에 대한 평가)
        
        Args:
            criterion: Loss function
            optimizer: Optimizer
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        """
        self.model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Mixed Precision Training
            if self.use_amp:
                try:
                    # PyTorch >= 2.0
                    with autocast('cuda'):
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                except TypeError:
                    # PyTorch < 2.0 (backward compatibility)
                    with autocast():
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                
                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping (unscale first for accurate clipping)
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Optimizer step with scaler
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard training without mixed precision
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            
            # Metrics 계산을 위한 값 저장
            with torch.no_grad():
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1)
            })
        
        # Metrics 계산
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        
        # AUC (positive class = target_label)
        try:
            if all_probs.shape[1] == 2:  # 이진 분류
                epoch_auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:  # 다중 분류
                epoch_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except:
            epoch_auc = 0.0
        
        # Metrics (positive class = target_label, using argmax)
        try:
            epoch_f1 = f1_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)
            epoch_precision = precision_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)
            epoch_recall = recall_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)
        except:
            epoch_f1 = 0.0
            epoch_precision = 0.0
            epoch_recall = 0.0
        
        epoch_loss = running_loss / len(self.train_loader)
        
        return epoch_loss, epoch_auc, epoch_f1, epoch_precision, epoch_recall
    
    def validate(self, criterion):
        """검증 (with Mixed Precision support)
        
        Loss: CrossEntropyLoss or FocalLoss (logits 입력, 내부적으로 softmax 적용)
        Metrics: AUC-ROC, F1-Score (target class에 대한 평가)
        """
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Mixed Precision Inference
                if self.use_amp:
                    try:
                        # PyTorch >= 2.0
                        with autocast('cuda'):
                            outputs = self.model(images)
                            loss = criterion(outputs, labels)
                    except TypeError:
                        # PyTorch < 2.0 (backward compatibility)
                        with autocast():
                            outputs = self.model(images)
                            loss = criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                
                # Metrics 계산을 위한 값 저장
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': running_loss / (pbar.n + 1)
                })
        
        # Metrics 계산
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        
        # AUC (positive class = target_label)
        try:
            if all_probs.shape[1] == 2:  # 이진 분류
                epoch_auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:  # 다중 분류
                epoch_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except:
            epoch_auc = 0.0
        
        # Metrics (positive class = target_label, using argmax)
        try:
            epoch_f1 = f1_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)
            epoch_precision = precision_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)
            epoch_recall = recall_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)
        except:
            epoch_f1 = 0.0
            epoch_precision = 0.0
            epoch_recall = 0.0
        
        epoch_loss = running_loss / len(self.val_loader)
        
        return epoch_loss, epoch_auc, epoch_f1, epoch_precision, epoch_recall
    
    def train(self, num_epochs=10, lr=0.001, weight_decay=0.0001, 
              save_best=True, early_stopping_patience=20,
              loss_type='focal', focal_gamma=2.0, max_grad_norm=1.0,
              scheduler_patience=5, scheduler_factor=0.5):
        """전체 학습 프로세스 (with improved training strategies)
        
        Args:
            num_epochs: 학습 에포크 수
            lr: 초기 학습률
            weight_decay: L2 regularization
            save_best: 최고 성능 모델 저장 여부
            early_stopping_patience: Early stopping patience
            loss_type: 'crossentropy', 'focal', or 'label_smoothing'
            focal_gamma: Focal loss gamma parameter (default: 2.0)
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
            scheduler_patience: LR scheduler patience (default: 5)
            scheduler_factor: LR scheduler reduction factor (default: 0.5)
        """
        # Calculate class weights for imbalanced dataset
        class_weights = None
        if hasattr(self.train_loader.dataset, 'get_class_distribution'):
            try:
                dist = self.train_loader.dataset.get_class_distribution()
                # Sort by key to ensure correct order (0, 1, ...)
                counts = [dist[k] for k in sorted(dist.keys())]
                total_samples = sum(counts)
                num_classes = len(counts)
                
                # Calculate weights: total / (num_classes * count)
                weights = [total_samples / (num_classes * count) for count in counts]
                class_weights = torch.FloatTensor(weights).to(self.device)
                print(f"Using {loss_type} loss with class weights: {weights}")
            except Exception as e:
                print(f"Failed to calculate class weights: {e}")
                class_weights = None
        
        # Create loss function
        criterion = get_loss_function(
            loss_type=loss_type, 
            class_weights=class_weights,
            gamma=focal_gamma
        )
        
        # Optimizer with improved settings
        optimizer = optim.AdamW(  # AdamW is better than Adam
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine Annealing with Warm Restarts (better than ReduceLROnPlateau)
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Initial restart period
            T_mult=2,  # Period multiplier
            eta_min=lr * 0.01  # Minimum learning rate
        )
        
        # Also use ReduceLROnPlateau as a safety net
        scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=scheduler_factor, 
            patience=scheduler_patience,
            min_lr=lr * 0.001
        )
        
        # 학습 파라미터 저장
        self.training_params = {
            'num_epochs': num_epochs,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'optimizer': 'AdamW',
            'loss_function': loss_type,
            'focal_gamma': focal_gamma if loss_type == 'focal' else None,
            'max_grad_norm': max_grad_norm,
            'scheduler': 'CosineAnnealingWarmRestarts + ReduceLROnPlateau',
            'scheduler_params': {
                'mode': 'max',
                'factor': scheduler_factor,
                'patience': scheduler_patience,
                'cosine_T0': 10,
                'cosine_T_mult': 2
            },
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_monitor': 'val_f1',
            'save_best_monitor': 'val_f1',
            'batch_size': self.train_loader.batch_size,
            'train_dataset_size': len(self.train_loader.dataset),
            'val_dataset_size': len(self.val_loader.dataset),
            'device': str(self.device),
            'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'multi_gpu': isinstance(self.model, torch.nn.DataParallel),
            'use_amp': self.use_amp,
            'experiment_name': self.experiment_name
        }
        
        best_val_f1 = -1.0
        patience_counter = 0
        
        print(f"\n{'='*50}")
        print(f"Starting training: {self.experiment_name}")
        print(f"Loss Function: {loss_type}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Gradient Clipping: {max_grad_norm}")
        print(f"Monitor: Validation F1 Score (higher is better)")
        print(f"Early Stopping Patience: {early_stopping_patience}")
        print(f"{'='*50}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_auc, train_f1, train_prec, train_rec = self.train_epoch(
                criterion, optimizer, max_grad_norm=max_grad_norm
            )
            self.history['train_loss'].append(train_loss)
            self.history['train_auc'].append(train_auc)
            self.history['train_f1'].append(train_f1)
            
            # Validate
            val_loss, val_auc, val_f1, val_prec, val_rec = self.validate(criterion)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            self.history['val_f1'].append(val_f1)
            
            # Learning rate scheduling
            scheduler_cosine.step()  # Step every epoch
            scheduler_plateau.step(val_f1)  # Step based on validation F1
            current_lr = optimizer.param_groups[0]['lr']
            
            log_msg = f"Epoch {epoch+1}/{num_epochs} | " \
                      f"LR: {current_lr:.6f} | " \
                      f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f} | " \
                      f"Val - Loss: {val_loss:.4f}, F1: {val_f1:.4f}"
            
            # Save best model (validation f1 기준)
            if save_best and val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_checkpoint('best_model.pth', epoch, val_loss, val_auc, val_f1)
                log_msg += f" | * Best F1: {val_f1:.4f}"
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(log_msg)
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"\n{'='*50}")
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"No improvement in validation F1 for {early_stopping_patience} epochs")
                print(f"{'='*50}")
                break
        
        # Save final model
        self.save_checkpoint('final_model.pth', num_epochs, val_loss, val_auc, val_f1)
        self.save_history()
        self.save_training_params()
        
        print(f"\n{'='*50}")
        print(f"Training completed!")
        print(f"Best Val F1: {best_val_f1:.4f}")
        print(f"{'='*50}\n")
        
        return self.history
    
    def save_checkpoint(self, filename, epoch, val_loss, val_auc, val_f1=None):
        """체크포인트 저장"""
        save_path = os.path.join(self.save_dir, filename)
        
        # DataParallel 사용 시 module.state_dict() 사용
        if isinstance(self.model, torch.nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if val_f1 is not None:
            save_dict['val_f1'] = val_f1
            
        torch.save(save_dict, save_path)
    
    def save_history(self):
        """학습 히스토리 저장"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def save_training_params(self):
        """학습 파라미터 저장"""
        params_path = os.path.join(self.save_dir, 'training_params.json')
        with open(params_path, 'w') as f:
            json.dump(self.training_params, f, indent=4)
        print(f"Training parameters saved to: {params_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # DataParallel 사용 시 module.load_state_dict() 사용
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Val Loss: {checkpoint.get('val_loss', 'N/A')}")
        print(f"Val AUC: {checkpoint.get('val_auc', 'N/A'):.4f}" if 'val_auc' in checkpoint else "Val AUC: N/A")
        return checkpoint

