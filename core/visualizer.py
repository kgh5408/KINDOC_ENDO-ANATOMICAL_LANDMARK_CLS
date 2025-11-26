import matplotlib.pyplot as plt
import os


def plot_training_history(history, save_dir=None, show=True):
    """학습 히스토리 시각화
    
    Args:
        history: 학습 히스토리 딕셔너리 (train_loss, train_auc, train_f1, val_loss, val_auc, val_f1)
        save_dir: 저장 디렉토리 (None이면 저장하지 않음)
        show: plt.show() 호출 여부
        
    Returns:
        save_path: 저장된 파일 경로 (저장한 경우)
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # AUC plot
    axes[1].plot(history['train_auc'], label='Train AUC', marker='o')
    axes[1].plot(history['val_auc'], label='Val AUC', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Training and Validation AUC')
    axes[1].legend()
    axes[1].grid(True)
    
    # F1-Score plot
    axes[2].plot(history['train_f1'], label='Train F1', marker='o')
    axes[2].plot(history['val_f1'], label='Val F1', marker='s')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_title('Training and Validation F1-Score')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return save_path


def print_training_summary(history):
    """학습 결과 요약 출력
    
    Args:
        history: 학습 히스토리 딕셔너리
    """
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total epochs trained: {len(history['train_loss'])}")
    print(f"\nBest Metrics:")
    print(f"  Best Val Loss: {min(history['val_loss']):.4f} (Epoch {history['val_loss'].index(min(history['val_loss'])) + 1})")
    print(f"  Best Val AUC: {max(history['val_auc']):.4f} (Epoch {history['val_auc'].index(max(history['val_auc'])) + 1})")
    print(f"  Best Val F1: {max(history['val_f1']):.4f} (Epoch {history['val_f1'].index(max(history['val_f1'])) + 1})")
    print(f"\nFinal Metrics:")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Train AUC: {history['train_auc'][-1]:.4f}")
    print(f"  Train F1: {history['train_f1'][-1]:.4f}")
    print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"  Val AUC: {history['val_auc'][-1]:.4f}")
    print(f"  Val F1: {history['val_f1'][-1]:.4f}")
    print("=" * 60)

