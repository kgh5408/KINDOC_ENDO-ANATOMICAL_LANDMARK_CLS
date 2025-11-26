import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class BinaryClassificationDataset(Dataset):
    """Binary classification을 위한 일반화된 Dataset 클래스
    
    레이블:
        - 0: Negative class (target_label이 아닌 경우)
        - 1: Positive class (target_label인 경우)
    
    모델 출력: [batch, 2] where class 0 = negative, class 1 = positive
    """
    
    def __init__(self, csv_path, image_dir, target_label, transform=None):
        """
        Args:
            csv_path: CSV 파일 경로
            image_dir: 이미지 디렉토리 경로
            target_label: binary classification의 positive class (e.g., 'cecum')
            transform: 이미지 변환
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.target_label = target_label
        self.transform = transform
        
        # 컬럼명 정규화 (대소문자 모두 지원)
        self.df.columns = self.df.columns.str.lower()
        
        # Binary label 생성: target_label이면 1, 아니면 0
        self.df['binary_label'] = (self.df['label_raw'] == target_label).astype(int)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(row['binary_label'], dtype=torch.long)
        
        return image, label
    
    def get_class_distribution(self):
        """클래스 분포 반환"""
        return self.df['binary_label'].value_counts().to_dict()


def get_default_transforms(train=True, img_size=224):
    """강화된 이미지 변환 - 의료 영상에 적합한 augmentation
    
    Medical image augmentation should be conservative but effective:
    - Geometric: rotation, flipping, affine
    - Color: brightness, contrast (moderate changes)
    - Quality: blur, noise (simulating different equipment)
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # Geometric augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10
            ),
            # Color augmentations (moderate for medical images)
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            # Perspective and quality
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            # RandomErasing after ToTensor (simulates occlusion)
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(train_csv, val_csv, image_dir, target_label, 
                       batch_size=32, num_workers=4, img_size=224):
    """DataLoader 생성"""
    train_transform = get_default_transforms(train=True, img_size=img_size)
    val_transform = get_default_transforms(train=False, img_size=img_size)
    
    train_dataset = BinaryClassificationDataset(
        train_csv, image_dir, target_label, train_transform
    )
    val_dataset = BinaryClassificationDataset(
        val_csv, image_dir, target_label, val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, train_dataset, val_dataset

