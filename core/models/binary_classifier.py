import torch
import torch.nn as nn
from torchvision import models


class BinaryClassifier(nn.Module):
    """일반화된 Binary Classification 모델 (SOTA 모델 지원)
    
    출력: [batch, num_classes] logits (raw scores, softmax 적용 전)
    - Binary classification (num_classes=2): [batch, 2] where [:, 0] = negative, [:, 1] = positive
    - 학습 시: CrossEntropyLoss가 내부적으로 softmax 적용
    - 평가 시: torch.softmax(outputs, dim=1)로 확률값 변환 필요
    
    지원 모델:
    - ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
    - EfficientNet: efficientnet_b0 ~ b7, efficientnet_v2_s/m/l
    - ConvNeXt: convnext_tiny, convnext_small, convnext_base, convnext_large
    - DenseNet: densenet121, densenet161, densenet169, densenet201
    - Vision Transformer: vit_b_16, vit_b_32, vit_l_16, vit_l_32
    - Swin Transformer: swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b
    - RegNet: regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf
    - MaxVit: maxvit_t (Hybrid CNN + ViT)
    """
    
    def __init__(self, model_name='resnet50', pretrained=True, num_classes=2):
        """
        Args:
            model_name: 백본 모델 이름 (위 지원 모델 참조)
            pretrained: pretrained weights 사용 여부
            num_classes: 출력 클래스 수 (binary는 2, multi-class는 3 이상)
        """
        super(BinaryClassifier, self).__init__()
        
        self.model_name = model_name
        
        # ==================== ResNet Family ====================
        if model_name == 'resnet18':
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(
                weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(
                weights=models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'resnet152':
            self.backbone = models.resnet152(
                weights=models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
            )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = self._build_classifier(in_features, num_classes)
        
        # ==================== EfficientNet Family ====================
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(
                weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(
                weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(
                weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(
                weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'efficientnet_b5':
            self.backbone = models.efficientnet_b5(
                weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'efficientnet_b6':
            self.backbone = models.efficientnet_b6(
                weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'efficientnet_b7':
            self.backbone = models.efficientnet_b7(
                weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'efficientnet_v2_s':
            self.backbone = models.efficientnet_v2_s(
                weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'efficientnet_v2_m':
            self.backbone = models.efficientnet_v2_m(
                weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'efficientnet_v2_l':
            self.backbone = models.efficientnet_v2_l(
                weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._build_classifier(in_features, num_classes)
        
        # ==================== ConvNeXt Family (Modern CNN) ====================
        elif model_name == 'convnext_tiny':
            self.backbone = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Linear(in_features, num_classes)
            
        elif model_name == 'convnext_small':
            self.backbone = models.convnext_small(
                weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Linear(in_features, num_classes)
            
        elif model_name == 'convnext_base':
            self.backbone = models.convnext_base(
                weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Linear(in_features, num_classes)
            
        elif model_name == 'convnext_large':
            self.backbone = models.convnext_large(
                weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Linear(in_features, num_classes)
        
        # ==================== DenseNet Family ====================
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(
                weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'densenet161':
            self.backbone = models.densenet161(
                weights=models.DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'densenet169':
            self.backbone = models.densenet169(
                weights=models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'densenet201':
            self.backbone = models.densenet201(
                weights=models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = self._build_classifier(in_features, num_classes)
        
        # ==================== Vision Transformer (ViT) ====================
        elif model_name == 'vit_b_16':
            self.backbone = models.vit_b_16(
                weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_features, num_classes)
            
        elif model_name == 'vit_b_32':
            self.backbone = models.vit_b_32(
                weights=models.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_features, num_classes)
            
        elif model_name == 'vit_l_16':
            self.backbone = models.vit_l_16(
                weights=models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_features, num_classes)
            
        elif model_name == 'vit_l_32':
            self.backbone = models.vit_l_32(
                weights=models.ViT_L_32_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_features, num_classes)
        
        # ==================== Swin Transformer ====================
        elif model_name == 'swin_t':
            self.backbone = models.swin_t(
                weights=models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, num_classes)
            
        elif model_name == 'swin_s':
            self.backbone = models.swin_s(
                weights=models.Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, num_classes)
            
        elif model_name == 'swin_b':
            self.backbone = models.swin_b(
                weights=models.Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, num_classes)
            
        elif model_name == 'swin_v2_t':
            self.backbone = models.swin_v2_t(
                weights=models.Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, num_classes)
            
        elif model_name == 'swin_v2_s':
            self.backbone = models.swin_v2_s(
                weights=models.Swin_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, num_classes)
            
        elif model_name == 'swin_v2_b':
            self.backbone = models.swin_v2_b(
                weights=models.Swin_V2_B_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, num_classes)
        
        # ==================== RegNet Family ====================
        elif model_name == 'regnet_y_400mf':
            self.backbone = models.regnet_y_400mf(
                weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'regnet_y_800mf':
            self.backbone = models.regnet_y_800mf(
                weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'regnet_y_1_6gf':
            self.backbone = models.regnet_y_1_6gf(
                weights=models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = self._build_classifier(in_features, num_classes)
            
        elif model_name == 'regnet_y_3_2gf':
            self.backbone = models.regnet_y_3_2gf(
                weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = self._build_classifier(in_features, num_classes)
        
        # ==================== MaxVit (Hybrid CNN + ViT) ====================
        elif model_name == 'maxvit_t':
            self.backbone = models.maxvit_t(
                weights=models.MaxVit_T_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[5].in_features
            self.backbone.classifier[5] = nn.Linear(in_features, num_classes)
            
        else:
            raise ValueError(f"Unsupported model: {model_name}. See docstring for supported models.")
    
    def _build_classifier(self, in_features, num_classes):
        """Build classifier head with GroupNorm for better multi-GPU compatibility
        
        GroupNorm doesn't have batch size dependency, so it works well with DataParallel
        even when some GPUs get small batches.
        """
        return nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GroupNorm(32, 512),  # 32 groups for 512 channels
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.GroupNorm(32, 256),  # 32 groups for 256 channels  
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Forward pass
        
        Args:
            x: 입력 이미지 [batch, 3, H, W]
            
        Returns:
            logits [batch, num_classes]: raw scores (softmax 적용 전)
        """
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Backbone을 freeze하고 classifier만 학습"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 마지막 layer만 학습 가능하게
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """전체 모델 학습 가능하게"""
        for param in self.backbone.parameters():
            param.requires_grad = True


def list_available_models():
    """사용 가능한 모든 모델 이름을 반환
    
    Returns:
        dict: 모델 계열별로 그룹화된 모델 이름들
    """
    return {
        'resnet': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
        'efficientnet': ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
                         'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                         'efficientnet_b6', 'efficientnet_b7'],
        'efficientnet_v2': ['efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'],
        'convnext': ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'],
        'densenet': ['densenet121', 'densenet161', 'densenet169', 'densenet201'],
        'vit': ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32'],
        'swin': ['swin_t', 'swin_s', 'swin_b'],
        'swin_v2': ['swin_v2_t', 'swin_v2_s', 'swin_v2_b'],
        'regnet': ['regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf'],
        'maxvit': ['maxvit_t']
    }


def print_available_models():
    """사용 가능한 모델 목록을 보기 좋게 출력"""
    models = list_available_models()
    print("\n" + "="*70)
    print("Available Models for Binary Classification")
    print("="*70)
    
    for family, model_list in models.items():
        print(f"\n{family.upper()}:")
        for i, model in enumerate(model_list, 1):
            print(f"  {i}. {model}")
    
    print("\n" + "="*70)
    print("Usage: model = create_model('model_name', pretrained=True)")
    print("="*70 + "\n")


def create_model(model_name='resnet50', pretrained=True, num_classes=2, device='cuda', use_multi_gpu=True):
    """모델 생성 헬퍼 함수 (SOTA 모델 지원)
    
    BinaryClassifier를 생성하고 device로 이동, Multi-GPU 설정까지 수행
    
    Args:
        model_name: 백본 모델 이름. 지원 모델:
            - ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
            - EfficientNet: efficientnet_b0~b7, efficientnet_v2_s/m/l
            - ConvNeXt: convnext_tiny/small/base/large
            - DenseNet: densenet121/161/169/201
            - ViT: vit_b_16/32, vit_l_16/32
            - Swin: swin_t/s/b, swin_v2_t/s/b
            - RegNet: regnet_y_400mf/800mf/1_6gf/3_2gf
            - MaxVit: maxvit_t
        pretrained: ImageNet pretrained weights 사용 여부
        num_classes: 출력 클래스 수 (binary=2, multi-class=3+)
        device: 'cuda' or 'cpu'
        use_multi_gpu: 여러 GPU 사용 여부 (DataParallel)
        
    Returns:
        model: 생성된 모델 (device로 이동 완료, Multi-GPU 설정 완료)
        
    Examples:
        >>> # ResNet50 (기본)
        >>> model = create_model('resnet50', pretrained=True, device='cuda')
        
        >>> # EfficientNetV2-M (효율적이고 강력)
        >>> model = create_model('efficientnet_v2_m', pretrained=True, device='cuda')
        
        >>> # ConvNeXt-Base (최신 CNN)
        >>> model = create_model('convnext_base', pretrained=True, device='cuda')
        
        >>> # Swin Transformer (의료 영상에 추천)
        >>> model = create_model('swin_b', pretrained=True, device='cuda')
    """
    model = BinaryClassifier(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
    model = model.to(device)
    
    # Multi-GPU 설정
    if use_multi_gpu and device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    return model

