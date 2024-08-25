import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
import pandas as pd
from PIL import Image

class ModifiedEfficientNetV2(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # 첫 번째 합성곱 레이어 수정
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False),
            *list(original_model.features)[1:]
        )
        
    def forward(self, x):
        return self.features(x)

# 모델 로드 및 수정
original_model = efficientnet_v2_s(pretrained=True)
model = ModifiedEfficientNetV2(original_model)
model.eval()

# 이미지 전처리 함수 (리사이즈 없음)
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    if image.size != (139, 139):
        raise ValueError("입력 이미지의 크기가 139x139여야 합니다.")
    return transform(image).unsqueeze(0)

# 특징 추출 함수
def extract_features(image_tensor):
    with torch.no_grad():
        return model(image_tensor).squeeze().flatten().numpy()

# 이미지 처리 및 특징 추출
image_path = '/content/ISIC_0015670.jpg'  # 139x139 크기의 이미지 경로를 지정하세요
image_tensor = preprocess_image(image_path)
extracted_features = extract_features(image_tensor)

# 데이터프레임 생성 및 저장
df = pd.DataFrame([extracted_features])

print(f"특징 벡터의 크기: {extracted_features.shape}")
