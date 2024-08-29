import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from pytorch_widedeep.models import TabTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 데이터 로드
data = pd.read_excel('test.xlsx')

# 타겟 변수 분리
target = data['target'].values
features = data.drop(['target'], axis=1)

# 범주형 변수와 연속형 변수 구분
cat_cols = features.select_dtypes(include=['object', 'category']).columns
cont_cols = features.select_dtypes(include=['int64', 'float64']).columns

print("Categorical columns:", cat_cols)
print("Continuous columns:", cont_cols)

# 결측치 처리
features[cat_cols] = features[cat_cols].fillna('Unknown')
features[cont_cols] = features[cont_cols].fillna(features[cont_cols].median())

# 범주형 변수 인코딩
le = LabelEncoder()
for col in cat_cols:
    features[col] = le.fit_transform(features[col].astype(str))

# 데이터 전처리
scaler = StandardScaler()
features[cont_cols] = scaler.fit_transform(features[cont_cols])

# NaN 값과 무한대 값 처리
features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

# 데이터를 numpy 배열로 변환
X = features.values.astype(np.float32)
y = target.astype(np.float32)

# FT-Transformer 모델 설정
def create_model(features):
    model = TabTransformer(
        column_idx={k: v for v, k in enumerate(features.columns)},
        cat_embed_input=[(col, len(features[col].unique())) for col in cat_cols],
        continuous_cols=list(cont_cols),
        n_heads=4,
        n_blocks=4,
        mlp_hidden_dims=[64, 32],
        mlp_dropout=0.1,
        embed_continuous_method=None
    )
    return model

# 128차원 벡터 생성을 위한 추가 레이어
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 최종 출력 레이어
class OutputLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.fc(x)

# 모델 학습 함수
def train_model(model, feature_extractor, output_layer, train_loader, test_loader, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    feature_extractor.to(device)
    output_layer.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(feature_extractor.parameters()) + list(output_layer.parameters()), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        feature_extractor.train()
        output_layer.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            features = feature_extractor(outputs)
            final_outputs = output_layer(features)
            loss = criterion(final_outputs, batch_y)
            
            if torch.isnan(loss):
                print("NaN loss encountered. Skipping batch.")
                continue
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Train Loss: {avg_loss:.4f}')
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            feature_extractor.eval()
            output_layer.eval()
            with torch.no_grad():
                test_loss = 0
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    features = feature_extractor(outputs)
                    final_outputs = output_layer(features)
                    test_loss += criterion(final_outputs, batch_y).item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss/len(test_loader):.4f}')

    return model, feature_extractor, output_layer

# K-Fold 교차 검증
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}/{n_splits}")
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = create_model(features)
    feature_extractor = FeatureExtractor(model.output_dim)
    output_layer = OutputLayer(128)

    model, feature_extractor, output_layer = train_model(model, feature_extractor, output_layer, train_loader, test_loader)

    # 특성 추출
    model.eval()
    feature_extractor.eval()
    with torch.no_grad():
        all_features = torch.FloatTensor(X).to(next(model.parameters()).device)
        model_outputs = model(all_features)
        fold_features = feature_extractor(model_outputs).cpu().numpy()

    print(f"Fold {fold+1} features shape:", fold_features.shape)
    np.save(f'tabular_features_fold_{fold+1}.npy', fold_features)
    print(f"Fold {fold+1} features saved to 'tabular_features_fold_{fold+1}.npy'")

    # 모델 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_extractor_state_dict': feature_extractor.state_dict(),
        'output_layer_state_dict': output_layer.state_dict()
    }, f'ft_transformer_model_fold_{fold+1}.pth')
    print(f"Fold {fold+1} model saved to 'ft_transformer_model_fold_{fold+1}.pth'")

print("K-Fold cross validation completed.")
