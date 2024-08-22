import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from model import EfficientNetV2TabRFusionWithVariableTabular

# Assuming EfficientNetV2TabRFusionWithVariableTabular is defined as before

class CombinedDataset(Dataset):
    def __init__(self, df, tabular_cols, id_col, label_col, image_dir, transform=None):
        self.tabular_data = df[tabular_cols].values
        self.isic_ids = df[id_col].values
        self.labels = df[label_col].values
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, idx):
        tabular = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
        isic_id = self.isic_ids[idx]
        image_path = os.path.join(self.image_dir, f"{isic_id}.jpg")  # Assuming .jpg extension, adjust if needed
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tabular, image, label

class TestDataset(Dataset):
    def __init__(self, df, tabular_cols, id_col, image_dir, transform=None):
        self.tabular_data = df[tabular_cols].values
        self.isic_ids = df[id_col].values
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, idx):
        tabular = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
        isic_id = self.isic_ids[idx]
        image_path = os.path.join(self.image_dir, f"{isic_id}.jpg")  # Assuming .jpg extension, adjust if needed
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return tabular, image, isic_id

def create_candidate_set(tabular_data, labels, num_candidates):
    indices = np.random.choice(len(tabular_data), num_candidates, replace=False)
    return torch.tensor(tabular_data[indices], dtype=torch.float32), torch.tensor(labels[indices], dtype=torch.long)

def train_model(model, train_loader, val_loader, candidate_x, candidate_labels, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for batch_tabular, batch_images, batch_labels in train_loader:
            batch_tabular, batch_images, batch_labels = batch_tabular.to(device), batch_images.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_images, batch_tabular, candidate_x, candidate_labels)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_tabular, batch_images, batch_labels in val_loader:
                batch_tabular, batch_images, batch_labels = batch_tabular.to(device), batch_images.to(device), batch_labels.to(device)
                
                outputs = model(batch_images, batch_tabular, candidate_x, candidate_labels)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()

        print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100.*correct/total:.2f}%')

def predict(model, test_loader, candidate_x, candidate_labels, device):
    model.eval()
    predictions = []
    isic_ids = []

    with torch.no_grad():
        for batch_tabular, batch_images, batch_isic_ids in test_loader:
            batch_tabular, batch_images = batch_tabular.to(device), batch_images.to(device)
            
            outputs = model(batch_images, batch_tabular, candidate_x, candidate_labels)
            _, predicted = outputs.max(1)
            
            predictions.extend(predicted.cpu().numpy())
            isic_ids.extend(batch_isic_ids)

    return isic_ids, predictions

if __name__ == "__main__":
    # Load training data
    df = pd.read_csv('train-metadata.csv')
    
    # Define your column names
    id_col = 'isic_id'
    label_col = 'target'

    # Define image directories
    train_image_dir = 'path_to_train_images_folder'
    test_image_dir = 'path_to_test_images_folder'

    # Dynamically select tabular columns
    all_cols = set(df.columns)
    excluded_cols = {id_col, label_col}
    tabular_cols = list(all_cols - excluded_cols)

    print(f"Using the following columns as tabular features: {tabular_cols}")

    # Encode labels
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])

    # Split the data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Define transforms for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets
    train_dataset = CombinedDataset(train_df, tabular_cols, id_col, label_col, train_image_dir, transform)
    val_dataset = CombinedDataset(val_df, tabular_cols, id_col, label_col, train_image_dir, transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create candidate set
    num_candidates = 1000
    candidate_x, candidate_labels = create_candidate_set(df[tabular_cols].values, df[label_col].values, num_candidates)

    # Initialize the model
    tabr_input_dim = len(tabular_cols)
    tabr_hidden_dim = 64
    num_classes = len(le.classes_)
    context_size = 96

    model = EfficientNetV2TabRFusionWithVariableTabular(tabr_input_dim, tabr_hidden_dim, num_classes, num_candidates, context_size)

    # Move model and data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    candidate_x = candidate_x.to(device)
    candidate_labels = candidate_labels.to(device)

    # Train the model
    num_epochs = 10
    train_model(model, train_loader, val_loader, candidate_x, candidate_labels, num_epochs, device)

    print("Training completed!")

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')

    # Load test data
    test_df = pd.read_csv('test-metadata.csv')
    
    # Create test dataset
    test_dataset = TestDataset(test_df, tabular_cols, id_col, test_image_dir, transform)

    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Make predictions
    isic_ids, predictions = predict(model, test_loader, candidate_x, candidate_labels, device)

    # Create a dataframe with predictions
    results_df = pd.DataFrame({
        'isic_id': isic_ids,
        'predicted_label': le.inverse_transform(predictions)  # Convert back to original labels
    })

    # Save predictions to CSV
    results_df.to_csv('predictions.csv', index=False)

    print("Predictions completed and saved to 'predictions.csv'")
