from torch.utils.data import Dataset
from io import BytesIO
import numpy as np
from PIL import Image
import h5py
import cv2

class ISICDataset(Dataset):
    def __init__(self, dataframe, feat, file_path, transforms=None):
        self.df = dataframe
        self.file_path = h5py.File(file_path, mode="r")
        self.transforms = transforms
        self.metadata = self.df[feat].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = row['target']
        isic_id = row['isic_id']
        
        metadata = self.metadata[idx]

        image = np.array(Image.open(BytesIO(self.file_path[isic_id][()])))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)["image"]
        
        return {'image': image, 'target': target, 'metadata': metadata}