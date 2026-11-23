import os
from PIL import Image
import torch
import torchvision.transforms as T

class PlacaDataset:
    def __init__(self, path, df,yolo,transform=None):
        self.df = df
        self.transform = transform
        self.path = path
        self.yolo = yolo

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        image = Image.open(os.path.join(self.path, self.df['image'][idx])).convert('L')
        results = self.yolo(image, verbose=False)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()

            if len(boxes) > 0:

                x1, y1, x2, y2 = map(int, boxes[0])

                padding = 5

                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)

                x2 = min(image.width, x2 + padding)
                y2 = min(image.height, y2 + padding)
                roi = image.crop([x1,y1,x2,y2])
            else:
                roi = image
        
        label = torch.tensor(data['label'], dtype=torch.int32)

        if self.transform is not None:
            roi = self.transform(roi)

        return roi, label


