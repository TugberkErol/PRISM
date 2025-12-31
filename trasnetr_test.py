import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
from transnetr import Model 

class Evaluator:
    
    
    def __init__(self, device: torch.device):
        self.device = device
        self.epsilon = 1e-15

    def calculate_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        
        
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        
        
        dice = (2. * intersection + self.epsilon) / (y_true.sum() + y_pred.sum() + self.epsilon)
        iou = (intersection + self.epsilon) / (union + self.epsilon)
        precision = (intersection + self.epsilon) / (y_pred.sum() + self.epsilon)
        recall = (intersection + self.epsilon) / (y_true.sum() + self.epsilon)

        return {
            "dice": dice.item(),
            "iou": iou.item(),
            "precision": precision.item(),
            "recall": recall.item()
        }

def preprocess_image(path: str, size: tuple) -> torch.Tensor:
    
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, size)
    img = np.transpose(img, (2, 0, 1)) / 255.0
    return torch.from_numpy(img).unsqueeze(0).float()

def preprocess_mask(path: str, size: tuple) -> torch.Tensor:
    
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, size)
    mask = mask / 255.0
    return torch.from_numpy(mask).unsqueeze(0).float()

def run_test(checkpoint_path: str, data_dir: str, size: tuple = (256, 256)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    model = Model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    
    img_dir = os.path.join(data_dir, "images/")
    mask_dir = os.path.join(data_dir, "masks/")
    img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

    evaluator = Evaluator(device)
    total_metrics = {"dice": 0., "iou": 0., "precision": 0., "recall": 0.}

    print(f"Testing on {len(img_paths)} images from data_c6...")

    with torch.no_grad():
        for x_path, y_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)):
            
            image = preprocess_image(x_path, size).to(device)
            mask = preprocess_mask(y_path, size).to(device)

            
            output = model(image)
            prediction = (torch.sigmoid(output) > 0.5).float()

            
            current_metrics = evaluator.calculate_metrics(mask, prediction)
            for key in total_metrics:
                total_metrics[key] += current_metrics[key]

    
    num_samples = len(img_paths)
    print("\n--- Test Results (PRISM Methodology) ---")
    print(f"Average Dice:      {total_metrics['dice'] / num_samples:.4f}")
    print(f"Average IoU:       {total_metrics['iou'] / num_samples:.4f}")
    print(f"Average Precision: {total_metrics['precision'] / num_samples:.4f}")
    print(f"Average Recall:    {total_metrics['recall'] / num_samples:.4f}")

if __name__ == "__main__":
    CHECKPOINT = "transnetr_momentum_90.pth"
    DATA_PATH = "ds/TestDataset/data_c6/"
    run_test(CHECKPOINT, DATA_PATH)
