import time
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data import load_data, Medical_Dataset
from transnetr import Model 

class Config:
   
    SEED = 42
    SIZE = (256, 256)
    BATCH_SIZE = 4
    EPOCHS = 30
    LR = 1e-4
    MOMENTUM = 0.90
    SD_THRESHOLD = 5  
    MODEL_SAVE_PATH = 'transnetr_momentum_90.pth'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seeding(seed: int = 42):
    
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DiceLoss(nn.Module):
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-4) -> torch.Tensor:
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

class SegTrainer:
    
    def __init__(self, model, teacher_model, optimizer, config):
        self.model = model.to(config.DEVICE)
        self.teacher_model = teacher_model.to(config.DEVICE)
        self.optimizer = optimizer
        self.config = config
        self.dice_loss = DiceLoss()
        self.mse_loss = nn.MSELoss()
        
        
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def update_teacher(self, epoch: int):
        
        with torch.no_grad():
            for m_param, t_param in zip(self.model.parameters(), self.teacher_model.parameters()):
                if epoch < self.config.SD_THRESHOLD:
                    t_param.data.copy_(m_param.data)
                else:
                    t_param.data = self.config.MOMENTUM * t_param.data + (1 - self.config.MOMENTUM) * m_param.data

    def train_epoch(self, loader, epoch):
        self.model.train()
        self.teacher_model.eval()
        epoch_loss = 0.0

        for imgs, labels in loader:
            imgs, labels = imgs.to(self.config.DEVICE), labels.to(self.config.DEVICE)
            
            self.optimizer.zero_grad()
            
           
            student_out = self.model(imgs)
            
            
            loss_dice = self.dice_loss(student_out, labels)
            loss_ce = F.binary_cross_entropy_with_logits(student_out, labels)
            total_loss = loss_dice + loss_ce

            
            if epoch >= self.config.SD_THRESHOLD:
                with torch.no_grad():
                    teacher_out = self.teacher_model(imgs)
               
                loss_sd = self.mse_loss(torch.sigmoid(student_out / 4.0), torch.sigmoid(teacher_out / 4.0))
                total_loss += 0.2 * loss_sd

            total_loss.backward()
            self.optimizer.step()
            
            self.update_teacher(epoch)
            epoch_loss += total_loss.item()

        return epoch_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        val_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.config.DEVICE), labels.to(self.config.DEVICE)
            outputs = self.model(imgs)
            val_loss += self.dice_loss(outputs, labels).item()
        return val_loss / len(loader)

def main():
    set_seeding(Config.SEED)

    
    images, masks = load_data("ds/training_images/", "ds/training_masks/")
    train_x, val_x, train_y, val_y = train_test_split(images, masks, test_size=0.1, random_state=Config.SEED)

    train_loader = DataLoader(Medical_Dataset(train_x, train_y, Config.SIZE), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Medical_Dataset(val_x, val_y, Config.SIZE), batch_size=Config.BATCH_SIZE, shuffle=False)

    
    model = Model()
    teacher_model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    
    trainer = SegTrainer(model, teacher_model, optimizer, Config)

    
    best_loss = float('inf')
    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate(val_loader)
        
        duration = time.time() - start_time
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {duration:.2f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"--> Model Saved: {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
