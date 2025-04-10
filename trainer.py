import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
from tqdm import tqdm
import os
from model import Diffusion
from data import DataPreprocessor
from visualization import TSNESaver

class DiffusionTrainer:
    def __init__(self, root_dir='./data/NCT', batch_size=16, oversampling_factor=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_processor = DataPreprocessor(root_dir, batch_size, oversampling_factor)
        self.train_loader, self.val_loader, self.test_loader = self.data_processor.prepare_data()
        self.model = Diffusion().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = nn.CrossEntropyLoss()

        self.num_timesteps = 1000
        self.betas = torch.linspace(0.0001, 0.02, self.num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.tsne_saver = TSNESaver()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 8)
        ).to(self.device)

        self.alpha_k = 1.0
        self.learning_rate_k = 1e-4
        self.oversampling_factor = 1.0
        self.current_epoch = 0
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0

    def importance_sampling(self, batch_size):
        weights = (self.alphas_cumprod / (1 - self.alphas_cumprod)).sqrt()
        weights = weights / weights.sum()
        return torch.multinomial(weights, batch_size, replacement=True).to(self.device)

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            t = self.importance_sampling(images.size(0))

            self.optimizer.zero_grad()
            outputs = self.model(images, t)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            t_steps = torch.randint(1, self.num_timesteps, (5,)).tolist()
            features = self.model.extract_features_at_steps(images, t_steps)
            class_logits = []
            for feature in features:
                class_logits.append(self.classifier(feature))
            fused = torch.cat(class_logits, dim=1)
            fused_loss = self.criterion(self.classifier(fused), labels)
            fused_loss.backward()
            self.optimizer.step()

        self.train_accuracy = 100. * correct / total
        return total_loss / len(self.train_loader), self.train_accuracy

    def validate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                t = self.importance_sampling(images.size(0))
                outputs = self.model(images, t)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_accuracy = 100. * correct / total
        return total_loss / len(loader), val_accuracy

    def mfdr_adjust_params(self, epoch, T=2, delta=0.1, gamma=0.01):
        if epoch % T == 0:
            self.alpha_k = self.alpha_k * (1 - delta)

        if self.train_accuracy - self.test_accuracy >= 5:
            self.learning_rate_k = self.learning_rate_k * (1 - gamma)
            self.oversampling_factor = self.oversampling_factor * 1.1

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate_k

        self.train_loader, _, _ = self.data_processor.prepare_data()

    def run(self, epochs=100):
        best_acc = 0
        for epoch in range(epochs):
            self.current_epoch = epoch
            self.mfdr_adjust_params(epoch)
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate(self.val_loader)
            self.scheduler.step()
            self.tsne_saver.save_tsne(self.model, self.val_loader, epoch)
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), "best_model.pth")

            self.test_accuracy = val_acc

        self.model.load_state_dict(torch.load("best_model.pth"))
        test_loss, test_acc = self.validate(self.test_loader)
        print(f"\nFinal Test Accuracy: {test_acc:.2f}%")