# train.py
import os
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from data_utils import create_data_loaders
from models.model import DiffusionModel
from helpers.checkpoint import save_checkpoint, load_checkpoint
import configs
from tqdm import tqdm

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建数据加载器
train_loader, val_loader, test_loader = create_data_loaders()

# 定义模型
model = DiffusionModel(
    input_channels=configs.input_channels,
    output_channels=configs.output_channels,
    base_channels=configs.base_channels,
    base_channels_multiples=configs.base_channels_multiples,
    apply_attention=configs.apply_attention,
    dropout_rate=configs.dropout_rate,
    time_multiple=configs.time_multiple,
    d_model=configs.d_model,
    num_classes=configs.num_classes
).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=configs.num_epochs)

# 定义 noising schedule
betas = torch.linspace(configs.beta_start, configs.beta_end, configs.num_timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)

# 加载检查点
start_epoch, start_loss = load_checkpoint(model, optimizer, scheduler, configs.checkpoint_path)

# MFDR 参数
def mfdr_adjust_params(epoch, T=2, delta=0.1, gamma=0.01, train_accuracy=0.0, test_accuracy=0.0):
    alpha_k = 1.0
    learning_rate_k = configs.learning_rate
    oversampling_factor = 1.0 

    if epoch % T == 0:
        alpha_k = alpha_k * (1 - delta)

    if train_accuracy - test_accuracy >= 5:
        learning_rate_k = learning_rate_k * (1 - gamma)
        oversampling_factor = oversampling_factor * 1.1 

    return alpha_k, learning_rate_k, oversampling_factor

# 训练和验证循环
for epoch in range(start_epoch, configs.num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # MFDR 动态调整过采样因子
    alpha_k, learning_rate_k, oversampling_factor = mfdr_adjust_params(epoch)
    train_loader, val_loader, test_loader = create_data_loaders()

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{configs.num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, torch.randint(0, configs.num_timesteps, (inputs.size(0),), device=device))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    print(f"Epoch {epoch + 1}/{configs.num_epochs}, Loss: {running_loss / len(train_loader)}, Train Accuracy: {train_accuracy}%")

    # 验证逻辑
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, torch.randint(0, configs.num_timesteps, (inputs.size(0),), device=device))
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val
    print(f"Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {val_accuracy}%")

    # 保存检查点
    save_checkpoint(model, optimizer, scheduler, epoch, running_loss / len(train_loader), configs.checkpoint_path)

    # 早停机制
    current_val_loss = val_loss / len(val_loader)
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        torch.save(model.state_dict(), configs.best_model_path)
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break

    scheduler.step()

# 测试
model.load_state_dict(torch.load(configs.best_model_path))
model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        batch_size = inputs.size(0)
        timesteps = torch.randint(0, configs.num_timesteps, (batch_size,), device=device)
        inputs, labels = inputs.to(device), labels.to(device)
        inputs_noisy = model.add_noise(inputs, timesteps, sqrt_alphas_cumprod)
        outputs = model(inputs_noisy, timesteps)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_accuracy = 100 * correct_test / total_test
print(f'Test Loss: {test_loss / len(test_loader)}, Test Accuracy: {test_accuracy}%')