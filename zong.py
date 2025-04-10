import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import math
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.multiprocessing as mp
from collections import Counter
from mamba_ssm import Mamba

# 设置文件共享策略
mp.set_sharing_strategy('file_system')


# ------------------ 修正后的模型定义 ------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512):
        super().__init__()
        half_dim = time_emb_dims // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        ts = torch.arange(total_time_steps, dtype=torch.float32)
        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        self.time_block = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(time_emb_dims, time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(time_emb_dims_exp, time_emb_dims_exp)
        )

    def forward(self, time):
        return self.time_block(time)


class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.downsample(x)


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.upsample(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.temporal_fc = nn.Sequential(
            nn.Linear(512, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x, t_emb):
        b, c, _, _ = x.size()
        channel_weights = self.temporal_fc(t_emb).view(b, c, 1, 1)
        return x * channel_weights


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_fc = nn.Linear(time_emb_dims, out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        if apply_attention:
            self.attention = ChannelAttention(out_channels)
            self.use_attention = True
        else:
            self.attention = nn.Identity()
            self.use_attention = False

    def forward(self, x, t_emb):
        h = nn.SiLU()(self.norm1(x))
        h = self.conv1(h)
        h += self.time_fc(nn.SiLU()(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = nn.SiLU()(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        if self.use_attention:
            h = self.attention(h, t_emb)
        else:
            h = self.attention(h)

        return h + self.shortcut(x)


class MultiStageClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(3):
            self.down_layers.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, current_channels * 2, 3, padding=1),
                    nn.BatchNorm2d(current_channels * 2),
                    nn.ReLU(),
                    DownSample(current_channels * 2)
                )
            )
            current_channels *= 2

        self.up_layers = nn.ModuleList()
        for _ in range(3):
            self.up_layers.append(
                nn.Sequential(
                    UpSample(current_channels),
                    nn.Conv2d(current_channels, current_channels // 2, 3, padding=1),
                    nn.BatchNorm2d(current_channels // 2),
                    nn.ReLU()
                )
            )
            current_channels = current_channels // 2

        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(current_channels, num_classes)
        )

    def forward(self, x):
        skips = []
        for layer in self.down_layers:
            x = layer(x)
            skips.append(x)

        for layer, skip in zip(self.up_layers, reversed(skips)):
            x = layer(x + skip)

        return self.final(x)


class Diffusion(nn.Module):
    def __init__(self, input_channels=3, base_channels=32, num_classes=8, d_model=32):
        super().__init__()
        self.time_emb = SinusoidalPositionEmbeddings(time_emb_dims=base_channels)
        self.ddmamba = Mamba(d_model=d_model)  # 添加 Mamba 模块
        # Encoder
        self.encoder = nn.ModuleList([
            ResnetBlock(3, 32, apply_attention=False),
            DownSample(32),
            ResnetBlock(32, 64, apply_attention=True),
            DownSample(64),
            ResnetBlock(64, 128, apply_attention=False)
        ])

        # Decoder with explicit channel control
        self.decoder = nn.ModuleList([
            nn.Sequential(
                ResnetBlock(128, 64),
                UpSample(64)
            ),  # Output: 64 channels
            nn.Sequential(
                ResnetBlock(64, 32),
                UpSample(32)
            ),  # Output: 32 channels
            nn.Sequential(
                ResnetBlock(32, 3),
                UpSample(3)
            )  # Output: 3 channels
        ])

        # Classifiers with matching input channels
        self.stage_classifiers = nn.ModuleList([
            MultiStageClassifier(in_channels=64, num_classes=num_classes),
            MultiStageClassifier(in_channels=32, num_classes=num_classes),
            MultiStageClassifier(in_channels=3, num_classes=num_classes)
        ])

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, t):
        t_emb = self.time_emb(t)

        # Encoder
        skips = []
        for layer in self.encoder:
            if isinstance(layer, ResnetBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
            if isinstance(layer, DownSample):
                skips.append(x)

        # 添加 Mamba 模块
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        x = self.ddmamba(x)
        x = x.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

        # Decoder stages
        stage_outputs = []
        for decoder_stage in self.decoder:
            for layer in decoder_stage:
                if isinstance(layer, ResnetBlock):
                    x = layer(x, t_emb)
                else:
                    x = layer(x)
            stage_outputs.append(x)

        # Channel verification
        assert len(stage_outputs) == 3, f"Expected 3 stages, got {len(stage_outputs)}"
        assert stage_outputs[0].size(1) == 64, f"Stage1 channels: {stage_outputs[0].size(1)}"
        assert stage_outputs[1].size(1) == 32, f"Stage2 channels: {stage_outputs[1].size(1)}"
        assert stage_outputs[2].size(1) == 3, f"Stage3 channels: {stage_outputs[2].size(1)}"

        # Classification
        class_logits = []
        for stage_x, classifier in zip(stage_outputs, self.stage_classifiers):
            class_logits.append(classifier(stage_x))

        # Fusion
        fused = torch.cat(class_logits, dim=1)
        return self.fusion_mlp(fused)

    def extract_features_at_steps(self, x, t_steps):
        features = []
        for t in t_steps:
            t = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)
            x = self.forward(x, t)
            features.append(x)
        return features


# ------------------ 训练逻辑 ------------------
class DiffusionTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader, self.val_loader, self.test_loader = self.prepare_data()
        self.model = Diffusion().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = nn.CrossEntropyLoss()

        self.num_timesteps = 1000
        self.betas = torch.linspace(0.0001, 0.02, self.num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 创建 t-SNE 图保存目录
        self.tsne_dir = "./tSNE"
        os.makedirs(self.tsne_dir, exist_ok=True)

        # 指定九类图例名称
        self.class_names = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

        # 初始化 MFDR 参数
        self.alpha_k = 1.0
        self.learning_rate_k = 1e-4
        self.oversampling_factor = 1.0
        self.current_epoch = 0
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0

        # 添加分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 8)  # 假设有8个类别
        ).to(self.device)

    def prepare_data(self, oversampling_factor=1.0):
        val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 加载原始训练集
        train_dataset = datasets.ImageFolder('./data/train', transform=train_transform)
        val_dataset = datasets.ImageFolder('./data/val', transform=val_transform)
        test_dataset = datasets.ImageFolder('./data/test', transform=val_transform)

        # 过采样
        label_counts = Counter(train_dataset.targets)
        target_count = max(label_counts.values())
        oversampled_train_dataset = self.oversample_dataset(train_dataset, label_counts, target_count, oversampling_factor)

        return (
            DataLoader(oversampled_train_dataset, batch_size=16, shuffle=True, num_workers=2),
            DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2),
            DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
        )

    def oversample_dataset(self, dataset, label_counts, target_count, oversampling_factor=1.0):
        oversampled_datasets = []
        for class_idx in range(len(label_counts)):
            class_samples = [sample for sample in dataset if sample[1] == class_idx]
            class_count = len(class_samples)
            if class_count < target_count:
                num_to_add = int((target_count - class_count) * oversampling_factor)
                augmented_samples = []
                for _ in range(num_to_add):
                    sample = class_samples[np.random.randint(class_count)]
                    augmented_image = self.augmentation_transforms(sample[0])
                    augmented_samples.append((augmented_image, class_idx))
                oversampled_datasets.append(ConcatDataset([dataset, augmented_samples]))
            else:
                oversampled_datasets.append(dataset)
        return ConcatDataset(oversampled_datasets)

    def augmentation_transforms(self, image):
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0), antialias=True),
            transforms.RandomGrayscale(p=0.1),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])(image)

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

            # 随机选择时间步长
            t_steps = torch.randint(1, self.num_timesteps, (5,)).tolist()  # 随机选择5个时间步长
            features = self.model.extract_features_at_steps(images, t_steps)
            class_logits = []
            for feature in features:
                class_logits.append(self.classifier(feature))
            fused = torch.cat(class_logits, dim=1)
            fused_loss = self.criterion(self.fusion_mlp(fused), labels)
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

    def generate_tsne(self, epoch):
        self.model.eval()
        features = []
        labels = []

        # 提取特征
        with torch.no_grad():
            # 禁用多进程加载数据
            single_loader = DataLoader(self.val_loader.dataset, batch_size=16, shuffle=False, num_workers=0)
            for images, batch_labels in tqdm(single_loader, desc="Extracting Features"):
                images = images.to(self.device)
                t = self.importance_sampling(images.size(0))
                outputs = self.model(images, t)
                features.append(outputs.cpu().numpy())
                labels.append(batch_labels.numpy())

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        # 使用 t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features)

        # 绘制 t-SNE 图
        plt.figure(figsize=(10, 8))
        for label in np.unique(labels):
            # 使用指定的类名作为图例
            class_name = self.class_names[label]
            plt.scatter(features_tsne[labels == label, 0], features_tsne[labels == label, 1], label=class_name)
        plt.title(f"t-SNE Visualization (Epoch {epoch + 1})")
        plt.legend()
        plt.savefig(os.path.join(self.tsne_dir, f"tsne_epoch_{epoch + 1}.png"))
        plt.close()

    def mfdr_adjust_params(self, epoch, T=2, delta=0.1, gamma=0.01):
        if epoch % T == 0:
            self.alpha_k = self.alpha_k * (1 - delta)

        if self.train_accuracy - self.test_accuracy >= 5:
            self.learning_rate_k = self.learning_rate_k * (1 - gamma)
            self.oversampling_factor = self.oversampling_factor * 1.1  # 增加过采样程度

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate_k

        self.train_loader = self.prepare_data(oversampling_factor=self.oversampling_factor)

    def run(self, epochs=100):
        best_acc = 0
        for epoch in range(epochs):
            self.current_epoch = epoch

            # 动态调整参数
            self.mfdr_adjust_params(epoch)

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate(self.val_loader)
            self.scheduler.step()

            # 生成 t-SNE 图
            self.generate_tsne(epoch)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), "best_model.pth")

            # 更新测试精度
            self.test_accuracy = val_acc

        self.model.load_state_dict(torch.load("best_model.pth"))
        test_loss, test_acc = self.validate(self.test_loader)
        print(f"\nFinal Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    trainer = DiffusionTrainer()
    trainer.run(epochs=100)