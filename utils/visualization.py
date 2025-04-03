import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm

from utils import data


def generate_tsne(model, val_loader, epoch, tsne_dir, class_names, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        single_loader = data.DataLoader(val_loader.dataset, batch_size=16, shuffle=False, num_workers=0)
        for images, batch_labels in tqdm(single_loader, desc="Extracting Features"):
            images = images.to(device)
            t = torch.randint(0, 1000, (images.size(0),)).to(device)
            outputs = model(images, t)
            features.append(outputs.cpu().numpy())
            labels.append(batch_labels.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        plt.scatter(features_tsne[labels == label, 0], features_tsne[labels == label, 1], label=class_names[label])
    plt.title(f"t-SNE Visualization (Epoch {epoch + 1})")
    plt.legend()
    plt.savefig(os.path.join(tsne_dir, f"tsne_epoch_{epoch + 1}.png"))
    plt.close()