import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

class TSNESaver:
    def __init__(self):
        self.tsne_dir = "./tSNE"
        os.makedirs(self.tsne_dir, exist_ok=True)
        self.class_names = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

    def save_tsne(self, model, val_loader, epoch):
        model.eval()
        features = []
        labels = []

        with torch.no_grad():
            single_loader = DataLoader(val_loader.dataset, batch_size=16, shuffle=False, num_workers=0)
            for images, batch_labels in tqdm(single_loader, desc="Extracting Features"):
                images = images.to(model.device)
                t = model.importance_sampling(images.size(0))
                outputs = model(images, t)
                features.append(outputs.cpu().numpy())
                labels.append(batch_labels.numpy())

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))
        for label in np.unique(labels):
            plt.scatter(features_tsne[labels == label, 0], features_tsne[labels == label, 1], label=self.class_names[label])
        plt.title(f"t-SNE Visualization (Epoch {epoch + 1})")
        plt.legend()
        plt.savefig(os.path.join(self.tsne_dir, f"tsne_epoch_{epoch + 1}.png"))
        plt.close()