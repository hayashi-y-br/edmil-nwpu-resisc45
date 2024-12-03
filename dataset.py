import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from sklearn.model_selection import StratifiedShuffleSplit


class RESISCBags(Dataset):
    def __init__(self, root='./data', train=True, valid=False, patch_size=32):
        self.root = root
        self.train = train
        self.valid = valid
        self.patch_size = patch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            self.Patchify(self.patch_size)
        ])
        dataset = datasets.ImageFolder(root=self.root, transform=self.transform)
        self.class_to_idx = dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        labels = np.array([label for _, label in dataset])
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=1/7, random_state=42)
        train_valid_indices, test_indices = next(sss_test.split(np.zeros(len(labels)), labels))
        train_valid_labels = labels[train_valid_indices]
        sss_valid = StratifiedShuffleSplit(n_splits=1, test_size=1/6, random_state=42)
        train_indices, valid_indices = next(sss_valid.split(np.zeros(len(train_valid_labels)), train_valid_labels))
        train_indices = train_valid_indices[train_indices]
        valid_indices = train_valid_indices[valid_indices]
        if self.train:
            if self.valid:
                self.dataset = Subset(dataset, valid_indices)
            else:
                self.dataset = Subset(dataset, train_indices)
        else:
            self.dataset = Subset(dataset, test_indices)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    class Patchify(object):
        def __init__(self, patch_size=32):
            self.patch_size = patch_size

        def __call__(self, img):
            c, h, w = img.shape
            img = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
            img = img.permute(1, 2, 0, 3, 4)
            img = img.contiguous().view(-1, c, self.patch_size, self.patch_size)
            return img


if __name__ == '__main__':
    dataset = RESISCBags(train=False)
    for i, (X, y) in enumerate(dataset):
        img = make_grid(X, nrow=8)
        img = img.permute(1, 2, 0)
        img = img * 0.5 + 0.5
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.imshow(img)
        fig.savefig(f'./{dataset.idx_to_class[y]}/img_{i}')
        plt.close(fig)