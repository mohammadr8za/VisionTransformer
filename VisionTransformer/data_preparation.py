import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
from os.path import join
from os.path import split
from torchvision import transforms
import matplotlib.pyplot as plt
import torch


def find_classes(path):

    class_names = [entry.name for entry in os.scandir(path) if entry.is_dir()]
    class_to_index = {cls: i for i, cls in enumerate(class_names)}

    return class_names, class_to_index


class CustomImageDataset(Dataset):

    def __init__(self,
                 root,
                 csv_file,
                 manual_transform=None):
        self.main_path = root
        self.data_frame = pd.read_csv(join(self.main_path,
                                           csv_file))
        self.transform = manual_transform
        self.classes_name, self.class_to_index = find_classes(join(root, "train"))

    def __len__(self):
        return len(self.data_frame)

    def load_data(self, idx):

        path = join(self.data_frame["path"][idx], self.data_frame["folder"][idx], self.data_frame["data"][idx])
        data = Image.open(path)
        return data

    def __getitem__(self, idx):

        img = self.load_data(idx=idx)
        class_name = self.data_frame["folder"][idx]
        class_idx = self.class_to_index[class_name]

        if self.transform:
            img = self.transform(img)

        return img, class_idx


def data_preparation(root,
                     batch_size: int,
                     manual_transform_train=None,
                     manual_transform_valid=None,
                     manual_transform_test=None,
                     test=False):

    if not test:

        train_dataset = CustomImageDataset(root=root,
                                           csv_file="train_annotation.csv",
                                           manual_transform=manual_transform_train)
        valid_dataset = CustomImageDataset(root=root,
                                           csv_file="valid_annotation.csv",
                                           manual_transform=manual_transform_valid)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)

        classes_name = train_dataset.classes_name

        return train_dataloader, valid_dataloader, classes_name

    if test:

        test_dataset = CustomImageDataset(root=root,
                                          csv_file="test_annotation.csv",
                                          manual_transform=manual_transform_test)

        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)

        classes_name = test_dataset.classes_name

        return test_dataloader, classes_name



if __name__ == "__main__":
    root = r"D:\mreza\Code\Python\DeepLearning\Projects\CustomDatasetBegin\Data\pizza_sushi_steak"
    csv_address = "train_annotation.csv"

    manual_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.RandomVerticalFlip(p=0.5),
                                            transforms.PILToTensor()])

    manual_transforms_v = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.PILToTensor()])

    train_data = CustomImageDataset(root=root,
                                    csv_file=csv_address,
                                    manual_transform=manual_transforms)

    img, label = train_data[20]
    print(f"Shape of image: {img.shape}")

    # Plot a few random samples
    k = 5
    random_numbers = torch.randint(high=train_data.__len__(), size=(k, ))
    fig, axs = plt.subplots(nrows=k, ncols=1, sharex=True, sharey=False)
    for count, idx in enumerate(random_numbers):
        img, label = train_data[idx.item()]
        axs[count].imshow(img.permute(1, 2, 0))
        axs[count].set_title(train_data.classes_name[label])
    fig.tight_layout()
    plt.show()

    train_dataloader = DataLoader(dataset=train_data, batch_size=8)

    root = r"D:\mreza\Code\Python\DeepLearning\Projects\CustomDatasetBegin\Data\pizza_sushi_steak"
    train_dataloader, valid_dataloader, classes_name = data_preparation(root=root,
                                                                        batch_size=8,
                                                                        manual_transform_train=manual_transforms,
                                                                        manual_transform_valid=manual_transforms_v)


