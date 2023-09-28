import os

from torchvision import transforms
import matplotlib.pyplot as plt
from os.path import join


def accuracy_fn(outputs, targets):

    acc_count = (outputs.argmax(dim=1) == targets).sum()

    return acc_count / targets.shape[0]


def image_manual_transform(img_size: int = 224, rotation: bool = False):

    if rotation:
        manual_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                               transforms.RandomRotation(0.5),
                                               transforms.PILToTensor()])
    if not rotation:
        manual_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                               transforms.PILToTensor()])

    return manual_transform


def plot_metrics(loss_train, acc_train, loss_valid, acc_valid, save_path):

    os.makedirs(join(save_path, "metrics"), exist_ok=True)

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].plot(loss_train, label="train_loss")
    ax[0].plot(loss_valid, label="valid_loss")
    ax[0].set_title("loss")
    ax[0].legend()

    ax[1].plot(acc_train, label="train_acc")
    ax[1].plot(acc_valid, label="valid_acc")
    ax[1].set_title("accuracy")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(join(save_path, "metrics", "accuracy.png"))
    plt.close()

