import torch
from torch import nn
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os
from os.path import join
import numpy as np
from models.Model import ViT
from data_preparation import data_preparation
from utils import image_manual_transform, accuracy_fn


def test(model: nn.Module,
         test_dataloader: torch.utils.data.DataLoader,
         loss_fn: nn.Module,
         acc_fn,
         save_path,
         classes_name):
    model.eval()
    loss, acc = 0, 0
    y_pred, y_true = [], []
    with torch.inference_mode():
        for input_data, target in test_dataloader:
            output = model(input_data)

            loss_batch = loss_fn(output, target)
            acc_batch = acc_fn(output, target)

            loss += loss_batch
            acc += acc_batch

            outputs = output.argmax(1).cpu().numpy()
            y_pred.extend(outputs)
            labels = target.cpu().numpy()
            y_true.extend(labels)

        loss /= len(test_dataloader)
        acc /= len(test_dataloader)

        # Build Confusion Matrix
        classes = classes_name
        cm = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None],
                             index=[i for i in classes],
                             columns=[i for i in classes])
        plt.Figure(figsize=(8, 8))
        sn.heatmap(df_cm, annot=True)
        os.makedirs(join(save_path, "metrics"), exist_ok=True)
        plt.savefig(join(save_path, "metrics", "cm.png"))
        plt.close()

        results_dict = {"acc": acc, "loss": loss}
        results_df = pd.DataFrame(results_dict)
        results_df.to_csv(join(save_path, "metrics", "test_results.csv"))

    return


if __name__ == "__main__":

    root = "D:\mreza\TestProjects\Python\DL\ViT"
    experiment_id = "BS8_LR1e-05_D0.05_G0.995_L1e-06"
    epoch = 10
    load_model_from = join(root, "Experiments", "train", experiment_id, "state", "epoch_" + str(epoch) + ".pt")

    state_dict = torch.load(load_model_from)
    model = ViT()
    model.load_state_dict(state_dict=state_dict)

    manual_transform_test = image_manual_transform(rotation=False)
    test_dataloader, classes = data_preparation(root=join(root, "Data", "pizza_sushi_steak"),
                                                batch_size=8,
                                                manual_transform_test=manual_transform_test,
                                                test=True)

    loss_fn = nn.CrossEntropyLoss()

    save_path = join(root, "Experiments", "test", experiment_id)

    test(model=model,
         test_dataloader=test_dataloader,
         loss_fn=loss_fn,
         acc_fn=accuracy_fn,
         save_path=save_path,
         classes_name=classes)


