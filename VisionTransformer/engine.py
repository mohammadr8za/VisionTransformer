import os
from utils import plot_metrics
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data
import wandb
from torch import nn
from tqdm.auto import tqdm
import wandb
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sn
from os.path import join
from torch.utils.tensorboard import SummaryWriter


def run(model: nn.Module,
        epochs: int,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        acc_fn,
        device: torch.device,
        config_dict,
        save_path,
        classes_name):

    #wandb.login(key='69d527074fdc79dd4aaa3b21a23f6c00e3d918e5')

    # wandb.init(
    #     project="VisionTransformer",
    #     config=config_dict
    # )
    writer = SummaryWriter(join(save_path, "runs"))

    loss_train, acc_train, loss_valid, acc_valid = [], [], [], []
    for epoch in tqdm(range(epochs)):

        print(f"\n_________ Epoch: {epoch + 1} _________")
        print(f"BS: {config_dict['batch_size']}| LR: {optimizer.param_groups[0]['lr']}|"
              f" Dropout: {config_dict['dropout']}| Gamma: {config_dict['gamma']}|"
              f" Lambda: {config_dict['lambda']}| Device: {device}")

        model.to(device)

        # T r a i n
        loss_t, acc_t = 0, 0
        for batch_count, (input_data, target) in enumerate(train_dataloader):
            input_data, target = input_data.to(device), target.to(device)

            output = model(input_data.type(torch.float))
            loss_batch = loss_fn(output, target)
            loss_t += loss_batch

            acc_batch = acc_fn(output, target)
            acc_t += acc_batch

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            if batch_count % 100 == 0:
                print(f"{batch_count}-th batch loss in epoch {epoch} equals {loss_batch}")

        loss_t /= len(train_dataloader)
        acc_t /= len(train_dataloader)

        # V a l i d a t i o n
        model.eval()
        loss_v, acc_v = 0, 0
        y_pred, y_true = [], []  # Confusion Matrix
        with torch.inference_mode():
            for input_data, target in valid_dataloader:
                input_data, target = input_data.to(device), target.to(device)

                output = model(input_data.type(torch.float))

                loss_batch = loss_fn(output, target)
                loss_v += loss_batch

                acc_batch = acc_fn(output, target)
                acc_v += acc_batch

                # Build Confusion Matrix
                # classes = ("Pizza", "Sushi", "Steak")

                outputs = output.argmax(1).cpu().numpy()
                y_pred.extend(outputs)
                labels = target.cpu().numpy()
                y_true.extend(labels)

            loss_v /= len(valid_dataloader)
            acc_v /= len(valid_dataloader)

        scheduler.step()

        print(f"Train Loss: {loss_t} \nTrain Acc: {acc_t} \nValid Loss: {loss_v} \nValid Acc: {acc_v}")

        loss_train.append(loss_t.item())
        acc_train.append(acc_t.item())
        loss_valid.append(loss_v.item())
        acc_valid.append(acc_v.item())

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

        # LOSS and ACCURACY
        plot_metrics(loss_train=loss_train,
                     acc_train=acc_train,
                     loss_valid=loss_valid,
                     acc_valid=acc_valid,
                     save_path=save_path)

        # # Weight and Bias Metrics and Log
        # wandb_loss = {"Loss/train": loss_t,
        #               "Loss/valid": loss_v}
        # wandb_acc = {"Accuracy/train": acc_t,
        #              "Accuracy/valid": acc_v}
        # wandb.log({**wandb_loss, **wandb_acc})

        writer.add_scalars(main_tag="LOSS",
                           tag_scalar_dict={"train": loss_t,
                                            "valid": loss_v},
                           global_step=epoch)

        writer.add_scalars(main_tag="ACCURACY",
                           tag_scalar_dict={"train": acc_t,
                                            "valid": acc_v},
                           global_step=epoch)
        # Save Model State_Dict
        print(f"Saving Checkpoint ...")
        state_dict = model.state_dict()
        os.makedirs(join(save_path, "state"), exist_ok=True)
        torch.save(obj=state_dict, f=join(save_path, "state", "epoch_" + f"{epoch}" + ".pt"))
        print(f"Saved!")

    # wandb.finish()
    writer.close()

    return


