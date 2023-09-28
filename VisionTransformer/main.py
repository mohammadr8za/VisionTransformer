import torch
from torch import nn
from torch.optim import lr_scheduler
from models import Model
from utils import accuracy_fn, image_manual_transform
from engine import run
from data_preparation import data_preparation
from os.path import join

epochs = 100
batch_size = [8]
learning_rate = [1e-05, 1e-06]
dropout = [0.05]
gamma = [0.995]
lambda_reg = [1e-06]
root = r"D:\mreza\TestProjects\Python\DL\ViT"
experiment_save_root = join(root, "Experiments")

# Device Diagnostic Code
device = "cuda" if torch.cuda.is_available() else "cpu"

config_dict = {"batch_size": None,
               "learning_rate": None,
               "dropout": None,
               "gamma": None,
               "lambda": None}

loss_fn = nn.CrossEntropyLoss()

for bs in batch_size:
    config_dict["batch_size"] = bs

    # Data Preparation
    manual_transform_t, manual_transform_v = image_manual_transform(rotation=True), image_manual_transform(rotation=False)
    train_dataloader, valid_dataloader, classes_name = data_preparation(root=join(root, "Data", "pizza_sushi_steak"),
                                                                        batch_size=config_dict["batch_size"],
                                                                        manual_transform_train=manual_transform_t,
                                                                        manual_transform_valid=manual_transform_v)

    for lr in learning_rate:
        config_dict["learning_rate"] = lr

        for d in dropout:
            config_dict["dropout"] = d

            for g in gamma:
                config_dict["gamma"] = g

                for l in lambda_reg:
                    config_dict["lambda"] = l

                    model = Model.ViT(num_classes=3)

                    optimizer = torch.optim.Adam(params=model.parameters(), lr=config_dict["learning_rate"],
                                                 weight_decay=config_dict["lambda"])

                    decay = lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=config_dict["gamma"])

                    experiment_id = fr"BS{bs}_LR{lr}_D{d}_G{g}_L{l}"
                    experiment_save_path = join(experiment_save_root, "train", experiment_id)

                    run(model=model,
                        epochs=epochs,
                        train_dataloader=train_dataloader,
                        valid_dataloader=valid_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        scheduler=decay,
                        acc_fn=accuracy_fn,
                        device=device,
                        config_dict=config_dict,
                        save_path=experiment_save_path,
                        classes_name=classes_name)
