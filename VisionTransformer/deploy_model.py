import torch
import gradio
from models import Model
from torchvision.transforms import transforms
from timeit import default_timer
from PIL import Image
from os.path import join
import pandas as pd
import numpy as np

state_dict = torch.load(r'D:\mreza\TestProjects\Python\DL\ViT\Experiments\train\BS8_LR1e-05_D0.05_G0.995_L1e-06\state\epoch_23.pt')
model = Model.ViT().cpu()
model.load_state_dict(state_dict)

vit_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.PILToTensor()])
classes_name = ['pizza', 'sushi', 'steak']


def prediction(img):

    start_time = default_timer()

    img = vit_transforms(img)

    model.eval()
    with torch.inference_mode():
        probs = torch.softmax(model(img.unsqueeze(dim=0).type(torch.float)), dim=1)

    preds = {classes_name[i]: prob.item() for i, prob in enumerate(probs[0, :])}

    end_time = default_timer()

    return preds, end_time-start_time


# example list:
n_examples = 5
path = r"D:\mreza\TestProjects\Python\DL\ViT\Data\pizza_sushi_steak"
df_data = pd.read_csv(join(path, "test_annotation.csv"))
random_numbers = torch.randint(low=0, high=len(df_data), size=(n_examples, ))
data_list = []
# data_list.append(join(df_data['path'][i], df_data['folder'][i], df_data['data'][i]) for i in random_numbers)
for i in range(n_examples):
    data_list.append(join(df_data['path'][i], df_data['folder'][i], df_data['data'][i]))
title = " mrpeyghan Food Classifier"
description = "Classification of Food Images using Vision Transformer"
article = "Created by mrpeyghan in 2023"

demo = gradio.Interface(fn=prediction,
                        inputs=gradio.Image(type='pil'),
                        outputs=[gradio.Label(num_top_classes=3, label="Predictions"),
                                 gradio.Number(label='time')],
                        examples=data_list,
                        title=title,
                        description=description,
                        article=article)

demo.launch(debug=False,
            share=True)


