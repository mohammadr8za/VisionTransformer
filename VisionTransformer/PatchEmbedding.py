import torch
from torch import nn
from PIL import Image
from os.path import join
from torchvision import transforms


class PatchEmbedding(nn.Module):

    def __init__(self,
                 input_channels: int = 3,
                 embedding_dim: int = 768,
                 patch_size: int = 16):
        super().__init__()

        self.patch_size = patch_size

        self.patcher = nn.Conv2d(in_channels=input_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim=3)

    def forward(self, x):
        # check if resolution is devidable by the patch size
        resolution = x.shape[-1]
        assert resolution % self.patch_size == 0, f"resolution is dividable by the patch size. " \
                                                  f"Resolution: {resolution}| patch_size: {self.patch_size}."

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0, 2, 1)


if __name__ == "__main__":
    sample_dir = r"D:\mreza\Code\Python\DeepLearning\Projects\CustomDatasetBegin\Data\pizza_sushi_steak\train\steak"
    sample_name = "165639"
    sample_image = Image.open(join(sample_dir, sample_name + ".jpg"))

    height, width = 224, 224
    manual_transform = transforms.Compose([transforms.Resize((height, width)),
                                           transforms.PILToTensor()])
    # Transform image from PIL to Tensor and add Batch dimension using Unsqueeze(0). Torch: (B, C, H, W)
    image_tensor = manual_transform(sample_image).unsqueeze(0).type(torch.float)
    patchify = PatchEmbedding()
    image_patch_embedded = patchify(image_tensor)
    print(f"Patch Embedded Image Shape: {image_patch_embedded.shape}")

    # Class Token and Position Embedding
