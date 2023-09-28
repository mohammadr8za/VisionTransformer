import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from os.path import join
import os
import sys
# from torchinfo import summary
parent = os.path.abspath('.')
sys.path.insert(1, parent)
import PatchEmbedding


class MultiHeadSelfAttentionBlock(nn.Module):

    def __init__(self,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 dropout: float = 0):

        super().__init__()
        # Define Layers
        self.norm_layer = nn.LayerNorm(normalized_shape=embed_dim)

        self.attention_layer = nn.MultiheadAttention(embed_dim=embed_dim,
                                                     num_heads=num_heads,
                                                     batch_first=True,
                                                     dropout=dropout)

    def forward(self, x):
        # Define Forward path
        x = self.norm_layer(x)
        # Q, K, V will be modified in each head with W_q, W_k and W_v
        att_output, _ = self.attention_layer(query=x,
                                             key=x,
                                             value=x,
                                             need_weights=False)
        return att_output


class MLPBlock(nn.Module):

    def __init__(self,
                 embed_dim: int = 768,
                 mlp_size: int = 3072,
                 dropout: float = 0):

        super().__init__()

        self.norm_layer = nn.LayerNorm(normalized_shape=embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=mlp_size, out_features=embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        x = self.norm_layer(x)
        mlp_output = self.mlp(x)

        return mlp_output


class TransformerEncoderBlock(nn.Module):

    def __init__(self,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 mlp_size: int = 3072,
                 dropout: float = 0):

        super().__init__()

        self.msa_block = MultiHeadSelfAttentionBlock(embed_dim=embed_dim,
                                                     num_heads=num_heads,
                                                     dropout=dropout)

        self.mlp_block = MLPBlock(embed_dim=embed_dim,
                                  mlp_size=mlp_size,
                                  dropout=dropout)

    def forward(self, x):

        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x

        return x


class ViT(nn.Module):

    def __init__(self,
                 input_channels: int = 3,
                 embed_dim: int = 768,
                 patch_size: int = 16,
                 img_reso: int = 224,
                 num_heads: int = 12,
                 mlp_size: int = 3076,
                 dropout: float = 0,
                 embed_drop: float = 0.1,
                 num_transformer_layer: int = 12,
                 num_classes: int = 3):

        super().__init__()

        # Creating Image Patches
        self.patch_embedding = PatchEmbedding.PatchEmbedding(input_channels=input_channels,
                                                             embedding_dim=embed_dim,
                                                             patch_size=patch_size)

        self.class_embedding = nn.Parameter(torch.randn(size=(1, 1, embed_dim)),
                                            requires_grad=True)

        number_of_patches = int((img_reso * img_reso) / patch_size ** 2)

        self.position_embedding = nn.Parameter(torch.randn(size=(1, number_of_patches + 1, embed_dim)))

        self.embedding_dropout = nn.Dropout(p=embed_drop)

        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_size=mlp_size, dropout=dropout)
                                                   for _ in range(num_transformer_layer)])

        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape=embed_dim),
                                        nn.Linear(in_features=embed_dim, out_features=num_classes))

    def forward(self, x):

        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        # Embedding (Patch, Class and Position)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x

        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)

        x = self.classifier(x[:, 0])

        return x


if __name__ == "__main__":

    sample_dir = r"D:\mreza\Code\Python\DeepLearning\Projects\CustomDatasetBegin\Data\pizza_sushi_steak\train\steak"
    sample_name = "165639"
    sample_image = Image.open(join(sample_dir, sample_name + ".jpg"))

    height, width = 224, 224
    manual_transform = transforms.Compose([transforms.Resize((height, width)),
                                           transforms.PILToTensor()])
    # Transform image from PIL to Tensor and add Batch dimension using Unsqueeze(0). Torch: (B, C, H, W)
    image_tensor = manual_transform(sample_image).unsqueeze(0).type(torch.float)
    patchify = PatchEmbedding.PatchEmbedding()
    image_patch_embedded = patchify(image_tensor)
    print(f"Patch Embedded Image Shape: {image_patch_embedded.shape}")

    att_block = MultiHeadSelfAttentionBlock()
    attention_block_output = att_block(image_patch_embedded)
    print(f"Attention Block Output Shape: {attention_block_output.shape}")

    mlp_block = MLPBlock()
    mlp_block_output = mlp_block(attention_block_output)
    print(f"MLP Block Output Shape: {mlp_block_output.shape}")

    transformer_encoder_block = TransformerEncoderBlock()
    transformer_encoder_block_output = transformer_encoder_block(image_patch_embedded)
    print(f"Transformer Encoder Block Output Shape: {transformer_encoder_block_output.shape}")

    summary(transformer_encoder_block, input_size=(1, 196, 768))

    model = ViT()
    summary(model=model,
            input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )

