# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   mae.py
    Time:        2022/09/29 20:05:46
    Editor:      Figo
-----------------------------------
'''

from requests import patch
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from vit_pytorch.vit import Transformer
from vit_pytorch.vit import ViT


class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        #! extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = encoder                                                  # 论文选择了ViT作为编码器(这里以ViT-B为例)
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]             # num_patches, encoder_dim = 197, 768

        """
            :: self.to_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            :: self.patch_to_emb = nn.Linear(patch_dim, dim),                                                               
        """
        self.patch_h, self.patch_w = encoder.patch_h, encoder.patch_w
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]              # pixel_values_per_patch = 768

        #! decoder parameters
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()          #! 对齐encoder输出和decoder接受的维度
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(                                                                                     #! 编码器为单个Transformer block
            dim=decoder_dim, 
            depth=decoder_depth, 
            heads=decoder_heads, 
            dim_head=decoder_dim_head, 
            mlp_dim=decoder_dim * 4
        )
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device
        b, c, h, w = img.shape
        # get patches
        patches = self.to_patch(img)                            #! torch.Size([1, 196, 768])
        batch, num_patches, *_ = patches.shape                  

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)                     #! torch.Size([1, 196, 786]) 这里是786，786是预先设定ViT-B的dim，而patches的维度是768
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]            #! 从1开始数196个，到第197.其中第一维放cls_embeding

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]       #! 选择mask和un-mask的编号

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]                                  #! 保留unmask-token进行图像信息的编码处理（送入Encoder）

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]                           #! 保留mask-token，进行后续的重构损失计算 (MSE-Loss)

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)                               #! 将unmask-tokens送入ViT-B的Transformer部分进行编码

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)                                #! 对齐Encoder输出 和 Decoder输入

        # reapply decoder position embedding to unmasked tokens
        decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)        #! Decoder同理也要加入位置编码信息

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)                #! masked的tokens同理也要加入位置编码信息

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)                                   #! 解码的时候对 (mask-token + unmask-token)的合并在一起送入Decoder解码


        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[:, :num_masked]                                    #! 将预测的mask-token部分取出来计算重构损失
        pred_pixel_values = self.to_pixels(mask_tokens)                                 #! tokens转化到像素空间

        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

        """
            :: inplace-error: 这里涉及新值在原地址上赋给旧值的操作，属于inplace错误，本质上和 x += tensor 是一致的
            :: 解决办法：如果不想共享，也就是不想被inplace影响，先clone()再view()
            :: e.x.
            >>> recons_patches[batch_range, masked_indices] = pred_pixel_values
            >>> patches[batch_range, masked_indices] = masked_patches
            >>> one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [196, 768]] is at version 1; 
            >>> expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, 
            >>> with torch.autograd.set_detect_anomaly(True).
        """

        # reconstruction image
        recons_patches = patches.clone()
        recons_patches[batch_range, masked_indices] = pred_pixel_values                 #! Error occurs!
        recons_img = recons_patches.view(
            b, h // self.patch_h, w // self.patch_w, 
            self.patch_h, self.patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

        # mask image
        masked_ = torch.randn_like(masked_patches, device=masked_patches.device)
        mask_patches = patches.clone()
        mask_patches[batch_range, masked_indices] = masked_                             #! Error occurs!
        patches_to_img = mask_patches.view(
            b, h // self.patch_h, w // self.patch_w, 
            self.patch_h, self.patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

        return recon_loss, recons_img, patches_to_img


if __name__ == "__main__":

    encoder = ViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072, 
        dropout=0.1,
        emb_dropout=0.1
    )
    
    model = MAE(
        encoder=encoder, 
        decoder_dim=128,
        masking_ratio=0.75,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64
    )

    images = torch.randn(1, 3, 224, 224)

    recon_loss, recons_img, patches_to_img = model(images)

    recon_loss.mean().backward()