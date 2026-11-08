import torch
import torch.nn as nn
import math
# from .common import LayerNorm2d
from .common import LayerNorm2d
from typing import *
import numpy as np

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=256, max_w=256):
        """
        Args:
            d_model (int): d_model % 4 == 0
            max_h (int): max height
            max_w (int): max width
        """
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4, got {d_model}")
            
        pe = torch.zeros(d_model, max_h, max_w) # (d_model, max_h, max_w)
        d_model_half = d_model // 2
        
        div_term = torch.exp(torch.arange(0., d_model_half, 2) * -(math.log(10000.0) / d_model_half))
        
        pos_h = torch.arange(0., max_h).unsqueeze(1) # (max_h, 1)
        pos_w = torch.arange(0., max_w).unsqueeze(1) # (max_w, 1)

        raw_h = pos_h * div_term # (max_h, d_model_half / 2)
        pe[0:d_model_half:2, :, :] = torch.sin(raw_h).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
        pe[1:d_model_half:2, :, :] = torch.cos(raw_h).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
        
        raw_w = pos_w * div_term # (max_w, d_model_half / 2)
        pe[d_model_half::2, :, :]  = torch.sin(raw_w).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)
        pe[d_model_half+1::2, :, :] = torch.cos(raw_w).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x + pe: (B, C, H, W)
        """
        return x + self.pe[:, :x.size(2), :x.size(3)].unsqueeze(0)


class UncertaintyPredictor(nn.Module):
    def __init__(self, feature_dim, channel_dim, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        """
        Args:
            feature_dim (int): input D
            channel_dim (int): output C
            d_model (int): internal hidden dim
            nhead (int): attention heads
            dim_feedforward (int): FFN dim
            dropout (float): dropout rate
        """
        super().__init__()
        self.d_model = d_model
        
        self.proj_kv = nn.Sequential(
            nn.Conv2d(feature_dim, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model)
        )
        
        self.proj_q = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model)
        )

        self.pos_encoder = PositionalEncoding2D(d_model)
        
        self.transformer_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        
        self.proj_out = nn.Conv2d(d_model, channel_dim, kernel_size=1)



    def _to_seq(self, x):
        """ (B, C, H, W) -> (B, H*W, C) """
        return x.flatten(2).transpose(1, 2).contiguous()

    def _to_spatial(self, x, H, W):
        """ (B, H*W, C) -> (B, C, H, W) """
        B, L, C = x.shape
        return x.transpose(1, 2).reshape(B, C, H, W).contiguous()

    def forward(self, image_embedding, global_uncertainty):
        """
        Args:
            image_embedding: (B, feature_dim, H, W)
            global_uncertainty: (H, W)
        Returns:
            uncertainty_embedding: (B, channel_dim, H, W)
        """

        image_embedding = image_embedding.detach()
        B, _, H, W = image_embedding.shape

        # global_uncertainty = 1 - global_uncertainty


        pool = nn.AdaptiveAvgPool2d((H, W))
        
        global_uncertainty = pool(global_uncertainty)

        
        # q_map: (B, 1, H, W)
        q_map = global_uncertainty.expand(B, -1, -1, -1).contiguous()
        # q_spatial: (B, d_model, H, W)
        q_spatial = self.pos_encoder(self.proj_q(q_map)).contiguous()
        # q_seq: (B, H*W, d_model)
        q_seq = self._to_seq(q_spatial)

        # kv_spatial: (B, d_model, H, W)
        kv_spatial = self.pos_encoder(self.proj_kv(image_embedding)).contiguous()
        # kv_seq: (B, H*W, d_model)
        kv_seq = self._to_seq(kv_spatial)
        
        # attn_out_seq: (B, H*W, d_model)
        attn_out_seq = self.transformer_layer(tgt=q_seq, memory=kv_seq)
        
        # attn_out_spatial: (B, d_model, H, W)
        attn_out_spatial = self._to_spatial(attn_out_seq, H, W)
        
        # output: (B, channel_dim, H, W)
        return self.proj_out(attn_out_spatial)

class UncertaintyFeatureExtractor(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        mask_in_chans: int = 16,
        activation: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.uncertainty_Conv = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )  # downsample to 1/4

    def forward(self, uncertainty: torch.Tensor) -> torch.Tensor:

        # uncertainty = 1 - uncertainty

        uncertainty_embeddings = self.uncertainty_Conv(uncertainty)
        
        return uncertainty_embeddings

    

# --- Example Usage & Test ---
if __name__ == "__main__":
    # Hyperparameters
    B, H, W = 4, 64, 64
    IMAGE_FEAT_DIM = 256
    OUTPUT_CHAN_DIM = 256
    
    model = UncertaintyPredictor(
        feature_dim=IMAGE_FEAT_DIM, 
        channel_dim=OUTPUT_CHAN_DIM, 
        d_model=256
    )


    model2 = UncertaintyFeatureExtractor()
    
    img_emb = torch.randn(B, IMAGE_FEAT_DIM, H, W)
    avg_unc = torch.randn(H, W)
    
    un = torch.randn(B, 1, H, W)
    
    try:
        result = model(img_emb, avg_unc)
        result2 = model2(un)
        print(f"Input Shape: {img_emb.shape}")
        print(f"Global Uncertainty Shape: {avg_unc.shape}")
        print(f"Output Shape: {result.shape}") 
        print(f"embedding shape: {result2.shape}")
        # Output should be (4, 128, 64, 64)
    except Exception as e:
        print(f"Execution failed: {e}")