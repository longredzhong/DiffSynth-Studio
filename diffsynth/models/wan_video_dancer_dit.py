import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List
from einops import rearrange
from .wan_video_dit import (
    flash_attention, modulate, sinusoidal_embedding_1d, precompute_freqs_cis_3d,
    rope_apply, RMSNorm, AttentionModule, SelfAttention, CrossAttention,
    GateModule, DiTBlock, MLP, Head
)
from .wan_video_dancer_modules import FactorConv3d, PoseRefNetNoBNV3, DYModule

class WanVideoDancerDiT(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        in_dim_c: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.in_dim_c = in_dim_c
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        # Standard WanModel components
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)

        # SteadyDancer specific components
        self.patch_embedding_fuse = nn.Conv3d(
            in_dim + in_dim_c + in_dim_c, dim, kernel_size=patch_size, stride=patch_size)
        
        self.patch_embedding_ref_c = nn.Conv3d(
            in_dim_c, dim, kernel_size=patch_size, stride=patch_size)

        # Synergistic Pose Modulation Modules
        self.condition_embedding_spatial = DYModule(inp=in_dim_c, oup=in_dim_c)
        
        self.condition_embedding_temporal = nn.Sequential(
            FactorConv3d(in_channels=in_dim_c, out_channels=in_dim_c, kernel_size=(3, 3, 3), stride=1),
            nn.SiLU(),
            FactorConv3d(in_channels=in_dim_c, out_channels=in_dim_c, kernel_size=(3, 3, 3), stride=1),
            nn.SiLU(),
            FactorConv3d(in_channels=in_dim_c, out_channels=in_dim_c, kernel_size=(3, 3, 3), stride=1),
            nn.SiLU(),
            FactorConv3d(in_channels=in_dim_c, out_channels=in_dim_c, kernel_size=(3, 3, 3), stride=1),
        )
        
        self.condition_embedding_align = PoseRefNetNoBNV3(
            in_channels_x=16,
            in_channels_c=16,
            hidden_dim=128,
            num_heads=8
        )

    def patchify(self, x: torch.Tensor):
        # This is handled differently in forward for SteadyDancer
        pass

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                condition: Optional[torch.Tensor] = None,
                ref_x: Optional[torch.Tensor] = None,
                ref_c: Optional[torch.Tensor] = None,
                clip_fea_c: Optional[torch.Tensor] = None,
                **kwargs,
                ):
        """
        x: [B, C, F, H, W]
        timestep: [B]
        context: [B, L, C]
        clip_feature: [B, 257, 1280] (clip_fea_x)
        y: [B, C, F, H, W] (conditional video inputs)
        condition: [B, C, F, H, W] (pose)
        ref_x: [B, C, H, W] (reference image latent)
        ref_c: [B, C, H, W] (reference pose)
        clip_fea_c: [B, 257, 1280] (clip feature of reference pose)
        """
        
        # Time embeddings
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        
        # Context embeddings
        context = self.text_embedding(context)
        
        if self.has_image_input:
            # Handle clip features
            # clip_feature is clip_fea_x
            context_clip_x = self.img_emb(clip_feature) if clip_feature is not None else None
            context_clip_c = self.img_emb(clip_fea_c) if clip_fea_c is not None else None
            
            if context_clip_x is not None:
                context_clip = context_clip_x if context_clip_c is None else context_clip_x + context_clip_c
                context = torch.cat([context_clip, context], dim=1)

        # Prepare inputs
        # x is a tensor [B, C, F, H, W]
        # In SteadyDancer code, x is a list of tensors, but here we assume batch processing
        
        # Temporal Motion Coherence Module
        # condition: [B, C, F, H, W]
        condition_temporal = self.condition_embedding_temporal(condition)
        
        # Spatial Structure Adaptive Extractor
        # condition: [B, C, F, H, W]
        b, c, f, h, w = condition.shape
        condition_reshape = rearrange(condition, 'b c f h w -> (b f) c h w')
        condition_spatial = self.condition_embedding_spatial(condition_reshape)
        condition_spatial = rearrange(condition_spatial, '(b f) c h w -> b c f h w', f=f, b=b)
        
        # Hierarchical Aggregation (1)
        condition_fused = condition + condition_temporal + condition_spatial
        
        # Frame-wise Attention Alignment Unit
        # x_noise_clone needs to be [B, C, F, H, W]
        x_noise_clone = x.clone()
        condition_aligned = self.condition_embedding_align(condition_fused, x_noise_clone)
        
        # Condition Fusion/Injection
        # x: [B, C, F, H, W]
        # condition_fused: [B, C, F, H, W]
        # condition_aligned: [B, C, F, H, W]
        x_fused = torch.cat([x, condition_fused, condition_aligned], dim=1)
        x_emb = self.patch_embedding_fuse(x_fused) # [B, dim, F_p, H_p, W_p]
        
        # Condition Augmentation
        # ref_x: [B, C, H, W] -> [B, C, 1, H, W]
        # ref_c: [B, C, H, W] -> [B, C, 1, H, W]
        if ref_x is not None:
            ref_x_emb = self.patch_embedding(ref_x.unsqueeze(2)) # [B, dim, 1, H_p, W_p]
        if ref_c is not None:
            ref_c_emb = self.patch_embedding_ref_c(ref_c.unsqueeze(2)) # [B, dim, 1, H_p, W_p]
            
        # Concatenate along temporal dimension (dim=2)
        # x_emb: [B, dim, F_p, H_p, W_p]
        # ref_x_emb: [B, dim, 1, H_p, W_p]
        # ref_c_emb: [B, dim, 1, H_p, W_p]
        
        # In SteadyDancer: x = [torch.cat([r, u, v], dim=2) for r, u, v in zip(x, ref_x, ref_c)]
        # Here we assume batch
        x_input = torch.cat([x_emb, ref_x_emb, ref_c_emb], dim=2)
        
        # Flatten and transpose
        # x_input: [B, dim, F_total, H_p, W_p]
        grid_size = torch.tensor(x_input.shape[2:], dtype=torch.long) # [F_total, H_p, W_p]
        x_input = x_input.flatten(2).transpose(1, 2) # [B, L, dim]
        
        # Freqs
        f, h, w = grid_size.tolist()
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        # Blocks
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            x_input = block(x_input, context, t_mod, freqs)

        # Head
        x_out = self.head(x_input, t)
        
        # Unpatchify
        x_out = self.unpatchify(x_out, grid_size)
        
        # Remove reference frames
        # x_out: [B, C, F_total, H, W]
        # We added 1 frame for ref_x and 1 frame for ref_c at the end?
        # Wait, in SteadyDancer: x = [torch.cat([r, u, v], dim=2) for r, u, v in zip(x, ref_x, ref_c)]
        # So ref_x and ref_c are appended at the end? No, `zip(x, ref_x, ref_c)` implies order.
        # Actually `x` in SteadyDancer forward is a list of tensors.
        # `x = [torch.cat([r, u, v], dim=2) for r, u, v in zip(x, ref_x, ref_c)]`
        # Wait, `x` comes from `patch_embedding_fuse`.
        # `ref_x` comes from `patch_embedding`.
        # `ref_c` comes from `patch_embedding_ref_c`.
        # The order in `cat` is `[r, u, v]`. `r` is `x` (fused), `u` is `ref_x`, `v` is `ref_c`.
        # So `x` (fused) is first, then `ref_x`, then `ref_c`.
        # `x` (fused) has `F` frames. `ref_x` has 1 frame. `ref_c` has 1 frame.
        # So total frames = F + 1 + 1 = F + 2.
        # We want to return the first F frames.
        
        real_seq = x.shape[2] # F
        x_out = x_out[:, :, :real_seq, :, :]
        
        return x_out

