# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.log import logger


class MLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        try:
            x = (x - u) / torch.sqrt(s + self.eps)
        except ZeroDivisionError:
            logger.error("LayerNorm2d zero division error")
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        downsample_channels: Tuple[int, ...] = (512, 1024),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
            downsample_channels (list): Channels for downsampling layers.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        in_channels = out_chans
        downsamples = []
        for out_channels in downsample_channels:
            downsamples.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            in_channels = out_channels
        self.downsamples = nn.Sequential(*downsamples)

        self.sam_hd = True
        if self.sam_hd:
            self.hd_alpha_downsamples = nn.Parameter(torch.zeros(1))
            self.neck_hd = copy.deepcopy(self.neck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        global_features = []
        for _, blk in enumerate(self.blocks):
            x = blk(x)
            if self.sam_hd and blk.window_size == 0:
                global_features.append(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        x_dtype = x.dtype
        x = F.interpolate(
            x.float(), size=(96, 96), mode="bilinear", align_corners=False
        ).to(x_dtype)
        x = self.downsamples(x)

        if self.sam_hd:
            first_global_feature = self.neck_hd(global_features[0].permute(0, 3, 1, 2))
            x_dtype = first_global_feature.dtype
            first_global_feature = F.interpolate(
                first_global_feature.float(),
                size=(96, 96),
                mode="bilinear",
                align_corners=False,
            )
            first_global_feature = self.downsamples(first_global_feature.to(x_dtype))
            x = x + first_global_feature * self.hd_alpha_downsamples

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        pad_hw = None
        h = None
        w = None
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            h, w = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (h, w))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(b, h * w, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, b * self.num_heads, h * w, -1).unbind(0)

        def do_attention(q, k, v):
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if self.use_rel_pos:
                args = RelPosArgs(
                    attn=attn,
                    q=q,
                    rel_pos_h=self.rel_pos_h,
                    rel_pos_w=self.rel_pos_w,
                    q_size=(h, w),
                    k_size=(h, w)
                )
                attn = add_decomposed_rel_pos(args=args)

            attn = attn.softmax(dim=-1)
            x = (
                (attn @ v)
                .view(b, self.num_heads, h, w, -1)
                .permute(0, 2, 3, 1, 4)
                .reshape(b, h, w, -1)
            )

            return x

        x = do_attention(q, k, v)
        x = self.proj(x)

        return x


def window_partition(
        x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    b, h, w, c = x.shape

    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w

    x = x.view(b, hp // window_size, window_size, wp // window_size, window_size, c)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    )
    return windows, (hp, wp)


def window_unpartition(
        windows: torch.Tensor,
        window_size: int,
        pad_hw: Tuple[int, int],
        hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    hp, wp = pad_hw
    h, w = hw
    b = windows.shape[0] // (hp * wp // window_size // window_size)
    x = windows.view(
        b, hp // window_size, wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, hp, wp, -1)

    if hp > h or wp > w:
        x = x[:, :h, :w, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    q_coords = None
    relative_coords = None
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    if q_size != 0:
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)

    if k_size != 0:
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


@dataclass
class RelPosArgs:
    """
    Args:
    attn (Tensor): attention map.
    q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
    rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
    rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
    q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
    k_size (Tuple): spatial sequence size of key k with (k_h, k_w).
    """
    attn: torch.Tensor
    q: torch.Tensor
    rel_pos_h: torch.Tensor
    rel_pos_w: torch.Tensor
    q_size: Tuple[int, int]
    k_size: Tuple[int, int]


def add_decomposed_rel_pos(args: RelPosArgs) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = args.q_size
    k_h, k_w = args.k_size
    rh = get_rel_pos(q_h, k_h, args.rel_pos_h)
    rw = get_rel_pos(q_w, k_w, args.rel_pos_w)

    b, _, dim = args.q.shape
    r_q = args.q.reshape(b, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, rw)

    attn = (
            args.attn.view(b, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, None]
            + rel_w[:, :, :, None, :]
    ).view(b, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16),
            stride: Tuple[int, int] = (16, 16),
            padding: Tuple[int, int] = (0, 0),
            in_chans: int = 3,
            embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


@dataclass
class SAMViTCfg:
    image_size: Union[Tuple[int, int], int] = 1024
    width: int = 1024
    layers: int = 23
    heads: int = 16
    patch_size: int = 16
    window_size: int = 14
    prompt_embed_dim: int = 256
    global_attn_indexes: Union[List[int], Tuple[int]] = (5, 11, 17, 23)
    downsample_channels: Union[List[int], Tuple[int]] = (512, 1024)


SAM_MODEL_CONFIG = {
    "sam_vit_b": {
        "width": 768,
        "layers": 12,
        "heads": 12,
        "global_attn_indexes": [2, 5, 8, 11],
        "downsample_channels": (),
    },
    "sam_b_downsample": {
        "width": 768,
        "layers": 12,
        "heads": 12,
        "global_attn_indexes": [2, 5, 8, 11],
        "downsample_channels": (512, 1024),
    },
    "sam_vit_l": {
        "width": 1024,
        "layers": 24,
        "heads": 16,
        "global_attn_indexes": [5, 11, 17, 23],
        "downsample_channels": (),
    },
    "sam_vit_h": {
        "width": 1280,
        "layers": 32,
        "heads": 16,
        "global_attn_indexes": [7, 15, 23, 31],
        "downsample_channels": (),
    },
}


def create_sam_vit(
        model_name: str = "sam_b_downsample",
        image_size: int = 1024,
        ckpt_path: str = "",
        **kwargs,
):
    sam_cfg = SAMViTCfg(**SAM_MODEL_CONFIG[model_name])
    image_encoder = ImageEncoderViT(
        depth=sam_cfg.layers,
        embed_dim=sam_cfg.width,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=sam_cfg.heads,
        patch_size=sam_cfg.patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=sam_cfg.global_attn_indexes,
        window_size=14,
        out_chans=sam_cfg.prompt_embed_dim,
        downsample_channels=sam_cfg.downsample_channels,
    )

    if ckpt_path:
        state_dict = torch.load(ckpt_path)
        image_encoder.load_state_dict(state_dict, strict=False)
        logger.info(f"SAM-ViT restores from {ckpt_path}")

    return image_encoder
