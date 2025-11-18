# Copyright Huawei Technologies Co., Ltd. 2024-2028. All rights reserved.
import torch
import torch.nn.functional as F

import numpy as np


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H, W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    pos = np.squeeze(pos)  # (1, H, W) -> (H, W)
    out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product
    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


# 2D sine-cosine position embedding
def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def recover_navit_subimages_with_pos_emb(
    sub_image_hidden_states,
    attention_mask,
    num_sub_images,
    visual_embedding_group,
    pos_hidden_size,
    thumbnail_only=False,
):
    if num_sub_images < 0:
        num_sub_images = 0
    _slice = int(np.sqrt(num_sub_images))
    _, _, sub_image_depth = sub_image_hidden_states.shape
    _, attention_mask_height, attention_mask_width = attention_mask.shape
    if thumbnail_only is True:
        num_sub_images += 1
    sub_image_hidden_states = sub_image_hidden_states.reshape(
        -1, num_sub_images, attention_mask_height, attention_mask_width, sub_image_depth
    )
    attention_mask = attention_mask.reshape(
        -1, num_sub_images, attention_mask_height, attention_mask_width
    )
    if thumbnail_only is True:
        sub_image_hidden_states = sub_image_hidden_states[:, -1:, :, :, :]
        attention_mask = attention_mask[:, -1:, :, :]
        _slice = 1

    def _infer_ori_image_patch_shape(sub_image_attention_mask):
        ind_h, ind_w = torch.where(sub_image_attention_mask > 0)
        return torch.max(ind_h) + 1, torch.max(ind_w) + 1

    def _pad_to_same(image_hidden):
        _dtype = image_hidden.dtype
        visual_downsample_stride = int(np.sqrt(visual_embedding_group))
        full_h, full_w, _ = image_hidden.shape
        target_h, target_w = attention_mask_height * _slice, attention_mask_width * _slice
        # ensure all contents are included during downsampling
        to_pad_h = (target_h - full_h) + (
            visual_downsample_stride - target_h % visual_downsample_stride
        ) % visual_downsample_stride
        to_pad_w = (target_w - full_w) + (
            visual_downsample_stride - target_w % visual_downsample_stride
        ) % visual_downsample_stride
        # (H,W,D) -> (1,D,H,W) to support replicate padding
        image_hidden = image_hidden.permute(2, 0, 1).unsqueeze(0)
        pad_size = (0, to_pad_w, 0, to_pad_h)
        # (1,D,H,W) -> (H,W,D)
        image_hidden = F.pad(image_hidden.to(torch.float32), pad_size, mode="replicate").squeeze(0).permute(1, 2, 0)
        return image_hidden.to(_dtype)

    image_hidden_states = list()
    valid_image_token = list()
    image_2d_pos = list()
    for batch_id, sub_image_hidden_state in enumerate(sub_image_hidden_states):
        ori_h, ori_w = _infer_ori_image_patch_shape(attention_mask[batch_id][0])
        full_h, full_w = ori_h * _slice, ori_w * _slice
        # (S,H,W,D) -> (S_h,S_w,H,W,D) -> (S_h,H,S_w,W,D) -> (S_h*H,S_w*W,D)
        this_image_hidden = (
            sub_image_hidden_state[:, 0:ori_h, 0:ori_w, :]
            .view(_slice, _slice, ori_h, ori_w, sub_image_depth)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(full_h, full_w, sub_image_depth)
        )
        pos_emb = get_2d_sincos_pos_embed(pos_hidden_size, grid_size_h=full_h, grid_size_w=full_w)  # (H, W, D)
        pos_emb = torch.tensor(pos_emb, dtype=this_image_hidden.dtype, device=this_image_hidden.device)
        image_hidden_states.append(_pad_to_same(this_image_hidden))
        image_2d_pos.append(_pad_to_same(pos_emb))
        valid_image_token.append([full_h, full_w])
    image_hidden_states = torch.stack(image_hidden_states)
    image_2d_pos = torch.stack(image_2d_pos)
    valid_image_token = torch.tensor(valid_image_token, dtype=torch.int64)
    return image_hidden_states, image_2d_pos, valid_image_token


def visiual_token_downsample(
    visual_downsampler, image_hidden_states, valid_image_token, visual_embedding_group, image_2d_pos
):
    if image_2d_pos is not None:
        image_hidden_states = image_hidden_states + image_2d_pos
    image_hidden_states = visual_downsampler(image_hidden_states)
    valid_image_token = torch.ceil(valid_image_token / np.sqrt(visual_embedding_group)).to(torch.int64)
    return image_hidden_states, valid_image_token


def merge_native_qformer(
    clip_embeddings_native_patch,
    valid_image_token_shape,
    clip_embeddings_qformer,
    visual_source_spliter,
    num_sub_images,
):

    def add_split_token_for_qformer_token(qformer_emb):
        # + 1 for thumbnail
        len_per_token = int(qformer_emb.size(0) // (num_sub_images + 1))
        qformer_emb_with_spliter = list()
        for i in range(num_sub_images + 1):
            qformer_emb_with_spliter.append(
                visual_source_spliter(torch.tensor([2 * i]).to(visual_source_spliter.weight.device))
            )
            qformer_emb_with_spliter.append(qformer_emb[i * len_per_token : (i + 1) * len_per_token])
            qformer_emb_with_spliter.append(
                visual_source_spliter(torch.tensor([2 * i + 1]).to(visual_source_spliter.weight.device))
            )
        return torch.cat(qformer_emb_with_spliter, dim=0)

    merged_visual_embeddings = list()
    for batch_id in range(clip_embeddings_native_patch.size(0)):
        h, w = valid_image_token_shape[batch_id]
        native_patch_emb = clip_embeddings_native_patch[batch_id][:h, :w, :].reshape(h * w, -1)
        if clip_embeddings_qformer is not None:
            qformer_emb = clip_embeddings_qformer[batch_id]
            qformer_emb = add_split_token_for_qformer_token(qformer_emb)
            merged_visual_embeddings.append(
                torch.cat(
                    [
                        visual_source_spliter(torch.tensor([10]).to(visual_source_spliter.weight.device)),
                        native_patch_emb,
                        visual_source_spliter(torch.tensor([11]).to(visual_source_spliter.weight.device)),
                        qformer_emb,
                    ],
                    dim=0,
                )
            )
        else:
            merged_visual_embeddings.append(
                torch.cat(
                    [
                        visual_source_spliter(torch.tensor([0]).to(visual_source_spliter.weight.device)),
                        native_patch_emb,
                        visual_source_spliter(torch.tensor([1]).to(visual_source_spliter.weight.device)),
                    ],
                    dim=0,
                )
            )

    return merged_visual_embeddings


def merge_visual_embed(navit980_images, vision_tower, downsampler, visual_source_spliter_emb):
    navit_pixel_values = navit980_images["navit_pixel_values"].npu()
    navit_patch_attention_mask = navit980_images["pixel_attention_mask"].npu()
    clip_visual_outputs = vision_tower(
        pixel_values=navit_pixel_values,
        patch_attention_mask=navit_patch_attention_mask,
    ).last_hidden_state

    super_image_hidden_states, _, valid_image_token_shape = recover_navit_subimages_with_pos_emb(
        clip_visual_outputs,
        navit_patch_attention_mask,
        num_sub_images=-1,
        visual_embedding_group=16,
        pos_hidden_size=4096,
        thumbnail_only=True,
    )

    clip_embeddings_native_patch, valid_image_token_shape = visiual_token_downsample(
        downsampler, super_image_hidden_states, valid_image_token_shape, visual_embedding_group=16, image_2d_pos=None
    )

    merged_visual_embeddings = merge_native_qformer(
        clip_embeddings_native_patch,
        valid_image_token_shape,
        clip_embeddings_qformer=None,
        visual_source_spliter=visual_source_spliter_emb,
        num_sub_images=-1,
    )
    return merged_visual_embeddings
