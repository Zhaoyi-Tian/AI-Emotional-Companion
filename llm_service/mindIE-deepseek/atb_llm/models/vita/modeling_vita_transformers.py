# coding=utf-8
# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved. 
# The below software and/or models in this distribution may have been modified 
# by THL A29 Limited ("Tencent Modifications"). 
# All Tencent Modifications are Copyright (C) THL A29 Limited.

import math
from dataclasses import dataclass

from distutils.util import strtobool as dist_strtobool
import torch
import torch.nn as nn


@dataclass
class ChunkParams:
    xs: torch.Tensor
    masks: torch.Tensor
    use_dynamic_chunk: bool
    use_dynamic_left_chunk: bool
    decoding_chunk_size: int
    static_chunk_size: int
    num_decoding_left_chunks: int


def strtobool(x):
    return bool(dist_strtobool(x))


def subsequent_chunk_mask(
    size: int,
    ck_size: int,
    num_l_cks: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_l_cks < 0:
            start = 0
        else:
            start = max((i // ck_size - num_l_cks) * ck_size, 0)
        ending = min((i // ck_size + 1) * ck_size, size)
        ret[i, start:ending] = True
    return ret


def add_optional_chunk_mask(chunkparams):  
    xs = chunkparams.xs
    masks = chunkparams.masks
    use_dynamic_chunk = chunkparams.use_dynamic_chunk
    use_dynamic_left_chunk = chunkparams.use_dynamic_left_chunk
    decoding_chunk_size = chunkparams.decoding_chunk_size
    static_chunk_size = chunkparams.decoding_chunk_size
    num_decoding_left_chunks = chunkparams.num_decoding_left_chunks
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_l_cks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_l_cks = num_decoding_left_chunks
        else:
            chunk_size = torch.randint(1, max_len, (1,)).item()
            num_l_cks = -1
            if chunk_size > max_len // 2:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_l_cks = torch.randint(0, max_left_chunks, (1,)).item()
        ck_masks = subsequent_chunk_mask(
            xs.size(1), chunk_size, num_l_cks, xs.device
        )  # (L, L)
        ck_masks = ck_masks.unsqueeze(0)  # (1, L, L)
        ck_masks = masks & ck_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_l_cks = num_decoding_left_chunks
        ck_masks = subsequent_chunk_mask(
            xs.size(1), static_chunk_size, num_l_cks, xs.device
        )  # (L, L)
        ck_masks = ck_masks.unsqueeze(0)  # (1, L, L)
        ck_masks = masks & ck_masks  # (B, L, L)
    else:
        ck_masks = masks
    return ck_masks


def repeat(number, fn):
    """Repeat module N times.

    :param int N: repeat time
    :param function fn: function to generate module
    :return: repeated modules
    :rtype: MultiSequential
    """
    return MultiSequential(*[fn(n) for n in range(number)])


class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, x, masks, pos_emb):

        """Repeat."""
        for m in self:
            x, masks, pos_emb = m(x, masks, pos_emb)
        return x, masks, pos_emb

    @torch.jit.export
    def infer(self, x, pos_emb, buffer, buffer_index, buffer_out):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        """Repeat."""
        for m in self:
            x, pos_emb, buffer, buffer_index, buffer_out = m.infer(
                x, pos_emb, buffer, buffer_index, buffer_out
            )
        return x, pos_emb, buffer, buffer_index, buffer_out

    @torch.jit.export
    def infer_hidden(self, x, pos_emb, buffer, buffer_index, buffer_out, hidden_out):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        """Repeat."""
        for m in self:
            x, pos_emb, buffer, buffer_index, buffer_out = m.infer(
                x, pos_emb, buffer, buffer_index, buffer_out
            )
            hidden_out.append(x)
        return x, pos_emb, buffer, buffer_index, buffer_out, hidden_out


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, n_head, n_feat, dropout_rate, chunk_size, left_chunks, pos_enc_class):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        if n_feat % n_head != 0:
            raise ValueError("n-feat must be divisible by n-head")
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.min_value = float(torch.finfo(torch.float16).min)
        # chunk par
        if chunk_size > 0 and left_chunks > 0:  # for streaming mode
            self.buffersize = chunk_size * (left_chunks)
            self.left_chunk_size = chunk_size * left_chunks
        else:  # for non-streaming mode
            self.buffersize = 1
            self.left_chunk_size = 1
        self.chunk_size = chunk_size

        # encoding setup
        if pos_enc_class == "rel-enc":
            self.rel_enc = True
            self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
            # these two learnable bias are used in matrix c and matrix d
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
            torch.nn.init.xavier_uniform_(self.pos_bias_u)
            torch.nn.init.xavier_uniform_(self.pos_bias_v)
        else:
            self.rel_enc = False
            self.linear_pos = nn.Identity()
            self.pos_bias_u = torch.tensor([0])
            self.pos_bias_v = torch.tensor([0])

        # buffer
        self.key_buffer_size = 1 * self.h * self.buffersize * self.d_k
        self.value_buffer_size = 1 * self.h * self.buffersize * self.d_k
        if self.chunk_size > 0:
            self.buffer_mask_size = 1 * self.h * self.chunk_size * self.buffersize
        else:
            self.buffer_mask = torch.ones([1, self.h, 1, 1], dtype=torch.bool)

    @torch.jit.unused
    def rel_shift(self, x, zero_triu: bool = False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """

        zero_pad = torch.zeros(
            (x.size()[0], x.size()[1], x.size()[2], 1), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
        return x

    @torch.jit.export
    def forward(self, query, key, value, mask=None, pos_emb=torch.tensor(1.0)):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], Tensor) -> Tensor
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        if self.rel_enc:
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)
            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb.to(query.dtype)).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)  # (batch, head, time1, d_k)
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
            # compute attention score
            # first compute matrix a and matrix c
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
            # compute matrix b and matrix d
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            # Remove rel_shift since it is useless in speech recognition,
            # and it requires special attention for streaming.
            scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, self.min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)

        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

    @torch.jit.export
    def infer(self, query, key, value, pos_emb, buffer, buffer_index, buffer_out):
        n_batch = query.size(0)
        q = (self.linear_q(query).view(n_batch, -1, self.h, self.d_k).transpose(1, 2))  # (batch, head, len_q, d_k)
        k = (self.linear_k(key).view(n_batch, -1, self.h, self.d_k).transpose(1, 2))  # (batch, head, len_k, d_k)
        v = (self.linear_v(value).view(n_batch, -1, self.h, self.d_k).transpose(1, 2))  # (batch, head, len_v, d_k)

        key_value_buffer = buffer[
            buffer_index : buffer_index + self.key_buffer_size + self.value_buffer_size
        ].reshape([1, self.h, self.buffersize * 2, self.d_k])
        key_buffer = torch.cat([key_value_buffer[:, :, : self.buffersize, :], k], dim=2)
        value_buffer = torch.cat([key_value_buffer[:, :, self.buffersize :, :], v], dim=2)
        buffer_out.append(
            torch.cat(
                [key_buffer[:, :, self.chunk_size :, :], value_buffer[:, :, self.chunk_size :, :]],
                dim=2,
            ).reshape(-1)
        )
        buffer_index = buffer_index + self.key_buffer_size + self.value_buffer_size

        if self.rel_enc:
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)
            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)  # (batch, head, time1, d_k)
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
            matrix_ac = torch.matmul(q_with_bias_u, key_buffer.transpose(-2, -1))
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        else:
            scores = torch.matmul(q, key_buffer.transpose(-2, -1)) / math.sqrt(
                self.d_k
            ) 

        attn = torch.softmax(scores, dim=-1)

        x = torch.matmul(attn, value_buffer)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k) 
        return self.linear_out(x), buffer, buffer_index, buffer_out 

    @torch.jit.export
    def infer_mask(self, query, key, value, mask, buffer, buffer_index, buffer_out, is_static):
        n_batch = query.size(0)
        q = (
            self.linear_q(query).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        )  # (batch, head, len_q, d_k)
        k = (
            self.linear_k(key).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        )  # (batch, head, len_k, d_k)
        v = (
            self.linear_v(value).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        )  # (batch, head, len_v, d_k)

        if is_static:
            key_buffer = k
            value_buffer = v
        else:
            key_value_buffer = buffer[
                buffer_index : buffer_index + self.key_buffer_size + self.value_buffer_size
            ].reshape([1, self.h, self.buffersize * 2, self.d_k])
            key_buffer = torch.cat([key_value_buffer[:, :, : self.buffersize, :], k], dim=2)
            value_buffer = torch.cat([key_value_buffer[:, :, self.buffersize :, :], v], dim=2)
            buffer_out.append(
                torch.cat(
                    [
                        key_buffer[:, :, self.chunk_size :, :],
                        value_buffer[:, :, self.chunk_size :, :],
                    ],
                    dim=2,
                ).reshape(-1)
            )
            buffer_index = buffer_index + self.key_buffer_size + self.value_buffer_size

        scores = torch.matmul(q, key_buffer.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )  # (batch, head, len_q, buffersize)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, self.min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        x = torch.matmul(attn, value_buffer)  # (batch, head, len_q, d_k)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x), buffer_index, buffer_out  # (batch, time1, d_model)



class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(
        self, d_model: int, dropout_rate: float, max_len: int = 1500, reverse: bool = False
    ):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, offset: int = 0):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int): position offset
        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """
        if offset + x.size(1) >= self.max_len:
            raise ValueError("length of offset and x too lang.")
        self.pe = self.pe.to(x.device)
        pos_emb = self.pe[:, offset : offset + x.size(1)]
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int, size: int):
        """For getting encoding in a streaming fashion
        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.
        Args:
            offset (int): start offset
            size (int): requried size of position encoding
        Returns:
            torch.Tensor: Corresponding encoding
        """
        if offset + size >= self.max_len:
            raise ValueError("length of offset and x too lang.")
        return self.dropout(self.pe[:, offset : offset + size])


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        chunk_size: int,
        left_chunks: int,
        max_len: int = 5000,
    ):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)
        self.chunk_size = chunk_size
        self.left_chunks = left_chunks
        self.full_chunk_size = (self.left_chunks + 1) * self.chunk_size

        self.div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        self.max_len = self.chunk_size * (max_len // self.chunk_size) - self.full_chunk_size

    @torch.jit.export
    def forward(self, x: torch.Tensor, offset: int = 0):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.pe = self.pe.to(x.device)
        x = x * self.xscale
        pos_emb = self.pe[:, offset : offset + x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)

    @torch.jit.export
    def infer(self, xs, pe_index):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        pe_index = pe_index % self.max_len
        xs = xs * self.xscale

        pe = torch.zeros(self.full_chunk_size, self.d_model)
        position = torch.arange(
            pe_index, pe_index + self.full_chunk_size, dtype=torch.float32
        ).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        pos_emb = pe.unsqueeze(0)

        pe_index = pe_index + self.chunk_size
        return xs, pos_emb, pe_index


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.
    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate
    """

    def __init__(self, idim, hidden_units, dropout_rate):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Forward funciton."""
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))

    @torch.jit.export
    def infer(self, xs, buffer, buffer_index, buffer_out):
        return self.w_2(torch.relu(self.w_1(xs))), buffer, buffer_index, buffer_out


class TransformerLayer(nn.Module):
    """Transformer layer module.

    :param int size: input dim
    :param self_attn: self attention module
    :param feed_forward: feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(
        self, size, self_attn, feed_forward, dropout_rate, normalize_before=True, concat_after=False
    ):
        """Construct an TransformerLayer object."""
        super(TransformerLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = torch.nn.LayerNorm(size)
        self.norm2 = torch.nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        else:
            self.concat_linear = nn.Identity()

    @torch.jit.unused
    def forward(self, x, mask, pos_emb):
        """Compute encoded features.

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x, x, x, mask, pos_emb)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x, x, x, mask, pos_emb))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask, pos_emb

    @torch.jit.export
    def infer(self, x, pos_emb, buffer, buffer_index, buffer_out):
        residual = x.clone()
        if self.normalize_before:
            x = self.norm1(x)
        if self.concat_after:
            x_att, buffer, buffer_index, buffer_out = self.self_attn.infer(
                x, x, x, pos_emb, buffer, buffer_index, buffer_out
            )
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x_att, buffer, buffer_index, buffer_out = self.self_attn.infer(
                x, x, x, pos_emb, buffer, buffer_index, buffer_out
            )
            x = residual + x_att
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x.clone()
        if self.normalize_before:
            x = self.norm2(x)
        x_feed, buffer, buffer_index, buffer_out = self.feed_forward.infer(
            x, buffer, buffer_index, buffer_out
        )
        x = residual + x_feed
        if not self.normalize_before:
            x = self.norm2(x)

        return x, pos_emb, buffer, buffer_index, buffer_out


class Transformer(torch.nn.Module):
    def __init__(
        self,
        args,
        input_dim=None,
        output_dim=None,
        attention_dim=None,
        attention_heads=None,
        linear_units=None,
        num_blocks=None,
        dropout_rate=None,
        positional_dropout_rate=None,
        attention_dropout_rate=None,
        input_layer=None,
        pos_enc_class=None,
        normalize_before=None,
        concat_after=None,
        positionwise_layer_type=None,
        positionwise_conv_kernel_size=None,
        chunk_size=None,
        left_chunks=None,
    ):
        """Construct an Encoder object."""
        super(Transformer, self).__init__()
        if args is None:
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.attention_dim = attention_dim
            self.attention_heads = attention_heads
            self.linear_units = linear_units
            self.num_blocks = num_blocks
            self.dropout_rate = dropout_rate
            self.positional_dropout_rate = positional_dropout_rate
            self.attention_dropout_rate = attention_dropout_rate
            self.input_layer = input_layer
            self.pos_enc_class = pos_enc_class
            self.normalize_before = normalize_before
            self.concat_after = concat_after
            self.positionwise_layer_type = positionwise_layer_type
            self.positionwise_conv_kernel_size = positionwise_conv_kernel_size
            self.chunk_size = chunk_size
            self.left_chunks = left_chunks
        else:
            self.input_dim = args.transformer_input_dim
            self.output_dim = args.transformer_output_dim
            self.attention_dim = args.transformer_attention_dim
            self.attention_heads = args.transformer_attention_heads
            self.linear_units = args.transformer_linear_units
            self.num_blocks = args.transformer_num_blocks
            self.dropout_rate = args.transformer_dropout_rate
            self.positional_dropout_rate = args.transformer_positional_dropout_rate
            self.attention_dropout_rate = args.transformer_attention_dropout_rate
            self.input_layer = args.transformer_input_layer
            self.pos_enc_class = args.transformer_pos_enc_class
            self.normalize_before = args.transformer_normalize_before
            self.concat_after = args.transformer_concat_after
            self.positionwise_layer_type = args.transformer_positionwise_layer_type
            self.positionwise_conv_kernel_size = args.transformer_positionwise_conv_kernel_size
            self.chunk_size = args.transformer_chunk_size
            self.left_chunks = args.transformer_left_chunks
            self.transformer_dynamic_chunks = args.transformer_dynamic_chunks

        if self.pos_enc_class == "abs-enc":
            pos_enc_args = (self.attention_dim, self.positional_dropout_rate)
            pos_enc_class = PositionalEncoding
        elif self.pos_enc_class == "rel-enc":
            pos_enc_args = (
                self.attention_dim,
                self.positional_dropout_rate,
                self.chunk_size,
                self.left_chunks,
            )
            pos_enc_class = RelPositionalEncoding

        if self.input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.attention_dim),
                torch.nn.LayerNorm(self.attention_dim),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.ReLU(),
            )
        elif self.input_layer == "none":
            self.embed = torch.nn.Sequential(torch.nn.Identity())
        else:
            raise ValueError("unknown input_layer: " + self.input_layer)
        self.pe = pos_enc_class(*pos_enc_args)
        self.embed_layer_num = len(self.embed)

        if self.positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (self.attention_dim, self.linear_units, self.dropout_rate)
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        def create_transformer_layer(lnum):
            return TransformerLayer(
                self.attention_dim,
                MultiHeadedAttention(
                    self.attention_heads,
                    self.attention_dim,
                    self.attention_dropout_rate,
                    self.chunk_size,
                    self.left_chunks,
                    self.pos_enc_class,
                ),
                positionwise_layer(*positionwise_layer_args),
                self.dropout_rate,
                self.normalize_before,
                self.concat_after,
            )

        self.encoders = repeat(
            self.num_blocks,
            create_transformer_layer,
        )
        if self.normalize_before:
            self.after_norm = torch.nn.LayerNorm(self.attention_dim)
            
    @staticmethod
    def add_arguments(group):
        """Add TDNN common arguments."""
        group.add_argument(
            "--transformer-input-dim", default=256, type=int, help="Input dim of Transformer."
        )
        group.add_argument(
            "--transformer-output-dim", default=4, type=int, help="Output dim of Transformer."
        )
        group.add_argument(
            "--transformer-attention-dim", default=256, type=int, help="Dimention of attention."
        )
        group.add_argument(
            "--transformer-attention-heads",
            default=4,
            type=int,
            help="The number of heads of multi head attention.",
        )
        group.add_argument(
            "--transformer-linear-units",
            default=1024,
            type=int,
            help="The number of units of position-wise feed forward.",
        )
        group.add_argument(
            "--transformer-num-blocks", default=6, type=int, help="The number of attention blocks."
        )
        group.add_argument(
            "--transformer-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate in Transformer.",
        )
        group.add_argument(
            "--transformer-attention-dropout-rate",
            default=0.0,
            type=float,
            help="Dropout rate in attention.",
        )
        group.add_argument(
            "--transformer-positional-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate after adding positional encoding.",
        )
        group.add_argument(
            "--transformer-input-layer", default="linear", type=str, help="Type of input layer"
        )
        group.add_argument("--transformer-pos-enc-class", default="abs-enc", type=str, help="")
        group.add_argument(
            "--transformer-normalize-before",
            default=True,
            type=strtobool,
            help="Whether to use layer-norm before the first block.",
        )
        group.add_argument(
            "--transformer-concat-after",
            default=False,
            type=strtobool,
            help="Whether to concat attention layer's input and output.",
        )
        group.add_argument(
            "--transformer-positionwise-layer-type",
            default="linear",
            type=str,
            help="Linear of conv1d.",
        )
        group.add_argument(
            "--transformer-positionwise-conv-kernel_size",
            default=1,
            type=int,
            help="Kernel size of positionwise conv1d layer.",
        )
        group.add_argument("--transformer-chunk_size", default=-1, type=int, help="")
        group.add_argument("--transformer-left_chunks", default=-1, type=int, help="")
        group.add_argument("--transformer-dynamic-chunks", default=True, type=strtobool, help="")
        return group

    @torch.jit.unused
    def forward(self, xs, ilens=None, masks=None):
        """Embed positions in tensor.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """

        if self.transformer_dynamic_chunks is True:  # and self.training:
            chunk_masks = add_optional_chunk_mask(ChunkParams(xs, masks, True, True, 0, 0, -1))
        else:
            chunk_masks = add_optional_chunk_mask(
                ChunkParams(xs, masks, False, False, self.chunk_size, self.chunk_size, self.left_chunks)
            ).to(xs.device)
        xs = self.embed(xs)
        xs, pos_emb = self.pe(xs)
        xs, chunk_masks, pos_emb = self.encoders(xs, chunk_masks, pos_emb)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, ilens, masks

    @torch.jit.export
    def infer(self, xs, buffer, buffer_index, buffer_out):
        xs = self.embed(xs)

        xs, pos_emb, _ = self.pe.infer(xs, 0)
        xs, pos_emb, buffer, buffer_index, buffer_out = self.encoders.infer(
            xs, pos_emb, buffer, buffer_index, buffer_out
        )

        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, buffer, buffer_index, buffer_out

    @torch.jit.export
    def infer_hidden(self, xs, buffer, buffer_index, buffer_out, hidden_out):
        xs = self.embed(xs)

        xs, pos_emb, _ = self.pe.infer(xs, 0)
        xs, pos_emb, buffer, buffer_index, buffer_out, hidden_out = self.encoders.infer_hidden(
            xs, pos_emb, buffer, buffer_index, buffer_out, hidden_out
        )

        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, buffer, buffer_index, buffer_out, hidden_out