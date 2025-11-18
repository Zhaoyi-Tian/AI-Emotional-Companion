# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi


from .modeling_vita_transformers import Transformer


def add_encoder_args(group):
    """Add Encoder common arguments."""
    group.add_argument(
        "--encoder-layer-config",
        type=str,
        default="tdnn-dtc",
        help="Layer config of encoder. Format layername-layername-..., default(conv1d-fsmn-rnn)",
    )
    group.add_argument(
        "--encoder-input-dim",
        type=int,
        default=256,
        help="Input dim of encoder. Must equal to the input dim of the first Component (default=40)",
    )
    group.add_argument(
        "--encoder-output-dim",
        type=int,
        default=256,
        help="Output dim of encoder. Must enqual to the output dim of the last Component ! (default=256)",
    )
    # Add args of all kinds of components.
    # If you add a new component, DO NOT forget to add args to add_component_args func.
    group = Transformer.add_arguments(group)
    group = Subsampling.add_arguments(group)


def assign_args_from_dict(args, diction, prefix_key=None):
    if prefix_key is not None:
        diction = diction[prefix_key]
    for k, _ in diction.items():
        k_args = k.replace("-", "_")
        if hasattr(args, k_args):
            setattr(args, k_args, diction[k])


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


class GlobalCMVN(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, istd: torch.Tensor, norm_var: bool = True):
        """
        Args:
            mean (torch.Tensor): mean stats
            istd (torch.Tensor): inverse std, std which is 1.0 / std
        """
        super().__init__()
        if mean.shape != istd.shape:
            raise ValueError("mean.shape is not equal istd.shape")
        self.norm_var = norm_var
        # The buffer can be accessed from this module using self.mean
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (batch, max_len, feat_dim)

        Returns:
            (torch.Tensor): normalized feature
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x
    

class BaseSubsampling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.subsampling_rate = 1
        self.right_context = 0

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.right_context = 6
        self.subsampling_rate = 4

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x, x_mask[:, :, 2::2][:, :, 2::2]


class Subsampling(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.subsampling_rate = args.subsampling_rate
        self.subsampling_input_dim = args.subsampling_input_dim
        self.subsampling_output_dim = args.subsampling_output_dim
        self.subsampling_dropout_rate = args.subsampling_dropout_rate

        if self.subsampling_rate == 4:
            self.core = Conv2dSubsampling4(
                self.subsampling_input_dim,
                self.subsampling_output_dim,
                self.subsampling_dropout_rate,
            )

    @staticmethod
    def add_arguments(group):
        """Add Subsampling common arguments."""
        group.add_argument("--subsampling-rate", default=4, type=int)
        group.add_argument("--subsampling-input-dim", default=256, type=int)
        group.add_argument("--subsampling-output-dim", default=256, type=int)
        group.add_argument("--subsampling-dropout-rate", default=0.1, type=float)

        return group

    def forward(self, xs, ilens, masks):
        xs, masks = self.core(xs, masks)
        ilens = masks.squeeze(1).sum(1)
        return xs, ilens, masks


class WhaleEncoder(torch.nn.Module):
    def __init__(self, input_dim, overview_conf=None, para_conf=None, global_cmvn=None):
        super(WhaleEncoder, self).__init__()

        parser = argparse.ArgumentParser()
        add_encoder_args(parser)
        args, _ = parser.parse_known_args()

        assign_args_from_dict(args, overview_conf)

        self.config = args.encoder_layer_config.split("-")
        encoder_input_dim = args.encoder_input_dim
        encoder_output_dim = args.encoder_output_dim
        prev_output_dim = encoder_input_dim
        prev_component_name = "encoder"
        self.enc = torch.nn.ModuleList([])
        for name in self.config:
            assign_args_from_dict(args, para_conf[name])
            if len(name.split("_")) == 2:
                name = name.split("_")[0]
            elif len(name.split("_")) == 1:
                name = name
            else:
                raise ValueError(f"WRONG CONFIG! encoder {name} is not valid")

            if name == "transformer":
                self.enc.append(Transformer(args))
            elif name == "subsampling":
                self.enc.append(Subsampling(args))
            else:
                raise ValueError(f"WRONG CONFIG! encoder {name} is not valid")
            component_input_dim = getattr(args, name + "_input_dim")
            if component_input_dim != prev_output_dim:
                # This is the first layer
                raise ValueError(f"WRONG CONFIG! --{prev_component_name}-output-dim ({prev_output_dim}) \
                    does not equal to --{name}-input-dim ({component_input_dim})")
            
            prev_output_dim = getattr(args, name + "_output_dim")
            prev_component_name = name

        self.global_cmvn = global_cmvn
        if prev_output_dim != encoder_output_dim:
            raise ValueError(f"WRONG CONFIG! --encoder-output-dim ({encoder_output_dim}) \
                does not equal to --{name}-input-dim ({prev_output_dim})")

        self._output_size = encoder_output_dim

    def output_size(self) -> int:
        return self._output_size

    @torch.jit.unused
    def forward(self, xs, ilens, decoding_chunk_size=None, num_decoding_left_chunks=None):
        """Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """

        if decoding_chunk_size is not None and num_decoding_left_chunks is not None:
            for layer in self.enc:
                if hasattr(layer, "chunk_size"):
                    layer.chunk_size = decoding_chunk_size
                if hasattr(layer, "left_chunks"):
                    layer.left_chunks = num_decoding_left_chunks
                if hasattr(layer, "transformer_dynamic_chunks"):
                    layer.transformer_dynamic_chunks = False

        t = xs.size(1)
        masks = ~make_pad_mask(ilens, t).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        for module in self.enc:
            xs, ilens, masks = module(xs, ilens, masks)
        return xs, masks

    @torch.jit.export
    def infer(self, xs_pad, buffer, buffer_index, buffer_out):
        for module in self.enc:
            xs_pad, buffer, buffer_index, buffer_out = module.infer(
                xs_pad, buffer, buffer_index, buffer_out
            )
        return xs_pad, buffer, buffer_index, buffer_out

    @torch.jit.export
    def infer_hidden(self, xs_pad, buffer, buffer_index, buffer_out, hidden_out):
        for module in self.enc:
            xs_pad, buffer, buffer_index, buffer_out, hidden_out = module.infer_hidden(
                xs_pad, buffer, buffer_index, buffer_out, hidden_out
            )
        return xs_pad, buffer, buffer_index, buffer_out, hidden_out

    @torch.jit.ignore(drop=True)
    def get_extra_loss(self) -> Dict[str, torch.Tensor]:
        return None


class AudioEncoderProcessor:
    def __init__(
        self,
        dataset_conf: dict = None,
    ):
        self.dataset_conf = dataset_conf

    def process(self, wav_path):
        waveform, sample_rate = torchaudio.load(wav_path)
        if sample_rate != self.dataset_conf["resample_conf"]["resample_rate"]:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.dataset_conf["resample_conf"]["resample_rate"]
            )(waveform)

        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(
            waveform,
            num_mel_bins=self.dataset_conf["fbank_conf"]["num_mel_bins"],
            frame_length=self.dataset_conf["fbank_conf"]["frame_length"],
            frame_shift=self.dataset_conf["fbank_conf"]["frame_shift"],
            dither=self.dataset_conf["fbank_conf"]["dither"],
            energy_floor=0.0,
            sample_frequency=sample_rate,
        )
        attn_mask = torch.ones(mat.shape[0])
        attn_mask = attn_mask[2::2][2::2][0::2]

        return mat, attn_mask.shape[0]


class CNNSubsampling(torch.nn.Module):
    def __init__(
        self,
        enc_out_dim: int = 512,
        llm_embed_dim: int = 4096,
        kernel_size: int = 5,
        activation_func: str = "relu",
        norm: str = "batch",
    ):
        super().__init__()

        if enc_out_dim * 4 < llm_embed_dim:
            self.left_padding1 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
            self.conv1d1 = nn.Conv1d(enc_out_dim, 2 * enc_out_dim, kernel_size, 1, 0)
            self.bn1 = nn.BatchNorm1d(2 * enc_out_dim, eps=1e-3, momentum=0.99)
            self.relu1 = nn.ReLU()

            self.left_padding2 = nn.ConstantPad1d((0, kernel_size - 1), 0.0)
            self.conv1d2 = nn.Conv1d(2 * enc_out_dim, 4 * enc_out_dim, kernel_size, 2, 0)
            self.bn2 = nn.BatchNorm1d(4 * enc_out_dim, eps=1e-3, momentum=0.99)
            self.relu2 = nn.ReLU()

            self.project = nn.Linear(4 * enc_out_dim, llm_embed_dim)
            self.cnn_num = 2
        else:
            self.left_padding2 = nn.ConstantPad1d((0, kernel_size - 1), 0.0)
            self.conv1d2 = nn.Conv1d(enc_out_dim, 2 * enc_out_dim, kernel_size, 2, 0)
            if norm == "batch":
                self.bn2 = nn.BatchNorm1d(2 * enc_out_dim, eps=1e-3, momentum=0.99)
            elif norm == "layer":
                self.bn2 = nn.LayerNorm(2 * enc_out_dim, eps=1e-3)
            if activation_func == "gelu":
                self.relu2 = nn.GELU()
            else:
                self.relu2 = nn.ReLU()

            self.project = nn.Linear(2 * enc_out_dim, llm_embed_dim)
            self.cnn_num = 1

    def forward(self, x, mask_pad):
        """
        x: B, T, enc_out_dim
        mask: (B, T) or (B, 1, T)
        """
        x = x.transpose(1, 2)  # B, channels, T

        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        if self.cnn_num == 2:
            x = self.left_padding1(x)
            x = self.conv1d1(x)
            x = self.bn1(x)
            x = self.relu1(x)

        x = self.left_padding2(x)
        x = self.conv1d2(x)
        if isinstance(self.bn2, nn.LayerNorm):
            x = x.transpose(1, 2)
        x = self.bn2(x)
        if isinstance(self.bn2, nn.LayerNorm):
            x = x.transpose(1, 2)
        x = self.relu2(x)

        x = x.transpose(1, 2)
        x = self.project(x)

        return x, mask_pad[:, :, 0::2]


class AudioEncoder(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        llm_path: str,
        freeze_llm: bool = True,
        enc_out_dim: int = 512,
        llm_embed_dim: int = 4096,
        kernel_size: int = 3,
        ignore_id: int = -100,
        adpter_type: str = "cnn",
        add_audio_bos_eos: bool = False,
        task_num: int = 10,
        task_before_audio: bool = False,
        task_type: str = "prompt",
        freeze_encoder: bool = False,
        freeze_adpter: bool = False,
        audio_prompt_finetune: bool = False,
        audio_prompt_num: int = 25,
        activation_func: str = "relu",
        norm: str = "batch",
        chat_template=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.enc_out_dim = enc_out_dim
        self.llm_embed_dim = llm_embed_dim
        self.ignore_id = ignore_id
        self.add_audio_bos_eos = add_audio_bos_eos
        self.task_before_audio = task_before_audio
        self.task_type = task_type
        self.freeze_encoder = freeze_encoder
        self.freeze_adpter = freeze_adpter
        self.audio_prompt_finetune = audio_prompt_finetune
        self.audio_prompt_num = audio_prompt_num

        if adpter_type == "subsampling":
            self.adpter = CNNSubsampling(
                enc_out_dim, llm_embed_dim, kernel_size, activation_func, norm
            )

        if self.freeze_encoder:
            self.encoder.eval()
            for (_, param) in self.encoder.named_parameters():
                param.requires_grad = False
        if self.freeze_adpter:
            self.adpter.eval()
            for (_, param) in self.adpter.named_parameters():
                param.requires_grad = False

        if self.audio_prompt_finetune:
            self.prompt_embeddings = nn.Embedding(audio_prompt_num, llm_embed_dim)
            self.prompt_ids = torch.tensor([i for i in range(audio_prompt_num)]).long()

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:

        speech = speech.to(next(self.parameters()).dtype)

        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        inputs_embeds, encoder_mask = self.adpter(encoder_out, encoder_mask)  # B, T, D
        attention_mask = encoder_mask.squeeze(1)  # B, T

        # audio bos/eos
        if self.add_audio_bos_eos:
            inputs_embeds, attention_mask, target = self._add_bos_eos(
                "audio", "/audio", inputs_embeds, attention_mask,
            )

        b, _, _ = inputs_embeds.shape
        if self.audio_prompt_finetune:
            prompt_ids = self.prompt_ids.repeat(b, 1).to(inputs_embeds.device)
            prompt_embeds = self.prompt_embeddings(
                                prompt_ids.to(inputs_embeds.device)) # B, 5, D
            inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), 1) # B, (T+5), D

        outputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }

        return outputs

    def _add_bos_eos(self, bos, eos, inputs_embeds, attention_mask, target=None):
        b_length = len(inputs_embeds)
        bos_embed = self.task_embeddings(
            torch.full([b_length, 1], self.task_ids[bos]).to(inputs_embeds.device)
        )  # B, 1, D
        eos_embed = self.task_embeddings(
            torch.full([b_length, 1], self.task_ids[eos]).to(inputs_embeds.device)
        )  # B, 1, D
        bos_eos_target = torch.full([b_length, 2], self.ignore_id).to(inputs_embeds.device)  # B, 2
        bos_eos_mask = torch.full([b_length, 1], True).to(inputs_embeds.device)  # B, 1

        inputs_embeds = torch.cat((bos_embed, inputs_embeds), 1)  # B, (1+T), D
        inputs_embeds = torch.cat((inputs_embeds, eos_embed), 1)  # B, (1+T+1), D
        attention_mask = torch.cat((bos_eos_mask, attention_mask), 1)  # B, (1+T)
        attention_mask = torch.cat((attention_mask, bos_eos_mask), 1)  # B, (1+T+1)
        if target is not None:
            target = torch.cat((target, bos_eos_target), 1)  # B, (T+2), D

        return inputs_embeds, attention_mask, target