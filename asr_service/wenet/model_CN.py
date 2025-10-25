import torchaudio
import torchaudio.compliance.kaldi as kaldi
from ais_bench.infer.interface import InferSession
import numpy as np
import logging

logger = logging.getLogger("ASR_Service")


class WeNetASRCN:
    def __init__(self, model_path, vocab_path):
        """初始化模型，加载词表"""
        self.vocabulary = load_vocab(vocab_path)
        self.model = InferSession(0, model_path)
        # 获取模型输入特征的最大长度
        self.max_len = self.model.get_inputs()[0].shape[1]
        # 计算安全的音频分段长度（秒）
        # 留出余量避免边界问题，使用80%的最大长度
        self.safe_chunk_duration = (self.max_len * 0.01) * 0.8  # ~7.7秒

    def transcribe(self, wav_file):
        """执行模型推理，将录音文件转为文本。支持长音频自动分段。"""
        # 加载音频获取时长
        waveform, sample_rate = torchaudio.load(wav_file)
        audio_duration = waveform.shape[1] / sample_rate

        # 如果音频短于安全长度，使用原方法
        if audio_duration <= self.safe_chunk_duration:
            feats_pad, feats_lengths = self.preprocess(wav_file)
            output = self.model.infer([feats_pad, feats_lengths])
            txt = self.post_process(output)
            return txt
        else:
            # 使用分段识别
            logger.info(f"📊 音频时长 {audio_duration:.2f}秒，启用分段识别...")
            return self.transcribe_long_audio(wav_file)

    def transcribe_long_audio(self, wav_file):
        """
        长音频分段识别
        将长音频切分为多个片段，分别识别后拼接
        """
        waveform, sample_rate = torchaudio.load(wav_file)
        waveform, sample_rate = resample(waveform, sample_rate, resample_rate=16000)

        total_samples = waveform.shape[1]
        total_duration = total_samples / sample_rate

        # 分段参数
        chunk_duration = self.safe_chunk_duration  # 每段时长（秒）
        overlap_duration = 0.25  # 重叠时长（秒），避免截断词语
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
        step_samples = chunk_samples - overlap_samples

        # 计算分段数量
        num_chunks = int(np.ceil((total_samples - overlap_samples) / step_samples))
        logger.info(f"🔪 将音频切分为 {num_chunks} 段进行识别...")

        results = []
        for i in range(num_chunks):
            start_sample = i * step_samples
            end_sample = min(start_sample + chunk_samples, total_samples)

            # 提取音频片段
            chunk_waveform = waveform[:, start_sample:end_sample]

            # 计算该片段的特征
            feature = compute_fbank(chunk_waveform, sample_rate)

            # 预处理和推理
            feats_pad = pad_sequence(feature,
                                    batch_first=True,
                                    padding_value=0,
                                    max_len=self.max_len)
            feats_pad = feats_pad.numpy().astype(np.float32)
            feats_lengths = np.array([feature.shape[0]]).astype(np.int32)

            output = self.model.infer([feats_pad, feats_lengths])
            text = self.post_process(output)

            chunk_start_time = start_sample / sample_rate
            chunk_end_time = end_sample / sample_rate
            logger.info(f"  ✓ 片段 {i+1}/{num_chunks} ({chunk_start_time:.1f}s-{chunk_end_time:.1f}s): {text[:30]}...")

            results.append(text)

        # 拼接结果
        final_text = self.merge_segments(results)
        logger.info(f"✅ 分段识别完成，总文本长度: {len(final_text)} 字符")

        return final_text

    def merge_segments(self, segments):
        """
        智能拼接分段识别结果
        由于中文没有空格，简单连接即可
        """
        return ''.join(segments)

    def preprocess(self, wav_file):
        """数据预处理"""
        waveform, sample_rate = torchaudio.load(wav_file)
        # 音频重采样，采样率16000
        waveform, sample_rate = resample(waveform, sample_rate, resample_rate=16000)
        # 计算fbank特征
        feature = compute_fbank(waveform, sample_rate)
        feats_lengths = np.array([feature.shape[0]]).astype(np.int32)

        # 检查音频长度并打印警告
        feat_len = feature.shape[0]
        max_duration_sec = self.max_len * 0.01  # 每帧10ms
        actual_duration_sec = feat_len * 0.01
        if feat_len > self.max_len:
            import logging
            logger = logging.getLogger("ASR_Service")
            logger.warning(f"⚠️ 音频时长({actual_duration_sec:.2f}秒)超过模型最大限制({max_duration_sec:.2f}秒)，将被截断!")
            logger.warning(f"   建议使用时长 ≤ {max_duration_sec:.1f}秒 的音频，或等待分段识别功能")

        # 对输入特征进行padding，使符合模型输入尺寸
        feats_pad = pad_sequence(feature,
                                 batch_first=True,
                                 padding_value=0,
                                 max_len=self.max_len)
        feats_pad = feats_pad.numpy().astype(np.float32)
        return feats_pad, feats_lengths

    def post_process(self, output):
        """对模型推理结果进行后处理，根据贪心策略选择概率最大的token，去除重复字符和空白字符，得到最终文本。"""
        encoder_out_lens, probs_idx = output[1], output[4]
        token_idx_list = probs_idx[0, :, 0][:encoder_out_lens[0]]
        token_idx_list = remove_duplicates_and_blank(token_idx_list)
        text = ''.join(self.vocabulary[token_idx_list])
        return text


def remove_duplicates_and_blank(token_idx_list):
    """去除重复字符和空白字符"""
    res = []
    cur = 0
    BLANK_ID = 0
    while cur < len(token_idx_list):
        if token_idx_list[cur] != BLANK_ID:
            res.append(token_idx_list[cur])
        prev = cur
        while cur < len(token_idx_list) and token_idx_list[cur] == token_idx_list[prev]:
            cur += 1
    return res


def pad_sequence(seq_feature, batch_first=True, padding_value=0, max_len=966):
    """对输入特征进行padding，使符合模型输入尺寸"""
    feature_shape = seq_feature.shape
    feat_len = feature_shape[0]
    if feat_len > max_len:
        # 如果输入特征长度大于模型输入尺寸，则截断
        seq_feature = seq_feature[:max_len].unsqueeze(0)
        return seq_feature

    batch_size = 1
    trailing_dims = feature_shape[1:]
    if batch_first:
        out_dims = (batch_size, max_len) + trailing_dims
    else:
        out_dims = (max_len, batch_size) + trailing_dims

    out_tensor = seq_feature.data.new(*out_dims).fill_(padding_value)
    if batch_first:
        out_tensor[0, :feat_len, ...] = seq_feature
    else:
        out_tensor[:feat_len, 0, ...] = seq_feature
    return out_tensor


def resample(waveform, sample_rate, resample_rate=16000):
    """音频重采样"""
    waveform = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=resample_rate)(waveform)
    return waveform, resample_rate


def compute_fbank(waveform,
                  sample_rate,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """提取filter bank音频特征"""
    AMPLIFY_FACTOR = 1 << 15
    waveform = waveform * AMPLIFY_FACTOR
    mat = kaldi.fbank(waveform,
                      num_mel_bins=num_mel_bins,
                      frame_length=frame_length,
                      frame_shift=frame_shift,
                      dither=dither,
                      energy_floor=0.0,
                      sample_frequency=sample_rate)
    return mat


def load_vocab(txt_path):
    """加载词表"""
    vocabulary = []
    LEN_OF_VALID_FORMAT = 2
    with open(txt_path, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            # 词表格式：token id
            if len(arr) != LEN_OF_VALID_FORMAT:
                raise ValueError(f"Invalid line: {line}. Expect format: token id")
            vocabulary.append(arr[0])
    return np.array(vocabulary)
