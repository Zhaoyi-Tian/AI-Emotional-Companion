# 音频缓存功能说明

## 功能概述

为了避免重复生成相同的唤醒回复和打断回复音频，系统实现了音频缓存机制。首次生成的音频会被保存到本地缓存目录，后续直接从缓存加载，显著提升响应速度并减少 API 调用。

## 缓存机制

### 缓存目录

- **位置**: `audio_cache/` (与 voice_chat.py 同级目录)
- **自动创建**: 系统启动时自动创建该目录
- **文件命名**: 使用文本内容的 MD5 哈希值前16位命名
  - 唤醒回复: `wake_reply_{hash}.pcm`
  - 打断回复: `interrupt_reply_{hash}.pcm`

### 缓存生命周期

1. **首次使用**
   - 调用 TTS API 生成音频
   - 保存到缓存目录
   - 更新内存中的缓存引用

2. **后续使用**
   - 直接从缓存文件读取
   - 跳过 TTS API 调用
   - 播放速度更快

3. **配置变更时**
   - 当 `wake_reply` 或 `interrupt_reply` 文本修改时
   - 自动清除内存中的缓存引用
   - 下次使用时重新生成并缓存新音频

## 实现细节

### 核心方法

#### `_get_cache_filename(text, cache_type)`
根据文本内容和缓存类型生成缓存文件名。

```python
import hashlib
text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
return self.cache_dir / f"{cache_type}_reply_{text_hash}.pcm"
```

#### `_load_cached_audio()`
系统启动时加载已有的缓存文件。

```python
wake_cache_file = self._get_cache_filename(self.WAKE_REPLY, 'wake')
if wake_cache_file.exists():
    self.wake_reply_audio_cache = str(wake_cache_file)
    logger.info(f"✅ 加载唤醒回复音频缓存: {self.WAKE_REPLY}")
```

#### `_save_audio_cache(text, cache_type, audio_file)`
保存新生成的音频到缓存目录。

```python
cache_file = self._get_cache_filename(text, cache_type)
shutil.copy2(audio_file, cache_file)
logger.info(f"💾 已保存{cache_type}回复音频缓存")
return str(cache_file)
```

#### `_clear_audio_cache()`
清除所有缓存文件（如需重置）。

```python
if self.cache_dir.exists():
    shutil.rmtree(self.cache_dir)
    self.cache_dir.mkdir(exist_ok=True)
```

### 使用缓存的场景

#### 1. 唤醒回复 (Wake Reply)
用户说出唤醒词后，系统播放确认音频。

**代码位置**: `voice_chat.py` 第 1316 行
```python
self.quick_reply(self.WAKE_REPLY, output_device)
```

**缓存逻辑**:
- 检查 `self.wake_reply_audio_cache` 是否存在
- 存在则直接使用缓存文件
- 不存在则调用 TTS 生成，并保存到缓存

#### 2. 打断回复 (Interrupt Reply)
用户打断 AI 说话时，系统播放确认音频。

**代码位置**: `voice_chat.py` 第 873 行
```python
self.quick_reply(self.INTERRUPT_REPLY, output_device)
```

**缓存逻辑**: 同唤醒回复

### 配置热重载集成

当通过 Web UI 或 API 修改唤醒/打断回复文本时:

```python
if old_wake_reply != self.WAKE_REPLY:
    logger.info(f"🔄 唤醒回复已更新: {old_wake_reply} → {self.WAKE_REPLY}")
    # 清除旧的缓存
    self.wake_reply_audio_cache = None
    logger.info("🗑️ 已清除唤醒回复音频缓存，将在下次使用时重新生成")
```

**代码位置**: `voice_chat.py` 第 396-400, 414-418 行

## 性能优化

### 首次启动
```
用户说唤醒词 → TTS API调用(~500ms) → 保存缓存 → 播放音频
```

### 后续使用
```
用户说唤醒词 → 读取缓存文件(~10ms) → 播放音频
```

**性能提升**: 约 **50倍** 的响应速度提升

### API 调用节省
假设每天被唤醒/打断 100 次:
- **无缓存**: 100 次 TTS API 调用
- **有缓存**: 1 次 TTS API 调用 (首次)

**API 调用节省**: **99%**

## 日志输出示例

### 首次使用 (生成并缓存)
```
💬 快速回复: 哎呦，谁在叫我呀？
💾 已保存wake回复音频缓存: 哎呦，谁在叫我呀？ -> wake_reply_a1b2c3d4e5f6g7h8.pcm
🔊 开始播放音频...
✅ 音频播放完成
```

### 后续使用 (从缓存加载)
```
💬 快速回复: 哎呦，谁在叫我呀？
🎵 使用唤醒回复音频缓存
🔊 开始播放音频...
✅ 音频播放完成
```

### 配置更新 (清除缓存)
```
🔄 唤醒回复已更新: 哎呦，谁在叫我呀？ → 你好，我在听
🗑️ 已清除唤醒回复音频缓存，将在下次使用时重新生成
```

## 维护建议

### 定期清理
虽然缓存文件很小(每个约 50-200KB)，但如果频繁修改回复文本，可能会积累较多旧缓存文件。

**手动清理**:
```bash
rm -rf audio_cache/*
```

**程序清理**:
```python
assistant._clear_audio_cache()
```

### 备份重要音频
如果对某个特定的音频效果满意，可以备份缓存文件:
```bash
cp audio_cache/wake_reply_*.pcm backup/
```

## 故障排查

### 缓存未生效
1. 检查 `audio_cache/` 目录是否存在
2. 检查日志是否显示 "✅ 加载唤醒回复音频缓存"
3. 确认文本内容完全一致(包括空格、标点)

### 音频异常
1. 删除对应的缓存文件
2. 重启语音对话服务
3. 系统将重新生成新的缓存

### 缓存文件丢失
- 系统会自动检测并重新生成
- 不影响正常功能，只是首次响应会稍慢

## 技术参数

- **缓存格式**: PCM (22050Hz, mono, 16-bit)
- **Hash 算法**: MD5 (前16位)
- **缓存位置**: `audio_cache/`
- **自动加载**: 系统启动时
- **自动清理**: 配置文本变更时

## 相关配置

在 `config.yaml` 中配置回复文本:

```yaml
voice_chat:
  wake_reply: "哎呦,谁在叫我呀?"    # 唤醒确认语音
  interrupt_reply: "好吧好吧,我不说了还不行吗~"  # 打断确认语音
```

修改这些配置后，旧缓存会自动失效，新的音频会在下次使用时生成并缓存。
