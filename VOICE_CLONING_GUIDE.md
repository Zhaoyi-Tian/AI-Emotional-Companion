# CosyVoice 音色克隆使用指南

## 功能概述

本系统集成了阿里云CosyVoice的音色克隆功能，使用10~20秒音频样本即可生成高度相似且自然的定制声音，无需传统训练过程。

## 音频要求

高质量的输入音频是获得优质复刻效果的基础：

| 项目 | 要求 |
|------|------|
| 支持格式 | WAV (16bit), MP3, M4A |
| 音频时长 | 10 ~ 20 秒 |
| 文件大小 | ≤ 10 MB |
| 采样率 | ≥ 16 kHz |
| 声道 | 单声道 / 双声道 |
| 内容 | 至少包含一段5秒以上的连续、清晰、无背景音的朗读 |
| 语言 | 中、英 |

## 模型版本选择

| 版本 | 适用场景 | 申请状态 |
|------|---------|---------|
| cosyvoice-v3-plus | 追求最佳音质与表现力，预算充足 | 需申请，申请通过后将发放免费调用额度 |
| cosyvoice-v3 | 平衡效果与成本，综合性价比高 | 需申请，申请通过后将发放免费调用额度 |
| cosyvoice-v2 | 兼容旧版或低要求场景 | 直接使用 |
| cosyvoice-v1 | 兼容旧版或低要求场景 | 直接使用 |

**推荐**: 在资源与预算允许的情况下，推荐使用 `cosyvoice-v3-plus` 以获得最佳效果。

## 使用方式

### 方式一：通过Web界面（推荐）

1. **启动Web配置界面**
   ```bash
   python web_ui.py
   ```
   访问 `http://localhost:8080` (或您配置的端口)

2. **进入音色克隆页面**
   - 点击顶部导航的 "🎨 音色克隆" 标签页

3. **创建音色**
   - 选择目标模型（推荐 cosyvoice-v2 或更高版本）
   - 输入音色前缀（仅允许小写字母和数字，少于10个字符）
   - 输入音频文件的公网URL
   - 点击 "🎨 创建音色" 按钮
   - 记录返回的 Voice ID

4. **查询音色状态**
   - 在 "查询音色状态" 区域输入 Voice ID
   - 点击 "🔍 查询状态" 按钮
   - 状态说明：
     - ✅ OK: 审核通过，可以使用
     - ⏳ DEPLOYING: 审核中，请稍候
     - ❌ UNDEPLOYED: 审核未通过，无法使用

5. **使用音色**
   - 审核通过后，在 "🔊 TTS配置" 页面的 "发音人" 字段填入您的 Voice ID
   - 保存配置并测试

6. **管理音色**
   - **列出所有音色**: 查看您账户下所有已创建的音色
   - **更新音色**: 使用新的音频样本更新现有音色
   - **删除音色**: 删除不再需要的音色（操作不可逆）

### 方式二：通过API调用

所有音色克隆功能都通过TTS服务提供API接口：

#### 1. 创建音色
```bash
curl -X POST http://localhost:5003/voice/create \
  -H "Content-Type: application/json" \
  -d '{
    "target_model": "cosyvoice-v2",
    "prefix": "myvoice",
    "url": "https://your-audio-file-url.wav"
  }'
```

响应示例：
```json
{
  "success": true,
  "voice_id": "cosyvoice-v2-myvoice-xxxxxxxx",
  "request_id": "...",
  "message": "音色创建成功,请使用 /voice/query 查询状态"
}
```

#### 2. 查询音色状态
```bash
curl -X POST http://localhost:5003/voice/query \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "cosyvoice-v2-myvoice-xxxxxxxx"
  }'
```

响应示例：
```json
{
  "success": true,
  "voice_info": {
    "status": "OK",
    "gmt_create": "2024-09-13 11:29:41",
    "gmt_modified": "2024-09-13 11:29:41",
    "target_model": "cosyvoice-v2",
    "resource_link": "https://your-audio-file-url.wav"
  }
}
```

#### 3. 列出所有音色
```bash
curl -X POST http://localhost:5003/voice/list \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": "myvoice",
    "page_index": 0,
    "page_size": 10
  }'
```

#### 4. 更新音色
```bash
curl -X POST http://localhost:5003/voice/update \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "cosyvoice-v2-myvoice-xxxxxxxx",
    "url": "https://your-new-audio-file-url.wav"
  }'
```

#### 5. 删除音色
```bash
curl -X POST http://localhost:5003/voice/delete \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "cosyvoice-v2-myvoice-xxxxxxxx"
  }'
```

## 配置说明

音色克隆相关配置在 `config.yaml` 中：

```yaml
tts:
  api:
    api_key: sk-your-api-key  # 确保配置了有效的API Key
    model: cosyvoice-v2
    voice: longxiaochun_v2  # 可以使用自定义的Voice ID

  # 音色克隆配置
  voice_enrollment:
    default_model: cosyvoice-v2  # 默认使用的模型
    default_prefix: myvoice      # 默认音色前缀
    max_poll_attempts: 30        # 查询状态的最大尝试次数
    poll_interval: 10            # 查询间隔(秒)
```

## 重要提示

1. **音频URL要求**: 音频文件必须是公网可访问的URL。如果您的音频文件在本地，需要先上传到云存储服务（如阿里云OSS）并获取公网URL。

2. **模型一致性**: 声音复刻时使用的模型（target_model）必须与后续进行语音合成时使用的模型（model）保持一致。

3. **审核时间**: 音色创建后需要经过审核，通常需要几分钟到几十分钟。可以使用查询功能定期检查状态。

4. **配额限制**: 每个账户有音色创建数量限制，请合理管理音色。删除不再使用的音色可释放配额。

5. **API Key**: 确保在 `config.yaml` 中配置了有效的 DashScope API Key。

## 完整使用示例

假设您已经准备好了一段15秒的清晰朗读音频，并上传到了 `https://example.com/my-voice-sample.wav`：

1. **创建音色**（通过Web界面或API）
   - 目标模型: `cosyvoice-v2`
   - 前缀: `myperson`
   - URL: `https://example.com/my-voice-sample.wav`
   - 获得 Voice ID: `cosyvoice-v2-myperson-abc12345`

2. **等待审核**
   - 定期查询状态，直到状态变为 "OK"

3. **使用音色进行语音合成**
   - 在TTS配置中将 "发音人" 设置为: `cosyvoice-v2-myperson-abc12345`
   - 保存配置
   - 在测试页面输入文本进行合成测试

4. **在实际应用中使用**
   ```bash
   curl -X POST http://localhost:5003/synthesize \
     -H "Content-Type: application/json" \
     -d '{
       "text": "这是使用我自己声音的语音合成",
       "voice": "cosyvoice-v2-myperson-abc12345"
     }'
   ```

## 故障排除

### 问题1: 创建音色失败
- 检查音频URL是否可公网访问
- 确认音频格式、时长、大小是否符合要求
- 检查API Key是否有效
- 查看TTS服务日志: `logs/TTS.log`

### 问题2: 审核未通过（状态为UNDEPLOYED）
- 检查音频质量是否清晰
- 确认音频内容是否包含至少5秒连续朗读
- 避免背景噪音
- 尝试使用更高质量的音频样本

### 问题3: 使用音色合成失败
- 确认音色状态为 "OK"
- 确认使用的model与创建音色时的target_model一致
- 检查Voice ID是否正确

## 技术支持

- 查看日志文件: `logs/TTS.log`
- 检查服务状态: Web界面 -> 📊 服务状态
- API文档: 启动TTS服务后访问 `http://localhost:5003/docs`
