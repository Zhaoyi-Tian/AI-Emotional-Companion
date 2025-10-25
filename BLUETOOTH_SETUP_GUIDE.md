# 蓝牙音箱配置完整指南

## 概述

本指南帮助您配置蓝牙音箱作为语音对话系统的音频输出设备。

## 系统要求

- 蓝牙音箱（已配对并连接）
- PulseAudio音频服务器
- bluez蓝牙工具包

## 快速配置流程

### 1. 连接蓝牙音箱

```bash
# 启动蓝牙控制工具
bluetoothctl

# 扫描蓝牙设备
scan on

# 找到您的音箱后，配对（替换为实际MAC地址）
pair XX:XX:XX:XX:XX:XX

# 连接设备
connect XX:XX:XX:XX:XX:XX

# 信任设备（下次自动连接）
trust XX:XX:XX:XX:XX:XX

# 退出
exit
```

### 2. 验证PulseAudio识别

```bash
# 查看所有音频输出设备
pactl list sinks short

# 应该能看到类似这样的蓝牙设备：
# 1	bluez_sink.9F_52_6C_81_24_E2.a2dp_sink	module-bluez5-device.c	s16le 2ch 48000Hz	SUSPENDED
```

### 3. 设置蓝牙音箱为默认输出

**方法1：通过Web界面（推荐）**

1. 打开Web配置界面 (http://localhost:8080)
2. 进入"🎙️ 语音对话"标签页
3. 点击"🔵 检查蓝牙连接" - 确认蓝牙音箱已连接
4. 点击"🔊 设为默认输出" - 将蓝牙音箱设为系统默认音频输出
5. 验证成功提示信息

**方法2：通过命令行**

```bash
# 获取蓝牙音箱的sink名称
pactl list sinks short | grep bluez

# 设置为默认输出（替换为实际的sink名称）
pactl set-default-sink bluez_sink.9F_52_6C_81_24_E2.a2dp_sink
```

### 4. 测试音频输出

```bash
# 方法1：使用系统测试音频
paplay /usr/share/sounds/alsa/Front_Center.wav

# 方法2：通过Web界面TTS测试
# 前往"🔊 TTS配置"页面
# 输入测试文本，点击"测试合成"
# 确认声音从蓝牙音箱播放
```

### 5. 配置语音对话服务

1. Web界面 → "🎙️ 语音对话"标签页
2. 勾选"启用语音对话服务"
3. 配置USB麦克风（可选，或使用默认）
4. 设置唤醒词和VAD参数
5. 保存配置
6. 点击"重启"按钮启动服务

## 技术实现

### 音频播放优先级

系统使用以下优先级播放音频：

1. **PulseAudio (paplay)** - 首选方式
   - 对蓝牙设备支持最好
   - 自动使用系统默认输出设备
   - 兼容性强

2. **PyAudio** - 备用方式
   - 直接通过设备索引播放
   - 某些系统上蓝牙支持有限

### 工作原理

```
TTS服务 → PCM音频 → voice_chat.py
                        ↓
                    转换为WAV
                        ↓
                使用paplay播放
                        ↓
                PulseAudio路由
                        ↓
                    蓝牙音箱
```

## 故障排查

### 问题1: 检查蓝牙连接显示错误

**症状**：点击"检查蓝牙连接"显示"无法访问蓝牙服务"

**解决方案**：
```bash
# 检查蓝牙服务状态
systemctl status bluetooth

# 如果未运行，启动服务
sudo systemctl start bluetooth

# 设置开机自启
sudo systemctl enable bluetooth
```

### 问题2: PulseAudio未检测到蓝牙设备

**症状**：`pactl list sinks short` 看不到bluez设备

**解决方案**：
```bash
# 重启PulseAudio
pulseaudio -k
pulseaudio --start

# 检查蓝牙音箱是否真的已连接
bluetoothctl info [MAC地址]

# 确认"Connected: yes"字段
```

### 问题3: 设为默认输出后仍无声音

**症状**：点击"设为默认输出"成功，但播放时无声音

**解决方案**：
```bash
# 1. 检查当前默认输出设备
pactl info | grep "Default Sink"

# 2. 确认蓝牙音箱音量未静音
pactl list sinks | grep -A 15 "bluez"

# 3. 手动播放测试文件
paplay /usr/share/sounds/alsa/Front_Center.wav

# 4. 如果仍无声音，尝试重新连接蓝牙音箱
bluetoothctl
disconnect [MAC地址]
connect [MAC地址]
```

### 问题4: 语音对话无声音，但TTS测试有声音

**症状**：TTS测试页面有声音，但语音对话无声音

**解决方案**：
```bash
# 1. 检查语音对话服务日志
tail -f logs/语音对话.log

# 2. 重启语音对话服务
# 在Web界面 → 语音对话 → 点击"重启"

# 3. 确认paplay命令可用
which paplay
# 如果不存在，安装：
sudo apt install pulseaudio-utils
```

### 问题5: 音频播放卡顿或延迟

**症状**：蓝牙音箱播放有明显延迟或卡顿

**解决方案**：
```bash
# 调整PulseAudio的蓝牙编解码器
# 编辑 /etc/pulse/default.pa
sudo nano /etc/pulse/default.pa

# 找到 module-bluetooth-discover 行，修改为：
load-module module-bluetooth-discover a2dp_config="sbc_min_bp=53 sbc_max_bp=53"

# 重启PulseAudio
pulseaudio -k
pulseaudio --start
```

## 高级配置

### 自动重连蓝牙音箱

创建systemd服务自动重连：

```bash
# 创建服务文件
sudo nano /etc/systemd/system/bluetooth-autoconnect.service

# 内容：
[Unit]
Description=Bluetooth Auto Connect
After=bluetooth.service
Requires=bluetooth.service

[Service]
ExecStart=/usr/bin/bluetoothctl connect [您的MAC地址]
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target

# 启用服务
sudo systemctl enable bluetooth-autoconnect
sudo systemctl start bluetooth-autoconnect
```

### 优化音频质量

```bash
# 编辑 PulseAudio 配置
sudo nano /etc/pulse/daemon.conf

# 调整以下参数：
default-sample-rate = 48000
alternate-sample-rate = 44100
default-sample-format = s16le
default-fragments = 4
default-fragment-size-msec = 25

# 重启PulseAudio
pulseaudio -k
pulseaudio --start
```

## 验证清单

完成配置后，请验证以下项目：

- [ ] 蓝牙音箱已配对并连接
- [ ] PulseAudio可以检测到蓝牙设备 (`pactl list sinks short | grep bluez`)
- [ ] 蓝牙音箱已设为默认输出 (`pactl info | grep "Default Sink"`)
- [ ] 使用`paplay`可以正常播放音频
- [ ] TTS测试页面可以通过蓝牙音箱播放
- [ ] 语音对话服务已启动并正常运行

## 参考资料

- PulseAudio文档: https://www.freedesktop.org/wiki/Software/PulseAudio/
- BlueZ文档: http://www.bluez.org/
- 项目issue: 如遇问题请在项目仓库提交issue

---

**最后更新**: 2025-10-25
**适用版本**: AI语音助手 v1.0+
