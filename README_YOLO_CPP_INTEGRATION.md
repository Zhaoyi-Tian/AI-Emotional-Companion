# YOLO C++ 集成指南

## 概述

本指南说明如何使用C++版本的YOLO检测程序，并通过共享内存将检测结果传输到Web界面显示。

## 系统架构

```
C++ YOLO程序 → 共享内存 → Python读取器 → FastAPI服务 → Web UI
    ↓
带检测框的图像实时传输
```

## 快速开始

### 1. 编译C++程序

```bash
cd /home/HwHiAiUser/ai_助手/yolo_service/YOLOV5USBCamera

# 编译共享内存模块和主程序
./build_with_shared_memory.sh
```

### 2. 启动服务

#### 方法1：启动C++ YOLO服务（推荐）

```bash
cd /home/HwHiAiUser/ai_助手/yolo_service

# 启动C++ YOLO服务（自动启动C++检测程序）
python yolo_cpp_service.py
```

服务将在 http://localhost:5007 启动

#### 方法2：手动启动

```bash
# 终端1：启动C++检测程序
cd /home/HwHiAiUser/ai_助手/yolo_service/YOLOV5USBCamera/out
bash ../scripts/sample_run.sh stdout

# 终端2：启动共享内存读取器测试
cd /home/HwHiAiUser/ai_助手/yolo_service
python -m shared_memory_reader
```

### 3. 访问Web界面

1. 启动主Web UI：
```bash
cd /home/HwHiAiUser/ai_助手
python web_ui.py
```

2. 访问 http://localhost:8080

3. 点击"📹 YOLO检测"标签页

4. 点击"🚀 启动C++检测"按钮

5. 等待2秒让摄像头初始化

6. 观察实时视频和检测结果

## API接口说明

### HTTP端点

- `GET /` - 主页（带简单测试界面）
- `GET /health` - 健康检查
- `GET /frame` - 获取最新帧（Base64编码）
- `GET /detections` - 获取检测结果（JSON）
- `GET /video_feed` - MJPEG视频流

### WebSocket端点

- `WS /ws` - 实时双向数据流

### 测试API

```bash
# 健康检查
curl http://localhost:5007/health

# 获取最新帧
curl http://localhost:5007/frame | jq .

# 获取检测结果
curl http://localhost:5007/detections | jq .
```

## 关键文件说明

### C++端

1. **shared_memory.h/cpp**
   - 共享内存管理模块
   - 提供跨进程数据传输
   - 支持图像压缩和检测数据传输

2. **main.cpp** (已修改)
   - 添加了共享内存写入功能
   - GetResult函数输出带检测框的图像到共享内存

### Python端

1. **shared_memory_reader.py**
   - 从共享内存读取C++程序数据
   - 解码JPEG图像和检测结果
   - 提供后台读取线程

2. **yolo_cpp_service.py**
   - FastAPI服务，从共享内存读取数据
   - 提供WebSocket和HTTP接口
   - 自动管理C++程序生命周期

3. **web_ui.py** (已修改)
   - YOLO标签页支持C++版本
   - update_yolo_cpp_stream函数处理视频流
   - 定时刷新显示（200ms）

## 性能说明

- **C++检测速度**: 约10-15 FPS（取决于硬件）
- **Web显示延迟**: ~200ms（由于定时器刷新）
- **内存使用**: 共享内存约2MB
- **CPU使用**: C++程序约10-20%，Python服务约5%

## 故障排除

### 1. 共享内存问题

```bash
# 清理共享内存
sudo rm -f /dev/shm/_yolo_detection
```

### 2. 摄像头问题

```bash
# 检查摄像头设备
ls /dev/video*

# 测试摄像头
cheese  # 或使用其他摄像头软件
```

### 3. 编译错误

确保安装了必要的依赖：
```bash
sudo apt install libopencv-dev g++ cmake
```

### 4. 端口冲突

```bash
# 查看端口占用
netstat -tlnp | grep 5007

# 终止占用进程
sudo kill -9 <PID>
```

## 开发说明

### 修改检测参数

在C++ main.cpp的GetResult函数中：
- `confidenceThreshold = 0.25` (第85行)
- `NMSThreshold = 0.45` (第131行)
- `result[i].score >= 0.7` (第182行，写入共享内存的阈值)

### 添加新功能

1. **修改检测类别**：编辑C++ label.h文件
2. **自定义绘制**：修改GetResult函数中的绘制代码
3. **性能优化**：调整图像压缩质量（shared_memory.cpp第134行）

## 总结

现在您可以：
- ✅ 使用C++ YOLO程序进行实时检测
- ✅ 通过共享内存高效传输图像
- ✅ 在Web界面查看带检测框的实时视频
- ✅ 获取准确的检测置信度和位置信息
- ✅ 无需修改检测算法，只修改输出方式