# YOLO C++检测使用指南

## ✅ 已移除模拟检测

我已经：
1. 删除了所有随机生成的模拟检测代码
2. 移除了`detector.py`中的模拟实现
3. 创建了纯C++版本的YOLO服务

## 🔧 当前实现

### 纯C++检测模式
- **C++程序**: `/yolo_service/YOLOV5USBCamera/out/main`
- **启动脚本**: `/yolo_service/YOLOV5USBCamera/scripts/sample_run.sh stdout`
- **检测器**: `/yolo_service/cpp_detector.py`
- **API服务**: `/yolo_service/app_fastapi_cpp_only.py`

### 工作流程
1. C++程序进行真实的YOLO检测
2. 输出格式: `Detect Result: [label:score x1,y1,x2,y2 ...]`
3. Python解析器读取输出并提取检测结果
4. API返回结构化的检测结果

## 🚀 使用方法

### 1. 启动C++检测
```bash
# 启动服务
python yolo_service/app_fastapi_cpp_only.py

# 启动检测
curl -X POST http://localhost:5005/detect/start
```

### 2. 查看检测结果
C++程序会在独立窗口中显示检测结果画面，包括：
- 实时视频流
- 检测框标注
- 置信度显示

### 3. API获取检测数据
```bash
# 获取检测状态
curl http://localhost:5005/detect/status

# 返回格式
{
  "success": true,
  "status": {
    "is_running": true,
    "pid": 39270,
    "fps": 0,
    "last_detection_count": 2,
    "last_output": "Detect Result: [person:0.85 100,100,200,200 ..."
  },
  "detections": [
    {
      "label": "person",
      "confidence": 0.85,
      "bbox": [100, 100, 200, 200]
    }
  ]
}
```

## 📝 重要说明

### 1. 关于视频流
- C++版本会在**独立窗口**显示检测画面
- 不通过Web传输视频（避免性能损失）
- Web UI只显示检测结果的文本数据

### 2. 检测输出格式
C++程序输出示例：
```
Detect Result: [person:0.85 100,100,200,200  car:0.75 300,150,450,300]
```

### 3. 环境变量
启动前需要设置：
```bash
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub
```

### 4. 模型要求
确保模型文件存在：
- `/yolo_service/models/yolov5s.om`

## 🎯 Web界面集成

虽然C++版本在独立窗口显示，但Web UI仍然可以：
- 启动/停止检测
- 查看检测状态
- 获取检测对象列表
- 监控FPS

访问：http://localhost:8080 → 点击"📹 YOLO检测"

## ⚡ 性能对比

### 之前（模拟）
- ❌ 虚假检测结果
- ❌ 随机生成的对象
- ❌ 无实际意义

### 现在（C++真实检测）
- ✅ 真实YOLOv5推理
- ✅ 准确的目标检测
- ✅ 使用Ascend NPU加速

## 🛠 故障排查

### 1. C++程序无法启动
```bash
# 检查可执行文件
ls -la /yolo_service/YOLOV5USBCamera/out/main

# 手动测试
cd /yolo_service/YOLOV5USBCamera
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
bash scripts/sample_run.sh stdout
```

### 2. 检测窗口不显示
- 确保有X11显示环境
- 尝试设置 `export DISPLAY=:0`

### 3. 无检测结果
- 检查摄像头设备
- 确认模型文件存在
- 查看日志输出

## 📊 总结

现在系统**只使用真实的C++ YOLO检测**：
- ✅ 无任何模拟或随机生成
- ✅ 真实的目标检测功能
- ✅ 使用NPU硬件加速
- ✅ C++独立窗口显示
- ✅ API提供检测数据

请使用C++程序进行真实的YOLO检测！