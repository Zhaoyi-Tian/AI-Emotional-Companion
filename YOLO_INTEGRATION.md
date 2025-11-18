# YOLO实时检测集成文档

## 概述

本文档描述了如何在AI语音助手中集成YOLOv5实时目标检测功能。系统支持通过Web界面实时查看摄像头画面和检测结果。

## 功能特性

### 🎯 核心功能
- **实时检测**：基于YOLOv5的实时目标检测
- **多协议支持**：支持WebSocket、MJPEG流和REST API
- **Web界面集成**：在Gradio Web UI中添加了专门的YOLO检测标签页
- **参数可调**：支持动态调整置信度阈值和NMS阈值
- **多模式支持**：同时支持Python模拟检测和C++真实检测

### 📊 支持的检测类别
系统支持COCO数据集的80个类别：
- 人、自行车、汽车、摩托车、飞机、公交车、火车、卡车
- 交通灯、消防栓、停车标志、停车计费器、长凳
- 鸟、猫、狗、马、羊、牛、大象、熊、斑马、长颈鹿
- 背包、雨伞、手提包、领带、手提箱
- 飞盘、滑雪板、运动球、风筝、棒球棒、棒球手套
- 滑板、冲浪板、网球拍、瓶子、红酒杯、杯子
- 叉子、刀、勺子、碗、香蕉、苹果、三明治、橙子
- 西兰花、胡萝卜、热狗、披萨、甜甜圈、蛋糕
- 椅子、沙发、盆栽、床、餐桌、马桶、电视、笔记本电脑
- 鼠标、遥控器、键盘、手机、微波炉、烤箱、烤面包机
- 水槽、冰箱、书、时钟、花瓶、剪刀、泰迪熊、吹风机
- 牙刷

## 系统架构

```
摄像头采集 → YOLO检测服务 → 结果渲染
    ↓              ↓            ↓
OpenCV         FastAPI       Web UI
(640x480)      (Port 5005)   (Gradio)
    │              │            │
    └───→ WebSocket ←──────────┘
           (实时传输)
```

## 文件结构

```
ai_助手/
├── yolo_service/
│   ├── app_fastapi.py          # YOLO服务主程序
│   ├── detector.py             # Python检测器封装
│   └── YOLOV5USBCamera/        # C++ YOLO实现
│       ├── out/main            # C++可执行文件
│       └── python/src/         # Python实现
├── static/html/
│   └── yolo_stream.html        # WebSocket视频流页面
├── config.yaml                 # 配置文件（包含YOLO设置）
├── web_ui.py                   # Web UI（包含YOLO标签页）
└── test_yolo.py               # 测试脚本
```

## API端点

### REST API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/detect/start` | POST | 启动检测 |
| `/detect/stop` | POST | 停止检测 |
| `/detect/status` | GET | 获取检测状态 |
| `/detect/latest` | GET | 获取最新检测结果 |
| `/detect/update_settings` | POST | 更新检测参数 |
| `/camera/detect/stream` | GET | MJPEG视频流 |
| `/stream` | GET | HTML视频流页面 |
| `/ws/detect/stream` | WebSocket | WebSocket实时流 |

### 示例请求

#### 启动检测
```bash
curl -X POST http://localhost:5005/detect/start \
  -H "Content-Type: application/json" \
  -d '{"confidence_threshold": 0.5}'
```

#### 获取检测结果
```bash
curl http://localhost:5005/detect/latest
```

#### 更新参数
```bash
curl -X POST http://localhost:5005/detect/update_settings \
  -H "Content-Type: application/json" \
  -d '{"confidence_threshold": 0.7, "nms_threshold": 0.5}'
```

## 配置说明

在 `config.yaml` 中添加了YOLO配置节：

```yaml
yolo:
  model_path: yolo_service/models/yolov5s.om  # 模型路径
  model_width: 640                            # 模型输入宽度
  model_height: 640                           # 模型输入高度
  confidence_threshold: 0.5                   # 置信度阈值
  nms_threshold: 0.4                         # NMS阈值
  camera_index: null                         # 摄像头索引(null=自动检测)
  max_fps: 15                                # 最大FPS
  enable_streaming: true                     # 启用流式传输
```

## 使用方法

### 1. 通过Web UI使用

1. 启动完整系统：
   ```bash
   python start_all.py
   ```

2. 访问Web界面：
   http://localhost:8080

3. 点击"📹 YOLO检测"标签页

4. 点击"开始检测"启动实时检测

5. 调整参数：
   - 置信度阈值：过滤低置信度检测结果
   - NMS阈值：控制重叠检测框的抑制

### 2. 直接访问视频流页面

访问 http://localhost:5005/stream

该页面提供：
- WebSocket实时视频流
- 参数调节滑块
- 检测结果列表
- FPS和统计信息显示
- 截图保存功能

### 3. 使用MJPEG流

可以在支持MJPEG的播放器中打开：
- VLC Media Player
- OBS Studio
- 其他Web应用（通过img标签）

URL：http://localhost:5005/camera/detect/stream

## 性能优化

### 帧率控制
- 默认限制为15 FPS以降低CPU负载
- 可通过配置文件调整 `max_fps`

### 检测优化
- 支持调整置信度阈值过滤不重要的检测
- NMS阈值控制重叠检测框

### 内存管理
- 使用帧缓冲队列避免内存泄漏
- 自动清理断开的WebSocket连接

## 故障排查

### 常见问题

1. **摄像头无法打开**
   ```
   错误：Failed to open camera
   解决：检查摄像头设备 /dev/video* 是否存在
   ```

2. **检测结果显示空白**
   ```
   原因：可能是摄像头权限问题
   解决：将用户添加到video组
   ```

3. **FPS很低**
   ```
   优化方法：
   - 降低 max_fps 配置
   - 提高 confidence_threshold
   - 使用GPU加速（如果可用）
   ```

4. **WebSocket连接失败**
   ```
   检查：
   - 防火墙设置
   - 端口5005是否被占用
   ```

### 调试命令

查看YOLO服务日志：
```bash
tail -f logs/YOLO.log
```

检查服务状态：
```bash
curl http://localhost:5005/health
```

测试摄像头：
```python
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(f"Camera working: {ret}")
```

## 扩展开发

### 添加新的检测类别
编辑 `detector.py` 中的 `COCO_LABELS` 列表。

### 集成真实YOLO模型
1. 确保模型文件位于 `yolo_service/models/yolov5s.om`
2. 修改 `detector.py` 中的 `detect_objects` 方法
3. 替换模拟检测为真实模型推理

### 自定义检测逻辑
继承 `YOLODetector` 类并重写 `detect_objects` 方法：
```python
class CustomDetector(YOLODetector):
    def detect_objects(self, frame):
        # 自定义检测逻辑
        return detections
```

## 技术细节

### WebSocket协议
- 使用自定义二进制协议传输图像
- 消息格式：`b'FRAME' + 长度(4字节) + JPEG数据`
- JSON消息传输检测结果

### 图像处理
- 输入：640x480 RGB图像
- 输出：JPEG编码（质量85%）
- 检测框绘制：使用不同颜色区分类别

### 并发处理
- 使用asyncio处理多个WebSocket连接
- 每个连接独立处理，互不影响
- 自动管理连接生命周期

## 版本历史

- **v1.0.0** (2025-01-17)
  - 初始版本
  - 支持基本检测功能
  - Web UI集成
  - WebSocket实时传输

## 许可证

本模块遵循项目的整体许可证。