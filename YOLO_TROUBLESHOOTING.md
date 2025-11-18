# YOLO集成问题解决方案

## 问题总结

您遇到的两个主要问题：

1. **DeprecationWarning**: FastAPI的 `on_event` 已被弃用，建议使用 `lifespan` 事件处理器
2. **端口占用错误**: 端口5005被占用，导致服务无法启动

## 已实施的解决方案

### 1. 更新FastAPI事件处理器

已将 `@app.on_event("startup")` 和 `@app.on_event("shutdown")` 替换为新的 `lifespan` 模式：

```python
# 新的写法
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting YOLO Detection Service...")
    # ... 初始化代码 ...

    yield

    # Shutdown
    logger.info("Shutting down YOLO Detection Service...")
    # ... 清理代码 ...

app = FastAPI(lifespan=lifespan)
```

### 2. 端口管理

创建了清理脚本确保端口正确释放：

```bash
# 清理端口占用
python clean_ports.py

# 或手动清理
lsof -ti:5005 | xargs -r kill -9
```

## 当前系统状态

✅ **YOLO检测服务**: 正常运行在端口5005
✅ **YOLO检测功能**: 已启动，摄像头0可用
⚠️ **Web UI**: 运行中但健康检查路径可能不同

## 访问地址

### 主要入口
- **Web UI**: http://localhost:8080
  - 点击"📹 YOLO检测"标签页
  - 点击"开始检测"启动实时检测

### 直接访问YOLO功能
- **API文档**: http://localhost:5005
- **视频流页面**: http://localhost:5005/stream
- **健康检查**: http://localhost:5005/health
- **MJPEG流**: http://localhost:5005/camera/detect/stream

## 常用命令

### 服务管理
```bash
# 启动YOLO服务
python yolo_service/app_fastapi.py

# 启动Web UI
python web_ui.py

# 检查服务状态
python check_yolo_status.py

# 测试YOLO功能
python test_yolo.py
```

### 检测控制
```bash
# 启动检测
curl -X POST http://localhost:5005/detect/start \
  -H "Content-Type: application/json" \
  -d '{"confidence_threshold": 0.5}'

# 停止检测
curl -X POST http://localhost:5005/detect/stop

# 查看检测状态
curl http://localhost:5005/detect/status

# 获取最新检测结果
curl http://localhost:5005/detect/latest
```

## 日志查看

```bash
# YOLO服务日志
tail -f logs/YOLO.log

# Web UI日志
tail -f logs/Web配置界面.log

# 查看所有日志
tail -f logs/*.log
```

## 故障排查

### 端口被占用
```bash
# 查找占用端口的进程
lsof -i :5005

# 强制结束进程
kill -9 <PID>
```

### 摄像头问题
1. 检查摄像头设备：
   ```bash
   ls -la /dev/video*
   ```

2. 测试摄像头：
   ```python
   import cv2
   cap = cv2.VideoCapture(0)
   ret, frame = cap.read()
   print(f"摄像头工作: {ret}")
   ```

### 服务无法启动
1. 检查依赖：
   ```bash
   pip install fastapi uvicorn python-multipart
   ```

2. 检查Python路径：
   ```bash
   which python
   python --version
   ```

## 性能优化建议

1. **FPS限制**: 默认限制为15 FPS以降低CPU负载
2. **置信度阈值**: 调整到0.5-0.7之间过滤不重要的检测
3. **分辨率**: 使用640x480平衡质量和性能

## 下一步

1. 测试完整功能：
   - 访问 http://localhost:8080
   - 进入YOLO检测标签页
   - 启动检测并调整参数

2. 如需集成真实YOLO模型：
   - 将 `.om` 模型文件放入 `yolo_service/models/`
   - 修改 `detector.py` 中的检测逻辑

3. 生产环境部署：
   - 限制CORS允许的域名
   - 添加认证机制
   - 使用HTTPS

## 备注

- 系统当前使用模拟检测模式（便于测试）
- 需要真实模型时，替换检测逻辑即可
- 所有代码已按照FastAPI最新规范更新