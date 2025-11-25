# 目标检测（YoloV5s）

#### 样例介绍

通过USB接口连接Camera与开发板，从Camera获取视频，基于yolov5s模型对输入视频中的物体做实时检测，将推理结果信息使用imshow方式显示。    

样例结构如下所示：     
![输入图片说明](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/samples-pic/EdgeAndRobotics/%E5%8D%95%E7%BA%BF%E7%A8%8B%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87.png)

#### 版本配套表

   | 配套                                                         | 版本    | 环境准备指导                                                 |
   | :------------------------------------------------------------: | :-------: | :------------------------------------------------------------: |
   | 固件与驱动                                                   | 23.0.0   | [固件驱动安装准备](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit) |
   | CANN                                                         | 7.0.0 |[CANN软件包安装准备](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/softwareinst/instg/instg_0011.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)|
   | Python                                                       | 3.9.2   |[Python安装准备](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/softwareinst/instg/instg_0064.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit#ZH-CN_TOPIC_0000002017916412?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)|
   | 硬件设备型号                                                   | Orange Pi AIpro   | -                                                            |

#### 执行准备

1. 确认已安装带桌面的镜像且HDMI连接的屏幕正常显示

2. 以HwHiAiUser用户登录开发板。

3. 设置环境变量。

   ```
   # 配置程序编译依赖的头文件与库文件路径
   export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest 
   export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub
   ```

4. 安装ACLLite库。

   参考[ACLLite仓](https://gitee.com/ascend/ACLLite)安装ACLLite库。

#### 样例下载

可以使用以下两种方式下载，请选择其中一种进行源码准备。

- 命令行方式下载（**下载时间较长，但步骤简单**）。

  ```
  # 登录开发板，HwHiAiUser用户命令行中执行以下命令下载源码仓。    
  cd ${HOME}     
  git clone https://gitee.com/ascend/EdgeAndRobotics.git
  # 切换到样例目录
  cd EdgeAndRobotics/Samples/YOLOV5USBCamera
  ```

- 压缩包方式下载（**下载时间较短，但步骤稍微复杂**）。

  ```
  # 1. 仓右上角选择 【克隆/下载】 下拉框并选择 【下载ZIP】。     
  # 2. 将ZIP包上传到开发板的普通用户家目录中，【例如：${HOME}/EdgeAndRobotics-master.zip】。      
  # 3. 开发环境中，执行以下命令，解压zip包。      
  cd ${HOME} 
  chmod +x EdgeAndRobotics-master.zip
  unzip EdgeAndRobotics-master.zip
  # 4. 切换到样例目录
  cd EdgeAndRobotics-master/Samples/YOLOV5USBCamera
  ```

#### 运行样例

1. 准备测试视频。

   请从以下链接获取该样例的测试视频，放在data目录下。

   ```
   cd data
   wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/yolov5s/test.mp4 --no-check-certificate
   ```

   **注：**若需更换测试视频，则需自行准备测试视频，并将测试视频放到data目录下。

2. 获取PyTorch框架的Yolov5模型（\*.onnx），并转换为昇腾AI处理器能识别的模型（\*.om）。
   - 当设备内存**小于8G**时，可设置如下两个环境变量减少atc模型转换过程中使用的进程数，减小内存占用。
     ```
     export TE_PARALLEL_COMPILER=1
     export MAX_COMPILE_CORE_NUMBER=1
     ```
   - 为了方便下载，在这里直接给出原始模型下载及模型转换命令,可以直接拷贝执行。
     ```
     cd ../model
     wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/yolov5s/yolov5s.onnx --no-check-certificate
     wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/yolov5s/aipp.cfg --no-check-certificate
     atc --model=yolov5s.onnx --framework=5 --output=yolov5s --input_shape="images:1,3,640,640"  --soc_version=Ascend310B4  --insert_op_conf=aipp.cfg
     ```

     atc命令中各参数的解释如下，详细约束说明请参见[《ATC模型转换指南》](https://hiascend.com/document/redirect/CannCommunityAtc)。

     - --model：Yolov5网络的模型文件的路径。
     - --framework：原始框架类型。5表示ONNX。
     - --output：yolov5s.om模型文件的路径。请注意，记录保存该om模型文件的路径，后续开发应用时需要使用。
     - --input\_shape：模型输入数据的shape。
     - --soc\_version：昇腾AI处理器的版本。

3. 编译样例源码。

   执行以下命令编译样例源码。

   ```
   cd ../scripts 
   bash sample_build.sh
   ```

4. 运行样例。

   - 在HDMI连接屏幕场景，执行以下脚本运行样例。此时会以画面的形式呈现推理效果。
     ```
     bash sample_run.sh imshow
     ```
   - 在直连电脑场景，执行以下脚本运行样例。此时会以结果打屏的形式呈现推理效果。
     ```
     bash sample_run.sh stdout
     ```

#### 相关操作

- 获取更多样例，请单击[Link](https://gitee.com/ascend/samples/tree/master/inference/modelInference)。
- 获取在线视频课程，请单击[Link](https://www.hiascend.com/edu/courses?activeTab=%E5%BA%94%E7%94%A8%E5%BC%80%E5%8F%91)。
- 获取学习文档，请单击[AscendCL C&C++](https://hiascend.com/document/redirect/CannCommunityCppAclQuick)，查看最新版本的AscendCL推理应用开发指南。
- 查模型的输入输出

  可使用第三方工具Netron打开网络模型，查看模型输入或输出的数据类型、Shape，便于在分析应用开发场景时使用。
