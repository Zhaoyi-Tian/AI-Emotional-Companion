#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <queue>
#include <unordered_map>
#include <cmath>
#include <dirent.h>
#include <string.h>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "acllite_dvpp_lite/ImageProc.h"
#include "acllite_om_execute/ModelProc.h"
#include "acllite_media/CameraRead.h"
#include "acllite_dvpp_lite/VideoRead.h"
#include "acllite_common/Queue.h"
#include "label.h" // 确保这里定义了 emotion 的 label 数组

using namespace std;
using namespace acllite;
using namespace cv;

aclrtContext context = nullptr;

// --- 全局参数配置 ---
uint32_t cvWidth = 1920;
uint32_t cvHeight = 1080;
// 模型输入尺寸
uint32_t modelWidth = 640;
uint32_t modelHeight = 640;

// *** 关键修改点 1: 参数适配新模型 ***
const size_t classNum = 7;        // 情感识别是7类
const size_t outputBoxNum = 8400; // Ultralytics 导出模型的锚点数
const float scoreThreshold = 0.45f; // 置信度阈值
const float nmsThreshold = 0.45f;   // NMS 阈值

bool exitFlag = false;

struct MsgData {
    std::shared_ptr<uint8_t> data = nullptr;
    uint32_t size = 0;
    bool videoEnd = false;
    cv::Mat srcImg;
};

struct MsgOut {
    cv::Mat srcImg;
    bool videoEnd = false;
    vector<InferenceOutput> inferOutputs;
};

struct BoundBox {
    float x;
    float y;
    float width;
    float height;
    float score;
    size_t classIndex;
};

Queue<MsgData> msgDataQueue(32);
Queue<MsgOut> msgOutQueue(32);
std::string displayMode = "";

// Base64 编码函数 (保持不变)
string base64_encode(const string& input) {
    const string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    string encoded;
    int i = 0, j = 0;
    unsigned char char_array_3[3], char_array_4[4];
    size_t in_len = input.size();
    while (in_len--) {
        char_array_3[i++] = input[j++];
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            for(i = 0; i < 4; i++) encoded += base64_chars[char_array_4[i]];
            i = 0;
        }
    }
    if (i) {
        for(j = i; j < 3; j++) char_array_3[j] = '\0';
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;
        for (j = 0; j < i + 1; j++) encoded += base64_chars[char_array_4[j]];
        while((i++ < 3)) encoded += '=';
    }
    return encoded;
}

bool Initparam(int argc, char *argv[]) {
    DIR *dir;
    if ((dir = opendir("./out")) == NULL) system("mkdir ./out");
    if(argc != 2) {
        LOG_PRINT("[ERROR] please choose output display mode: [./main imshow] [./main stdout]");
        return false;
    }
    displayMode = argv[1];
    return true;
}

bool sortScore(BoundBox box1, BoundBox box2) {
    return box1.score > box2.score;
}
// void DumpFirstNAnchors(const float* outputData, int num_channels, int num_anchors, int dump_n) {
//     printf("==== Dump first %d anchors, %d channels ====\n", dump_n, num_channels);
//     for (int i = 0; i < dump_n; ++i) {
//         printf("Anchor %d: [", i);
//         for (int c = 0; c < num_channels; ++c) {
//             float v = outputData[c * num_anchors + i]; // [channel, anchor]
//             printf("%f", v);
//             if (c < num_channels - 1)
//                 printf(", ");
//         }
//         printf("]\n");
//     }
// }

void GetResult(std::vector<InferenceOutput>& inferOutputs,
    cv::Mat& srcImage, uint32_t modelWidth, uint32_t modelHeight)
{
    if (inferOutputs.empty()) return;

    // 真实输出: [1, 11, 8400], 即 [batch, channel, anchor]
    float* outputData = static_cast<float*>(inferOutputs[0].data.get());

    // // 测试输出前10个anchor所有通道值
    // DumpFirstNAnchors(outputData, 11, 8400, 10);  // 只定义一次

    int srcWidth = srcImage.cols;
    int srcHeight = srcImage.rows;
    vector<BoundBox> boxes;

    const int num_anchors = outputBoxNum; // 8400
    const int num_classes = classNum;     // 7

    for (int i = 0; i < num_anchors; ++i) {
        float cx = outputData[0 * num_anchors + i];
        float cy = outputData[1 * num_anchors + i];
        float w  = outputData[2 * num_anchors + i];
        float h  = outputData[3 * num_anchors + i];

        float maxScore = 0.0f;
        int maxClassId = -1;
        for (int c = 0; c < num_classes; ++c) {
            float score = outputData[(4 + c) * num_anchors + i];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = c;
            }
        }

        // maxScore很低，理论上不会输出大数
        if (maxScore > scoreThreshold && w > 1.0f && h > 1.0f) {
            BoundBox box;
            box.x = cx;
            box.y = cy;
            box.width = w;
            box.height = h;
            box.score = maxScore;        // 只用maxScore！
            box.classIndex = maxClassId;
            boxes.push_back(box);
        }
    }

    // NMS部分不变
    std::sort(boxes.begin(), boxes.end(), sortScore);
    vector<BoundBox> result;
    while (!boxes.empty()) {
        result.push_back(boxes[0]);
        size_t index = 1;
        while (index < boxes.size()) {
            float b1_x1 = boxes[0].x - boxes[0].width / 2;
            float b1_y1 = boxes[0].y - boxes[0].height / 2;
            float b1_x2 = boxes[0].x + boxes[0].width / 2;
            float b1_y2 = boxes[0].y + boxes[0].height / 2;

            float b2_x1 = boxes[index].x - boxes[index].width / 2;
            float b2_y1 = boxes[index].y - boxes[index].height / 2;
            float b2_x2 = boxes[index].x + boxes[index].width / 2;
            float b2_y2 = boxes[index].y + boxes[index].height / 2;
            float interX1 = max(b1_x1, b2_x1);
            float interY1 = max(b1_y1, b2_y1);
            float interX2 = min(b1_x2, b2_x2);
            float interY2 = min(b1_y2, b2_y2);
            float interW = max(0.0f, interX2 - interX1);
            float interH = max(0.0f, interY2 - interY1);
            float areaInter = interW * interH;
            float area1 = boxes[0].width * boxes[0].height;
            float area2 = boxes[index].width * boxes[index].height;
            float iou = areaInter / (area1 + area2 - areaInter);

            if (iou > nmsThreshold)
                boxes.erase(boxes.begin() + index);
            else
                ++index;
        }
        boxes.erase(boxes.begin());
    }

    // 输出格式与原实现一致
    ostringstream jsonOutput;
    jsonOutput << "{\"detections\":[";

    bool firstDetection = true;
    const vector<cv::Scalar> colors {
        cv::Scalar(237,149,100), cv::Scalar(0,215,255),
        cv::Scalar(50,205,50), cv::Scalar(139,85,26)
    };

    for (size_t i = 0; i < result.size(); ++i) {
        float cx = result[i].x;
        float cy = result[i].y;
        float w = result[i].width;
        float h = result[i].height;
        float x1 = cx - w/2;
        float y1 = cy - h/2;
        float x2 = cx + w/2;
        float y2 = cy + h/2;
        x1 = max(0.0f, min(x1, (float)srcWidth));
        y1 = max(0.0f, min(y1, (float)srcHeight));
        x2 = max(0.0f, min(x2, (float)srcWidth));
        y2 = max(0.0f, min(y2, (float)srcHeight));

        if (displayMode == "imshow") {
            cv::rectangle(srcImage, cv::Point(x1, y1), cv::Point(x2, y2), colors[i % colors.size()], 2);
            string className = label[result[i].classIndex];
            string markString = to_string(result[i].score).substr(0,4) + ":" + className;
            cv::putText(srcImage, markString, cv::Point(x1, y1-5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,255));
        }

        // Stdout JSON
        if (displayMode == "stdout") {
            if (!firstDetection) jsonOutput << ",";
            firstDetection = false;
            jsonOutput << "{"
                        << "\"class\":\"" << label[result[i].classIndex] << "\","
                        << "\"class_id\":" << result[i].classIndex << ","
                        << "\"confidence\":" << fixed << setprecision(3) << result[i].score << ","
                        << "\"bbox\":{"
                        << "\"x\":" << x1 << ","
                        << "\"y\":" << y1 << ","
                        << "\"width\":" << (x2-x1) << ","
                        << "\"height\":" << (y2-y1)
                        << "}}";
        }
    }
    jsonOutput << "],\"timestamp\":" << time(NULL);

    if (displayMode == "stdout") {
        vector<uchar> buffer;
        vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
        cv::imencode(".jpg", srcImage, buffer, params);
        string img_str(buffer.begin(), buffer.end());
        jsonOutput << ",\"frame\":\"" << base64_encode(img_str) << "\"";
        jsonOutput << "}";
        cout << jsonOutput.str() << endl;
    } else if (displayMode == "imshow") {
        cv::imshow("usb-show-demo", srcImage);
        cv::waitKey(1);
    }
}
// *** 关键点 3: 使用 AIPP 后，GetInput 只需要传递 YUV 数据 ***
void* GetInput(void* arg) {
    bool ret = SetCurContext(context);
    CHECK_RET(ret, LOG_PRINT("[ERROR] set cur context for pthread  %ld failed.", pthread_self()); return NULL);
    int32_t deviceId = *(int32_t *)arg;
    string devPath = "/dev/video0";
    CameraRead cap(devPath, deviceId);
    CHECK_RET(cap.IsOpened(), LOG_PRINT("[ERROR] open %s failed.", devPath.c_str()); return NULL);
    ImageProc imageProcess;
    ImageData frame;
    ImageSize modelSize(modelWidth, modelHeight);
    int isHost = GetRunMode(); // 获取运行模式，用于判断是否需要拷贝
    LOG_PRINT("[INFO] start to decode...");
    
    while(1) {
        ret = cap.Read(frame);
        if (ret) {
            ImageData dst;
            // 1. DVPP Resize (输出 YUV 数据给模型)
            imageProcess.Resize(frame, dst, modelSize, RESIZE_PROPORTIONAL_UPPER_LEFT);
            
            MsgData msgData;
            msgData.data = dst.data; // 推理数据，无需拷贝
            msgData.size = dst.size;
            msgData.videoEnd = false;
            
            // ******************* 关键修改区域 (用于显示) *******************
            // 注意：摄像头输出通常是 YUYV (CV_8UC2) 或 NV21/NV12 (CV_8UC1 * 1.5 height)
            // 根据你的旧代码，这里尝试使用 YUYV 的方式。如果颜色不对，可能需要调整。
            // 确保 yuyvImg 的尺寸和类型正确匹配 CameraRead 出来的 frame。
            
            // 假设 CameraRead 出来的 frame.size 是 YUYV 的大小 (Width * Height * 2)
            cv::Mat yuyvImg(frame.height, frame.width, CV_8UC2); 
            
            if (isHost) {
                // Device 模式，需要拷贝到 Host
                void* hostDataBuffer = CopyDataToHost(frame.data.get(), frame.size);
                if (hostDataBuffer == nullptr) {
                    LOG_PRINT("[ERROR] CopyDataToHost failed.");
                    continue; 
                }
                memcpy(yuyvImg.data, (unsigned char*)hostDataBuffer, frame.size);
                FreeHostMem(hostDataBuffer);
            } else {
                // Host 模式（或零拷贝环境），直接 memcpy
                memcpy(yuyvImg.data, (unsigned char*)frame.data.get(), frame.size);
            }
            
            // 颜色转换用于显示
            cv::cvtColor(yuyvImg, msgData.srcImg, cv::COLOR_YUV2BGR_YUYV);
            // *******************************************************************
            
            while (1) {
                if (msgDataQueue.Push(msgData)) {
                    break;
                }
                usleep(100);
            }
        } else {
            LOG_PRINT("[INFO] frame read end.");
            break;
        }
    }
    cap.Release();
    MsgData msgData;
    msgData.videoEnd = true;
    while (1) {
        if (msgDataQueue.Push(msgData)) {
            break;
        }
        usleep(100);
    }
    LOG_PRINT("[INFO] preprocess add end msgData. tid : %ld", pthread_self());
    return NULL;
}

void* ModelExecute(void* arg) {
    bool ret = SetCurContext(context);
    CHECK_RET(ret, LOG_PRINT("[ERROR] set cur context failed."); return NULL);
    ModelProc modelProcess;
    // 确保这里的模型是加了 AIPP 的版本
    string modelPath = "../model/yolov5s_emotion.om";
    ret = modelProcess.Load(modelPath);
    CHECK_RET(ret, LOG_PRINT("[ERROR] load model %s failed.", modelPath.c_str()); return NULL);
    
    while(1) {
        if(!msgDataQueue.Empty()) {
            MsgData msgData = msgDataQueue.Pop();
            if (msgData.videoEnd) break;
            
            // 使用 AIPP 时，CreateInput 接收 YUV 数据大小是合法的
            ret = modelProcess.CreateInput(static_cast<void*>(msgData.data.get()), msgData.size);
            if (!ret) {
                LOG_PRINT("[ERROR] Create model input failed. Check AIPP config!");
                continue;
            }
            
            MsgOut msgOut;
            msgOut.srcImg = msgData.srcImg;
            msgOut.videoEnd = false;
            modelProcess.Execute(msgOut.inferOutputs);
            
            while (1) {
                if (msgOutQueue.Push(msgOut)) break;
                usleep(100);
            }
        }
        usleep(100);
    }
    modelProcess.DestroyResource();
    MsgOut msgOut;
    msgOut.videoEnd = true;
    msgOutQueue.Push(msgOut);
    LOG_PRINT("[INFO] infer msg end.");
    return NULL;
}

void* PostProcess(void* arg) {
    while(1) {
        if(!msgOutQueue.Empty()) {
            MsgOut msgOut = msgOutQueue.Pop();
            if (msgOut.videoEnd) break;
            GetResult(msgOut.inferOutputs, msgOut.srcImg, modelWidth, modelHeight);
        }
        usleep(100);
    }
    LOG_PRINT("[INFO] *************** all get done ***************");
    exitFlag = true;
    return NULL;
}

int main(int argc, char *argv[]) {
    int32_t deviceId = 0;
    AclLiteResource aclResource(deviceId);
    bool ret = aclResource.Init();
    if(!ret) return 1;
    context = aclResource.GetContext();
    
    if(!Initparam(argc, argv)) return 1;
 
    pthread_t preTids, exeTids, posTids;
    pthread_create(&preTids, NULL, GetInput, (void*)&deviceId);
    pthread_create(&exeTids, NULL, ModelExecute, NULL);
    pthread_create(&posTids, NULL, PostProcess, NULL);

    pthread_detach(preTids);
    pthread_detach(exeTids);
    pthread_detach(posTids);
 
    while(!exitFlag) {
        sleep(10);
    }
    aclResource.Release();
    return 0;
}