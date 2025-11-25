#!/bin/bash
# 情感识别版本构建脚本，与原有版本共存
ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"

# 情感版本专用目录（与原有版本区分）
EmotionOutDir="${ScriptPath}/../out_emotion"
EmotionBuildDir="${ScriptPath}/../build_emotion/intermediates/host"

function build()
{
  # 清理情感版本旧文件（不影响原有版本）
  if [ -d ${EmotionOutDir} ];then
    rm -rf ${EmotionOutDir}
  fi

  if [ -d ${EmotionBuildDir} ];then
    rm -rf ${EmotionBuildDir}
  fi

  # 创建情感版本专用构建目录
  mkdir -p ${EmotionBuildDir}
  cd ${EmotionBuildDir}

  # 编译情感识别版本（假设源码在src_emotion目录，若与原有共用src可保持路径不变）
  # 注意：若情感模型代码在单独目录，需修改此处路径（如../../../src_emotion）
  cmake ../../../src_emotion -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE \
        -DCMAKE_INSTALL_PREFIX=${EmotionOutDir}  # 指定安装到情感专用输出目录
  if [ $? -ne 0 ];then
    echo "[ERROR] emotion version cmake error, Please check your environment!"
    return 1
  fi

  make -j4
  if [ $? -ne 0 ];then
    echo "[ERROR] emotion version build failed, Please check your environment!"
    return 1
  fi

  # 安装到情感专用输出目录
  make install
  if [ $? -ne 0 ];then
    echo "[ERROR] emotion version install failed!"
    return 1
  fi

  cd - > /dev/null
}

function main()
{
  echo "[INFO] Emotion sample preparation"
  build
  if [ $? -ne 0 ];then
    return 1
  fi
  echo "[INFO] Emotion sample preparation is complete. Output: ${EmotionOutDir}"
}

main