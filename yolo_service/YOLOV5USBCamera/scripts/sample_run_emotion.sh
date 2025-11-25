#!/bin/bash
# 情感识别版本运行脚本（与原版本共存）
ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"

function main()
{
  # 保持与原脚本一致的参数校验
  if [ "$1" != "imshow" ] && [ "$1" != "stdout" ]; then
      echo "[ERROR] please choose output display mode: [bash sample_run_emotion.sh imshow] [bash sample_run_emotion.sh stdout]"
      return
  fi

  echo "[INFO] Emotion sample starts to run"
  # 切换到情感版本专用输出目录（对应构建脚本的 out_emotion）
  cd ${ScriptPath}/../out_emotion
  # 运行情感版本的 main 程序
  ./main $1
  if [ $? -ne 0 ];then
      echo "[INFO] Emotion program runs failed"
  else
      echo "[INFO] Emotion program runs successfully"
  fi
}

main $1