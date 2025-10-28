#!/bin/bash
# ===============================
# Unified WebUI Launcher (with cleanup)
# ===============================

GREEN='\033[1;32m'
CYAN='\033[1;36m'
RED='\033[1;31m'
NC='\033[0m'

cd "$(dirname "$0")"

echo -e "${CYAN}🚀 Starting Python backend (webui.py)...${NC}"

# 启动后端后台运行，并记录 PID
rm -f .webui_port
nohup python run_backend.py >/dev/null 2>&1 &
BACKEND_PID=$!

# 注册退出钩子：脚本终止时自动清理后台进程
cleanup() {
  echo -e "${RED}\n🧹 Cleaning up backend process (PID: ${BACKEND_PID})...${NC}"
  kill -9 "$BACKEND_PID" 2>/dev/null
}
trap cleanup EXIT

# 等待端口文件生成
echo -e "${CYAN}⌛ Waiting for backend to select port...${NC}"
for ((i=1; i<=50; i++)); do
  if [ -f ".webui_port" ]; then
    WS_PORT=$(cat .webui_port | tr -d '\n\r ')

    # 判断是否为正整数（1~65535 端口范围）
    if [[ "$WS_PORT" =~ ^[0-9]+$ ]] && [ "$WS_PORT" -ge 1 ] && [ "$WS_PORT" -le 65535 ]; then
      break
    fi
  fi

  sleep 0.3
done

if [ -z "$WS_PORT" ]; then
  echo -e "${RED}❌ Failed to detect backend port (.webui_port not found)!${NC}"
  cleanup
  exit 1
fi

echo -e "${GREEN}✅ Backend running on port ${WS_PORT}${NC}"

if [ -d "react-webui" ]; then
  cd react-webui
else
  echo -e "${RED}❌ Cannot find react-webui/ folder!${NC}"
  cleanup
  exit 1
fi

echo -e "${CYAN}🌐 Starting React frontend with VITE_WS_PORT=${WS_PORT}...${NC}"
VITE_WS_PORT=$WS_PORT npm run dev