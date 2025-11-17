#!/bin/bash
# ===============================
# Unified WebUI Launcher (Non-blocking)
# ===============================

GREEN='\033[1;32m'
CYAN='\033[1;36m'
RED='\033[1;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

cd "$(dirname "$0")"

echo -e "${CYAN}🚀 Starting Python backend (webui.py)...${NC}"

# 清理旧文件并启动后端
rm -f .webui_port
nohup /data/yuanwen/miniconda3/bin/python -u run_backend.py >/tmp/webui_backend.log 2>&1 &
BACKEND_PID=$!

# 注册退出钩子：脚本终止时自动清理后台进程
cleanup() {
  echo -e "${RED}\n🧹 Cleaning up processes...${NC}"
  kill -9 "$BACKEND_PID" 2>/dev/null
  kill -9 "$FRONT_PID" 2>/dev/null
}
trap cleanup EXIT

# 等待端口文件生成
echo -e "${CYAN}⌛ Waiting for backend to select port...${NC}"
for ((i=1; i<=50; i++)); do
  if [ -f ".webui_port" ]; then
    WS_PORT=$(tr -d '\n\r ' < .webui_port)
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

# 启动 React 前端
if [ -d "react-webui" ]; then
  cd react-webui
else
  echo -e "${RED}❌ Cannot find react-webui/ folder!${NC}"
  cleanup
  exit 1
fi

# --- ensure nvm is loaded ---
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use default >/dev/null 2>&1 || true
# -----------------------------

echo -e "${CYAN}🌐 Starting React frontend with VITE_WS_PORT=${WS_PORT}...${NC}"
nohup bash -c "VITE_WS_PORT=${WS_PORT} npm run dev -- --host 0.0.0.0 --port 4000" >/tmp/webui_frontend.log 2>&1 &
FRONT_PID=$!

sleep 1
echo -e "${GREEN}✅ Frontend running (PID: ${FRONT_PID})${NC}"
echo -e "${YELLOW}🌍 Access your app at: http://127.0.0.1:4000${NC}"
echo -e "${YELLOW}🔗 Backend WebSocket port: ${WS_PORT}${NC}"
echo -e "${CYAN}📜 Logs: tail -f /tmp/webui_backend.log /tmp/webui_frontend.log${NC}"

# 不阻塞：等待后台进程退出（Ctrl+C 可退出并清理）
wait
# VITE_WS_PORT=17860 npm run dev -- --host 0.0.0.0 --port 4000