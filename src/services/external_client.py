import json
import threading
import time
import websocket
import os
import uuid
from typing import List, Dict, Any, Optional, Callable


class ExternalClient:
    """
    ExternalClient
    ----------------
    Persistent WebSocket client for real-time communication with ExternalServer.

    ✅ Features:
    - Auto-connect (port read from .webui_port)
    - Continuous background listening (callback-based)
    - Auto-reconnect on disconnect
    - Supports send_message() / send_info() / get_messages()
    """

    def __init__(
        self,
        port: Optional[int] = None,
        auto_start: bool = True,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
        task_id: Optional[str] = None,
    ):
        self.port = port
        self.ws_url = f"ws://127.0.0.1:{self.port}"

        self._running = False
        self._thread = None
        self._ws = None
        # self._on_message = on_message or (lambda msg: print(f"📩 {msg}"))
        self._on_message = on_message or (lambda msg: None)
        if task_id is None:
            self.task_id = str(uuid.uuid4())
        else:
            self.task_id = task_id

        print(f"🔗 ExternalClient initialized — {self.ws_url}")

        if auto_start:
            self.start_listening()

    # -----------------------------------------------------
    # Core Connection
    # -----------------------------------------------------
    def _connect_forever(self):
        """持续监听服务器消息"""

        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._on_message(data)  # 实时回调
            except Exception as e:
                print(f"⚠️ Failed to parse message: {e}")

        def on_error(ws, error):
            print(f"⚠️ WebSocket error: {error}")

        def on_close(ws, *_):
            print("❌ Disconnected from server, retrying in 2s...")
            self._running = False
            time.sleep(2)
            self.start_listening()  # 自动重连

        def on_open(ws):
            self._ws = ws
            print("✅ Connected to ExternalServer")

        ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open,
        )

        ws.run_forever(ping_interval=20, ping_timeout=10)

    def start_listening(self, wait_until_connected: bool = True, timeout: float = 5.0):
        """后台启动监听线程，并可等待直到连接成功"""
        if self._running:
            return

        self._running = True
        self._connected_event = threading.Event()

        def _connect_forever_with_signal():
            """监听服务器消息，连接成功后触发事件"""

            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self._on_message(data)
                except Exception as e:
                    print(f"⚠️ Failed to parse message: {e}")

            def on_error(ws, error):
                print(f"⚠️ WebSocket error: {error}")

            def on_close(ws, *_):
                print("❌ Disconnected from server, retrying in 2s...")
                self._connected_event.clear()
                self._running = False
                time.sleep(2)
                self.start_listening(wait_until_connected=False)  # 自动重连

            def on_open(ws):
                print("✅ Connected to ExternalServer")
                self._ws = ws
                self._connected_event.set()  # ✅ 通知主线程已连接

            ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)

        # 启动后台线程
        self._thread = threading.Thread(
            target=_connect_forever_with_signal, daemon=True
        )
        self._thread.start()

        # ✅ 阻塞等待连接建立
        if wait_until_connected:
            connected = self._connected_event.wait(timeout=timeout)
            if connected:
                print("🔗 ExternalClient: Connection confirmed ready.")
            else:
                print("⚠️ Timeout: WebSocket connection not established in time.")

    # -----------------------------------------------------
    # Sending Messages
    # -----------------------------------------------------
    def send_message(self, message: Dict[str, Any]):
        if message.get("type") == "info":
            data = message.get("data", {})
            category = data.get("category", "")
            delta = data.get("level_delta", None)

            # ✅ 对 delta=0 的情况进行特殊处理
            if delta == 0 and len(category) != 0:
                enter_msg = {
                    "type": "info",
                    "task_id": self.task_id,
                    "data": {
                        **data,
                        "level_delta": +1,
                    },
                }
                self._send_message(enter_msg)
                exit_msg = {
                    "type": "info",
                    "task_id": self.task_id,
                    "data": {
                        **data,
                        "level_delta": -1,
                        "content": "",  # 不带内容，只控制层级
                    },
                }
                self._send_message(exit_msg)
                return
        message["task_id"] = self.task_id
        self._send_message(message=message)

    def _send_message(self, message: Dict[str, Any]):
        """通过已建立的 WebSocket 长连接发送消息"""
        try:
            if not self._ws or not self._ws.sock or not self._ws.sock.connected:
                print("⚠️ WebSocket not connected — message dropped.")
                return
            self._ws.send(json.dumps(message))
            # print(f"📤 Sent: {message}")
        except Exception as e:
            print(f"⚠️ Failed to send message: {e}")

    def send_choice(self, question: str, options: List[str]) -> str:
        cid = str(uuid.uuid4())
        message = {
            "type": "choice_request",
            "task_id": self.task_id,
            "data": {"choiceId": cid, "question": question, "options": options},
        }
        self.send_message(message=message)
        return cid

    # -----------------------------------------------------
    # Request messages from server
    # -----------------------------------------------------
    def get_messages(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        主动从 ExternalServer 请求 _incoming_queue 内容。
        实现逻辑：
        1. 通过当前长连接发送 {"type": "queue_request"}
        2. 临时监听 queue_response
        3. 收到后返回 data["messages"]
        """
        if not self._ws or not self._ws.sock or not self._ws.sock.connected:
            print("⚠️ WebSocket not connected — cannot request messages.")
            return []

        result: List[Dict[str, Any]] = []
        done = threading.Event()

        # 1️⃣ 临时回调
        def temp_handler(data: Dict[str, Any]):
            if data.get("type") == "queue_response":
                result.extend(data.get("data", {}).get("messages", []))
                done.set()  # 收到结果后解除阻塞

        # 2️⃣ 替换回调（保存旧的）
        old_handler = self._on_message
        self._on_message = temp_handler

        # 3️⃣ 发送请求
        try:
            payload = {"type": "queue_request", "task_id": self.task_id, "data": {}}
            self._ws.send(json.dumps(payload))
            # print("📤 Sent queue_request to server")
        except Exception as e:
            print(f"⚠️ Failed to send queue_request: {e}")
            self._on_message = old_handler
            return []

        # 4️⃣ 等待响应（阻塞）
        done.wait(timeout=timeout)

        # 5️⃣ 恢复回调
        self._on_message = old_handler

        # 6️⃣ 输出结果
        # if result:
        #     print(f"📥 Retrieved {len(result)} messages from server")
        # else:
        #     print("⚠️ No queue_response received within timeout.")

        return result

    def get_choice_response(
        self, question: str, options: List[str], timeout: Optional[float] = None
    ) -> str:
        cid = self.send_choice(question=question, options=options)
        response = self.get_choice_response_from_choice_id(
            choice_id=cid, timeout=timeout
        )
        self.remove_message(message=response)
        return response.get("data").get("optionText")

    def get_choice_response_from_choice_id(
        self, choice_id: str, timeout: Optional[float] = None
    ):
        """Wait for frontend response with timeout"""
        start = time.time()
        while timeout is None or time.time() - start < timeout:
            msgs = self.get_messages()
            for msg in msgs:
                if msg.get("type") == "choice_response":
                    data = msg.get("data", {})
                    if data.get("choiceId") == choice_id:
                        return msg
            time.sleep(0.2)
        return None

    def remove_message(self, message: Dict[str, Any]):
        self.send_message({"type": "queue_remove", "task_id": self.task_id, "data": {"target": message}})
        return

    def get_all_guidance(self, timeout: Optional[float] = None) -> str:
        """
        获取所有 human_guidance 消息并组合成一个字符串
        """
        messages = self.get_messages(timeout=timeout)
        guidance_messages = []

        for msg in messages:
            if msg.get("type") == "human_guidance":
                content = msg.get("data", {}).get("content", "")
                if content:
                    guidance_messages.append(content)
                self.remove_message(msg)

        # 将所有 guidance 组合成一个字符串，用换行符分隔
        combined_guidance = "\n".join(guidance_messages)
        return combined_guidance
