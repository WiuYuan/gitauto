import asyncio
import json
import threading
import time
import queue
import socket
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import websockets
import websocket
from websocket import WebSocketApp
import os


# ======================================
# Layer Definition
# ======================================
@dataclass
class Layer:
    category: str
    items: List[str] = field(default_factory=list)
    children: List["Layer"] = field(default_factory=list)


class LayerStack:
    """管理层级结构的类"""

    def __init__(self):
        self.root = Layer(category="root")
        self.stack = [self.root]
        self._last_category = None  # ✅ 记录上一次 category，用于判断是否同层拼接

    def add_layer_once(self, category: str, content: str = ""):
        new_layer = Layer(category=category or "(unnamed)")
        if content:
            new_layer.items.append(content)
        self.stack[-1].children.append(new_layer)

    def append_to_current(self, text: str):
        """在当前层追加文本到最后一个 item"""
        if not text:
            return
        if not self.stack[-1].items:
            self.stack[-1].items.append(text)
        else:
            self.stack[-1].items[-1] += text  # ✅ 拼接到同一个 item

    def apply(self, msg: Dict):
        cat = msg.get("category", "")
        content = msg.get("content", "")
        delta = int(msg.get("level_delta", 0))

        last_popped = None
        while delta < 0 and len(self.stack) > 1:
            last_popped = self.stack.pop()
            delta += 1

        if delta > 0:
            new_layer = Layer(category=cat or "(unnamed)")
            self.stack[-1].children.append(new_layer)
            self.stack.append(new_layer)
            if content:
                new_layer.items.append(content)
            self._last_category = cat
            return

        if content:
            target_layer = last_popped if last_popped else self.stack[-1]

            # ✅ 改进逻辑：
            # 1. 如果 category 没变，就拼接在最后
            # 2. 如果 category 改变或是第一次，就新建一条 item
            if self._last_category == cat and target_layer.items:
                target_layer.items[-1] += content
            else:
                target_layer.items.append(content)

            self._last_category = cat  # 更新最近的 category

    def to_dict(self, layer=None):
        if layer is None:
            layer = self.root
        return {
            "category": layer.category,
            "items": layer.items.copy(),
            "children": [self.to_dict(child) for child in layer.children],
        }


# ======================================
# ExternalServer
# ======================================
class ExternalServer:
    def __init__(self, port: Optional[int] = None, log_filepath: Optional[str] = None):
        self.port = port or self._find_free_port()
        self.connected_clients = set()
        self._incoming_queue = queue.Queue()
        self._request_queue = queue.Queue()
        self._send_queue = queue.Queue()
        self._stack = LayerStack()
        self._sender_thread_started = False
        self.log_filepath = log_filepath
        if log_filepath is not None:
            log_dir = os.path.dirname(self.log_filepath)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            with open(self.log_filepath, "w", encoding="utf-8") as f:
                pass  # 清空文件

        with open(".webui_port", "w") as f:
            f.write(str(self.port))
        print(f"🌐 ExternalServer initialized on ws://127.0.0.1:{self.port}")

    @staticmethod
    def _find_free_port(start_port=17860, end_port=17900):
        for port in range(start_port, end_port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("❌ No free port available!")

    async def _handle_client(self, websocket, path):
        self.connected_clients.add(websocket)
        print(f"📡 Client connected. Total: {len(self.connected_clients)}")

        try:
            async for message in websocket:
                if self.log_filepath is not None:
                    with open(self.log_filepath, "a", encoding="utf-8") as logf:
                        logf.write(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}\n"
                        )
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    self._incoming_queue.put(data)

                    if msg_type == "info":
                        payload = data.get("data", {})
                        self._stack.apply(payload)
                        self._request_queue.put(self._stack.to_dict())

                    elif msg_type == "choice_request":
                        await self._broadcast(data)

                    elif msg_type == "choice_response":
                        self._incoming_queue.put(data)

                    elif msg_type == "human_guidance":
                        # Store human guidance messages directly in incoming queue
                        self._incoming_queue.put(data)

                    elif msg_type == "queue_request":
                        msgs = list(self._incoming_queue.queue)

                        await self._broadcast(
                            {"type": "queue_response", "data": {"messages": msgs}}
                        )

                    elif msg_type == "queue_remove":
                        # 获取要删除的目标
                        target = data.get("data", {}).get("target", None)
                        removed = False

                        if target is not None:
                            temp = []
                            try:
                                while True:
                                    msg = self._incoming_queue.get_nowait()
                                    if not removed and msg == target:
                                        removed = True
                                        continue
                                    temp.append(msg)
                            except queue.Empty:
                                pass
                            for item in temp:
                                self._incoming_queue.put(item)
                        await self._broadcast(
                            {
                                "type": "queue_remove_ack",
                                "data": {"success": removed, "target": target},
                            }
                        )

                    elif msg_type == "clear_tree":
                        self._stack = LayerStack()
                        self._request_queue.queue.clear()
                        await self._broadcast(
                            {"type": "tree_update", "data": self._stack.to_dict()}
                        )

                except Exception as e:
                    print(f"⚠️ Error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            print("📡 Client disconnected")
        finally:
            self.connected_clients.discard(websocket)

    async def _broadcast(self, message: dict):
        if not self.connected_clients:
            return
        payload = json.dumps(message)
        for client in list(self.connected_clients):
            try:
                await client.send(payload)
            except websockets.exceptions.ConnectionClosed:
                self.connected_clients.discard(client)

    async def _stream_updates(self):
        last = None
        while True:
            try:
                data = self._request_queue.get(timeout=1)
                if data != last:
                    last = data
                    await self._broadcast({"type": "tree_update", "data": data})
            except queue.Empty:
                await asyncio.sleep(0.1)

    async def _server_main(self):
        server = await websockets.serve(self._handle_client, "0.0.0.0", self.port)
        print(f"🚀 WebSocket server started on ws://0.0.0.0:{self.port}")

        stream_task = asyncio.create_task(self._stream_updates())
        try:
            await asyncio.Future()
        finally:
            stream_task.cancel()
            server.close()
            await server.wait_closed()

    def launch_server(self):
        asyncio.run(self._server_main())

    def _sender_loop(self):
        ws_url = f"ws://127.0.0.1:{self.port}"

        def _connect():
            ws = WebSocketApp(ws_url)
            threading.Thread(target=ws.run_forever, daemon=True).start()
            time.sleep(0.3)
            return ws

        ws = _connect()
        while True:
            msg = self._send_queue.get()
            if msg is None:
                break
            try:
                if not ws.sock or not ws.sock.connected:
                    ws = _connect()
                ws.send(json.dumps(msg))
            except Exception as e:
                print(f"[ERROR] Send failed: {e}")
                time.sleep(0.5)

    def send_messages(self, message: dict):
        if not self._sender_thread_started:
            threading.Thread(target=self._sender_loop, daemon=True).start()
            self._sender_thread_started = True
        self._send_queue.put(message)

    def send_choice(self, question: str, options: List[str]) -> str:
        cid = str(uuid.uuid4())
        self.send_messages(
            {
                "type": "choice_request",
                "data": {"choiceId": cid, "question": question, "options": options},
            }
        )
        return cid

    def remove_message(self, target: Dict[str, Any]) -> bool:
        temp = []
        removed = False
        try:
            while True:
                msg = self._incoming_queue.get_nowait()
                if not removed and msg == target:
                    removed = True
                    continue
                temp.append(msg)
        except queue.Empty:
            pass
        for item in temp:
            self._incoming_queue.put(item)
        return removed