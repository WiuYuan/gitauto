import time
from src.services.external_client import ExternalClient

with open(".webui_port", "r", encoding="utf-8") as f:
    port = int(f.read())
es = ExternalClient(port=port)
es.send_message(
    {
        "type": "info",
        "data": {"category": "Final", "content": "Final", "level_delta": 0},
    }
)
# 启动 WebUI（后台线程）
# time.sleep(2)

# === 外层：主 LLM 层 ===
es.send_message(
    {
        "type": "info",
        "data": {
            "category": "LLM",
            "content": "Main LLM Start",
            "level_delta": 1,  # 进入层A
        },
    }
)
time.sleep(0.2)

# 输出几条内容
for token in ["[A] Step 1", "[A] Step 2"]:
    es.send_message(
        {
            "type": "info",
            "data": {"category": "", "content": token, "level_delta": 0},
        }
    )
    time.sleep(0.2)

# === 内层：模拟 help_add_child 调用 ===
es.send_message(
    {
        "type": "info",
        "data": {
            "category": "LLM",
            "content": "help_add_child Start",
            "level_delta": 1,  # 进入层B（嵌套）
        },
    }
)
time.sleep(0.2)

for token in ["[B] inner token 1", "[B] inner token 2"]:
    es.send_message(
        {
            "type": "info",
            "data": {"category": "", "content": token, "level_delta": 0},
        }
    )
    time.sleep(0.2)

# ⚠️ 问题点：内层提前关闭层
es.send_message(
    {
        "type": "info",
        "data": {
            "category": "",
            "content": "[B] inner finished",
            "level_delta": -1,  # 离开层B
        },
    }
)
time.sleep(0.2)

# 外层继续输出（此时理论上还在层A中）
# ❌ 实际在现有 LayerStack 逻辑下，这会错位到 root
es.send_message(
    {
        "type": "info",
        "data": {"category": "", "content": "[A] Step 3 after inner", "level_delta": 0},
    }
)
time.sleep(0.2)

# 外层结束层级
es.send_message(
    {
        "type": "info",
        "data": {"category": "", "content": "", "level_delta": -1},
    }
)

# es.send_message(
#     {
#         "type": "test",
#         "data": {"category": "", "content": "", "level_delta": 0},
#     }
# )

es.send_message(
    {
        "type": "info",
        "data": {"category": "Final", "content": "Final", "level_delta": 0},
    }
)
