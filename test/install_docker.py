from src.services.llm import LLM
from src.services.logical_tree import LogicalTree
from src.utils.build_agent import solve_goal_list
from typing import Callable, List, Dict, Any
from src.services.agents import Tool_Calls

# from src.services.agents import Agent, Tool
from src.services.custom_tools import custom_tools, clean_training_logs
from src.services.external_client import ExternalClient
import glob
import os

os.environ["NO_PROXY"] = "*"

# llm = LLM(model_name="qwen3:8b")

with open(".webui_port", "r", encoding="utf-8") as f:
    port = int(f.read())
ec = ExternalClient(port=port)
llm = LLM(
    model_name="deepseek-chat",
    llm_url="https://api.deepseek.com/chat/completions",
    api_key="sk-2332c3d16a8d4f4ba1b3503074ba04c5",
    format="openai",
    ec=ec,
)
# llm = LLM(
#     model_name="gpt-5-2025-08-07",
#     llm_url="https://api.aimlapi.com/v1/chat/completions",
#     api_key="9ce046a9681446c48427b3fe4dd7cdd4",
#     format="openai",
# )
# llm = LLM(
#     model_name="gpt-4o",
#     llm_url="https://api.aimlapi.com/v1/chat/completions",
#     api_key="9ce046a9681446c48427b3fe4dd7cdd4",
#     format="openai",
# )


ct = custom_tools(
    MAIN_DIR="/data/wyuan/workspace/agent_pro/tmp",
    PYTHON_PATH="python",
    MATLAB_PATH="/Applications/MATLAB_R2023b.app/bin/matlab",
    LOCAL_TMP_PATH="/data/wyuan/workspace/agent_pro/tmp",
    PMC_URL="https://pmc.ncbi.nlm.nih.gov/articles/PMC7567795",
    llm=llm,
)
from src.services.llm import load_messages, save_messages

tool_calls_path = (
    "/data/wyuan/workspace/agent_pro/tmp/tool_calls_path.json"
)
env_tool_calls_path = "/workspace/tool_calls_path.json"

tc = Tool_Calls(PATH=tool_calls_path, ENV_PATH=env_tool_calls_path, MAX_CHAR=50000)
tc.clear()

tools = [
    ct.func_ls,
    ct.func_cat,
    # ct.func_cat_with_llm,
    ct.func_write,
    ct.func_modify,
    ct.func_insert,
    ct.func_append,
    ct.func_prepend,
    ct.func_python,
    ct.func_cmd,
    ct.func_matlab,
    # ct.func_fetch_info_from_pmc_with_llm,
    ct.func_human,
    ct.func_guide,
    # ct.func_think,
    ct.add_child,
    ct.return_to_parent,
]
goal_text = "请你在HOME目录下安装podman, 注意我没有sudo权限, 你中途的所有下载文件请放在MAIN_DIR下, 只有最后解压后的才能放在HOME下, 最后请在.bashrc写好PATH, 同时直接把docker等价于podman, 我原来尝试过安装docker由于sudo失败, 所以有docker已经安装, 但不管他, 以及注意任何代码需要有时间限制, 不能一直跑, 比如下载的时候, 以及下载的时候参考curl -I --proxy socks5h://127.0.0.1:1080 https://www.google.com 来使用端口, 我已经下载好了podman-desktop-1.22.1.tar.gz在MAIN_DIR下"
message = solve_goal_list(
    goal_list=[("goal", goal_text)],
    ct=ct,
    max_steps=10000,
    tc=tc,
    tree_filepath="/data/wyuan/workspace/agent_pro/tmp/logical_tree.json",
    verbose=True,
    human_decision=False,
    tools=tools,
    human_check_before_calling=False,
)
# ct.func_write("test.txt", "text")
