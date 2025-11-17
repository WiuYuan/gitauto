from src.services.llm import LLM
from src.services.logical_tree import LogicalTree
from src.utils.workflow import query_based_on_tool_calls, fix_github_error
from typing import Callable, List, Dict, Any
from src.services.agents import Tool_Calls
from src.services.external_client import ExternalClient
import re

# from webui import launch_web

# from src.services.agents import Agent, Tool
from src.services.custom_tools import custom_tools, clean_training_logs
import glob
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int, default=0, help="输入数字参数")
args = parser.parse_args()

num = args.num
print("num =", num)

# llm = LLM(model_name="qwen3:8b")
with open(".webui_port", "r", encoding="utf-8") as f:
    port = int(f.read())
# ec = ExternalClient(port=port)
llm = LLM(
    model_name="/data/yuanwen/models/qwen3-32b",
    llm_url="http://localhost:8000/v1/chat/completions",
    api_key="",
    format="openai",
    # ec=ec,                       # 如有外部 WebUI 客户端可传入
)

# print("A")
import os
ct = custom_tools(
    MAIN_DIR=f"/data/yuanwen/workspace/swe-solution/lite/{num}",
    WRITE_DIR=f"/data/yuanwen/workspace/swe-solution/lite/{num}",
    MAIN_DATA_DIR=f"/data/yuanwen/workspace/swe-solution/lite/{num}",
    PYTHON_PATH="python",
    # MATLAB_PATH="/Applications/MATLAB_R2023b.app/bin/matlab",
    LOCAL_TMP_PATH=f"/data/yuanwen/workspace/tmp/{num}",
    CHUNK_SIZE=1000,
    llm=llm,
    verbose=False,
)
os.makedirs(ct.MAIN_DIR, exist_ok=True)
from src.services.llm import load_messages, save_messages

LOG_DIR = (
    f"/data/yuanwen/workspace/tmp/{num}"
)

package_name = "package"
tc = Tool_Calls(LOG_DIR=LOG_DIR, MAX_CHAR=10000)
tc.clear()

# launch_web()
os.environ["NO_PROXY"] = "*"
# message = fix_github_error(
#     sample=sample,
#     ct=ct,
#     max_steps=500,
#     tc=tc,
#     package_name=package_name,
#     verbose=False,
#     whether_recreate=False,
#     save_filepath=f"{ct.MAIN_DIR}/fix_issue.txt",
# )
print(llm.query("你是谁", verbose=False))
