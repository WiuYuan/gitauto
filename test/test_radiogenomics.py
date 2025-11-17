from src.services.llm import LLM
from src.services.logical_tree import LogicalTree
from src.utils.workflow import query_based_on_tool_calls, solve_goal_list
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
parser.add_argument("--verbose", type=bool, default=False, help="verbose")
args = parser.parse_args()

num = "radiogenomics_test"
verbose = args.verbose
print("num =", num, ",verbose =", verbose)

# llm = LLM(model_name="qwen3:8b")
with open(".webui_port", "r", encoding="utf-8") as f:
    port = int(f.read())
if verbose:
    ec = ExternalClient(port=port)
else:
    ec = None
llm = LLM(
    model_name="deepseek-chat",
    llm_url="https://api.deepseek.com/chat/completions",
    api_key="sk-8969134148cb48c88377e5eefc6322c0",
    format="openai",
    # proxies={
    #     "http": "socks5h://127.0.0.1:1081",
    #     "https": "socks5h://127.0.0.1:1081",
    # },
    ec=ec,
)
# llm = LLM(
#     model_name="/data/yuanwen/models/qwen3-32b",
#     llm_url="http://localhost:8000/v1/chat/completions",
#     api_key="",
#     format="openai",
#     ec=ec
# )
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

# print("A")
import os

ct = custom_tools(
    MAIN_DIR=f"/data/yuanwen/workspace/swe-solution/lite/{num}",
    WRITE_DIR=f"/data/yuanwen/workspace/swe-solution/lite/{num}",
    MAIN_DATA_DIR=f"/data/yuanwen/workspace/swe-solution/lite/{num}",
    PYTHON_PATH="python",
    PMC_URL="https://pmc.ncbi.nlm.nih.gov/articles/pmid/39218882/",
    # MATLAB_PATH="/Applications/MATLAB_R2023b.app/bin/matlab",
    LOCAL_TMP_PATH=f"/data/yuanwen/workspace/tmp/{num}",
    CHUNK_SIZE=5000,
    llm=llm,
    verbose=verbose,
)
os.makedirs(ct.MAIN_DIR, exist_ok=True)
from src.services.llm import load_messages, save_messages

LOG_DIR = f"/data/yuanwen/workspace/tmp/{num}"


package_name = "package"
tc = Tool_Calls(LOG_DIR=LOG_DIR, MAX_CHAR=40000)

tc.clear()

# launch_web()
os.environ["NO_PROXY"] = "*"
goal_list = [
    (
        "goal",
        "根据pmc文章信息, 复现所有你能复现的结果, 需要拿到所有的真实数据, 最后与文章结果比较",
    )
]
tools = [
    ct.func_cat,
    ct.func_write,
    ct.func_modify,
    ct.func_insert,
    ct.func_append,
    ct.func_prepend,
    ct.func_python,
    ct.func_cmd,
    ct.func_fetch_info_from_pmc_with_llm,
    ct.func_human,
    ct.func_guide,
    # ct.func_reflect,
    ct.add_child,
    ct.return_to_parent,
]
message = solve_goal_list(
    goal_list=goal_list,
    ct=ct,
    max_steps=1000,
    tc=tc,
    verbose=verbose,
    human_decision=False,
    tools=tools,
    human_check_before_calling=False,
)
