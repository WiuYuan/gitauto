from src.services.llm import LLM
from src.services.logical_tree import LogicalTree
from src.utils.workflow import (
    query_based_on_tool_calls,
    fix_github_error,
    solve_goal_list,
)
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

verbose = args.verbose
print("verbose =", verbose)

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
    MAIN_DIR=f"/data/yuanwen/workspace/goal_test",
    WRITE_DIR=f"/data/yuanwen/workspace/goal_test",
    MAIN_DATA_DIR=f"/data/yuanwen/workspace/goal_test",
    PYTHON_PATH="python",
    # MATLAB_PATH="/Applications/MATLAB_R2023b.app/bin/matlab",
    LOCAL_TMP_PATH=f"/data/yuanwen/workspace/tmp/solve_goal",
    CHUNK_SIZE=5000,
    llm=llm,
    verbose=verbose,
)
os.makedirs(ct.MAIN_DIR, exist_ok=True)
from src.services.llm import load_messages, save_messages

LOG_DIR = f"/data/yuanwen/workspace/tmp/solve_goal"

# sample = build_sample_from_github(
#     repo_url="https://github.com/SWE-agent/test-repo",
#     issue_url="https://github.com/SWE-agent/test-repo/issues/1",
# )
# sample["instance_id"] = "test"

package_name = "package"
tc = Tool_Calls(LOG_DIR=LOG_DIR, MAX_CHAR=40000)

tc.clear()

# launch_web()
os.environ["NO_PROXY"] = "*"
goal_list = [
    (
        "goal",
        "Using the dataset at https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis, after downloading it into the current directory, please provide me with a detailed report. The report should include multiple figures (corresponding to any basic statistics you can think of, at least covering 10 different types/categories). The final report should be compiled in Markdown format, and please also attempt to generate a PDF version. Note that you have already completed part of this task before — you can refer to the current directory for what has been done so far.",
    )
]
message = solve_goal_list(
    goal_list=goal_list,
    ct=ct,
    max_steps=100,
    tc=tc,
    verbose=verbose,
    human_decision=False,
    tools=None,
    human_check_before_calling=False,
)
