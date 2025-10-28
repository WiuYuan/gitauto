from src.services.llm import LLM
from src.services.logical_tree import LogicalTree
from src.utils.build_agent import query_based_on_tool_calls, fix_github_error
from typing import Callable, List, Dict, Any
from src.services.agents import Tool_Calls

# from webui import launch_web

# from src.services.agents import Agent, Tool
from src.services.custom_tools import custom_tools, clean_training_logs
import glob
import os

os.environ["NO_PROXY"] = "*"

# llm = LLM(model_name="qwen3:8b")

llm = LLM(
    model_name="deepseek-chat",
    llm_url="https://api.deepseek.com/chat/completions",
    api_key="sk-2332c3d16a8d4f4ba1b3503074ba04c5",
    format="openai",
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
    HOST_DIR="/Users/yuanwen/Desktop/Docker_Environment/intern2/3/agent/issue_fix",
    HOST_DATA_DIR="/Users/yuanwen/Desktop/Docker_Environment/intern2/3/data/issue_fix",
    MAIN_DIR="/workspace",
    MAIN_DATA_DIR="/data",
    PYTHON_PATH="python",
    MATLAB_PATH="/Applications/MATLAB_R2023b.app/bin/matlab",
    ENV_NAME="fix_github_issue",
    LOCAL_TMP_PATH="/Users/yuanwen/Desktop/Docker_Environment/intern2/3/tmp",
    REMOTE_TMP_PATH="/tmp",
    BASE_ENV="python:3.7",
    PMC_URL="https://pmc.ncbi.nlm.nih.gov/articles/PMC7567795",
    llm=llm,
)
from src.services.llm import load_messages, save_messages

tool_calls_path = (
    "/Users/yuanwen/Desktop/Docker_Environment/intern2/3/code/tool_calls_path.json"
)
env_tool_calls_path = "/workspace/tool_calls_path.json"

from datasets import load_from_disk

ds = load_from_disk(
    "/Users/yuanwen/Desktop/Docker_Environment/intern2/4/data/SWE-bench_Lite"
)
print(ds)
sample = ds[0]
package_name = "package"

tc = Tool_Calls(PATH=tool_calls_path, ENV_PATH=env_tool_calls_path, MAX_CHAR=50000)
tc.clear()

# launch_web()
message = fix_github_error(
    sample=sample,
    ct=ct,
    max_steps=10000,
    tc=tc,
    tree_filepath="/Users/yuanwen/Desktop/Docker_Environment/intern2/3/code/logical_tree_fix_issue.json",
    package_name=package_name,
    verbose=True,
    whether_recreate=False,
    save_filepath="/workspace/fix_issue.txt",
)
