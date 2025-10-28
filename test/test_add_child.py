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
    MAIN_DIR="/Users/yuanwen/Desktop/Docker_Environment/intern2/3/code",
    PYTHON_PATH="python",
    MATLAB_PATH="/Applications/MATLAB_R2023b.app/bin/matlab",
    LOCAL_TMP_PATH="/Users/yuanwen/Desktop/Docker_Environment/intern2/3/tmp",
    PMC_URL="https://pmc.ncbi.nlm.nih.gov/articles/PMC7567795",
    llm=llm,
)
from src.services.llm import load_messages, save_messages

tool_calls_path = (
    "/Users/yuanwen/Desktop/Docker_Environment/intern2/3/code/tool_calls_path.json"
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
goal_text = "测试一下helpadd_child功能, 返回功能, 结束任务功能"
message = solve_goal_list(
    goal_list=[("goal", goal_text)],
    ct=ct,
    max_steps=10000,
    tc=tc,
    tree_filepath="/Users/yuanwen/Desktop/Docker_Environment/intern2/3/code/logical_tree.json",
    verbose=True,
    human_decision=True,
    tools=tools,
    human_check_before_calling=True,
)
# ct.func_write("test.txt", "text")
