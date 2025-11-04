from src.services.llm import LLM
from src.services.logical_tree import LogicalTree
from src.utils.workflow import solve_goal_list
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
    MAIN_DIR="/data/yuanwen/workspace/gitauto",
    PYTHON_PATH="python",
    MATLAB_PATH="/Applications/MATLAB_R2023b.app/bin/matlab",
    LOCAL_TMP_PATH="/data/yuanwen/workspace/tmp",
    PMC_URL="https://pmc.ncbi.nlm.nih.gov/articles/PMC7567795",
    llm=llm,
)
from src.services.llm import load_messages, save_messages

tool_calls_path = (
    "/data/yuanwen/workspace/tmp/tool_calls_path.json"
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
goal_text = "参考src/services/external_server.py的task_id, 每个task_id对应一个layer树结构, 修改react前端部分显示采用task_id分治的方法，特别的如果一个任务没有task_id, 就当作default"
# goal_text = "在test/deal_with_basic_problem.py下, 给出了一个输入一个目标, 走对应流程, 但是我现在想把它做到react部分的网页端口, 作为长期任务, 以及这里的tool_call也需要做一定的持久化, 比如查询以前任务这种, 请你先跟我讨论一下怎么做, 等我决定方式之后再开始"
message = solve_goal_list(
    goal_list=[("goal", goal_text)],
    ct=ct,
    max_steps=10000,
    tc=tc,
    tree_filepath="/data/yuanwen/workspace/tmp/logical_tree.json",
    verbose=True,
    human_decision=False,
    tools=tools,
    human_check_before_calling=False,
)
# ct.func_write("test.txt", "text")
